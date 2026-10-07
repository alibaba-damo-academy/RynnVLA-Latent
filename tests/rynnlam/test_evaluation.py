"""CPU-only synthetic extraction/probe contract tests; no weights or LARY."""

import argparse
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
from _scripts import load_script

evaluate = load_script("evaluate_rynnlam")
extraction = load_script("stream_extract_tar")


class MockEncoder:
    checkpoint_sha256 = "a" * 64
    config = {"k_token_source": "features", "num_k_tokens": 2, "embed_dim": 2048}

    def __init__(self):
        self.calls = 0

    def __call__(self, images, representation="ktoken"):
        self.calls += 1
        assert not torch.is_grad_enabled()
        assert images.shape[1:] == (2, 238, 322, 3)
        assert images.min() >= 0 and images.max() <= 1
        return torch.ones(images.shape[0], 2, 2048)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.archives = self.root / "archives"
        self.archives.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def fixture(
        self, dataset="calvin", count=2, missing_action=False, absent_row=False
    ):
        rows = []
        stride = extraction.DATASETS[dataset]["stride"]
        dim = extraction.DATASETS[dataset]["action_dim"]
        image = cv2.imencode(".png", np.full((10, 15, 3), 120, np.uint8))[1].tobytes()
        for i in range(count):
            row = {
                "src_img": f"images/{i}_src.png",
                "tgt_img": f"images/{i}_tgt.png",
                "action": f"actions/{i}.npy",
                "robot_type": "robot_a",
            }
            rows.append(row)
            action = io.BytesIO()
            np.save(action, np.full((stride, dim), i, np.float32))
            with tarfile.open(self.archives / f"part{i}.tar.gz", "w:gz") as tar:
                members = [(row["src_img"], image), (row["tgt_img"], image)]
                if not missing_action:
                    members.append((row["action"], action.getvalue()))
                for name, raw in members:
                    info = tarfile.TarInfo(name)
                    info.size = len(raw)
                    tar.addfile(info, io.BytesIO(raw))
        if absent_row:
            rows.append(dict(rows[0], src_img="absent.png"))
        metadata = self.root / "metadata.csv"
        with open(metadata, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return metadata

    def args(self, metadata, dataset="calvin", **kwargs):
        args = extraction.build_parser().parse_args(
            [
                "--dataset",
                dataset,
                "--split",
                "train",
                "--csv",
                str(metadata),
                "--tar-dir",
                str(self.archives),
                "--tar-glob",
                "*.tar.gz",
                "--checkpoint",
                str(self.root / "mock.pt"),
                "--output-root",
                str(self.root / "outputs"),
                "--device",
                "cpu",
                "--precision",
                "fp32",
                "--batch-size",
                "2",
            ]
        )
        for key, value in kwargs.items():
            setattr(args, key, value)
        return args

    def test_all_four_datasets_and_real_ktoken_dimensions(self):
        for dataset in extraction.DATASETS:
            with self.subTest(dataset=dataset):
                metadata = self.fixture(dataset)
                encoder = MockEncoder()
                report_path = extraction.extract(self.args(metadata, dataset), encoder)
                report, rows, _ = evaluate.load_report(report_path)
                self.assertEqual(len(rows), 2)
                self.assertEqual(report["flat_feature_dim"], 4096)
                self.assertEqual(report["protocol"]["normalization"], "imagenet")
                self.assertEqual(report["protocol"]["height"], 238)
                with np.load(rows[0]["la_path"], allow_pickle=False) as features:
                    self.assertEqual(features["tokens"].shape, (2, 2048))

    def test_resume_binds_checkpoint_protocol_and_action(self):
        metadata = self.fixture()
        encoder = MockEncoder()
        args = self.args(metadata, resume=True)
        report = extraction.extract(args, encoder)
        before = encoder.calls
        self.assertEqual(report, extraction.extract(args, encoder))
        self.assertEqual(before, encoder.calls)
        _, rows, _ = evaluate.load_report(report)
        Path(rows[0]["action"]).write_bytes(b"broken")
        extraction.extract(args, encoder)
        self.assertGreater(encoder.calls, before)
        other = MockEncoder()
        other.checkpoint_sha256 = "b" * 64
        self.assertNotEqual(report, extraction.extract(args, other))
        args.normalization = "none"
        self.assertNotEqual(report, extraction.extract(args, encoder))

    def test_shards_only_own_rows_and_merge_checks_coverage(self):
        metadata = self.fixture()
        reports = [
            extraction.extract(
                self.args(metadata, shard=i, num_shards=2, resume=True), MockEncoder()
            )
            for i in range(2)
        ]
        for path in reports:
            report, rows, _ = evaluate.load_report(path, ("shard_complete",))
            self.assertEqual(len(rows), 1)
            self.assertEqual(report["owned_rows"], report["completed_rows"])
        merged = evaluate.merge(
            argparse.Namespace(reports=reports, output_dir=self.root / "merged")
        )
        self.assertEqual(len(evaluate.load_report(merged)[1]), 2)
        with self.assertRaises(ValueError):
            evaluate.merge(
                argparse.Namespace(reports=reports[:1], output_dir=self.root / "bad")
            )
        metadata = self.fixture(absent_row=True)
        reports = [
            extraction.extract(
                self.args(metadata, shard=i, num_shards=2), MockEncoder()
            )
            for i in range(2)
        ]
        with self.assertRaisesRegex(ValueError, "missing from all archives"):
            evaluate.merge(
                argparse.Namespace(reports=reports, output_dir=self.root / "missing")
            )

    def test_missing_data_never_publishes_valid_csv(self):
        metadata = self.fixture(missing_action=True)
        with self.assertRaisesRegex(RuntimeError, "Incomplete extraction"):
            extraction.extract(self.args(metadata), MockEncoder())
        self.assertFalse(list((self.root / "outputs").rglob("train_la.csv")))
        reports = list((self.root / "outputs").rglob("report.json"))
        self.assertEqual(json.loads(reports[0].read_text())["status"], "failed")

    def test_absent_source_is_failure(self):
        metadata = self.fixture(absent_row=True)
        with self.assertRaises(RuntimeError):
            extraction.extract(self.args(metadata), MockEncoder())

    def test_pool_mean_uses_actual_feature_dimension(self):
        result = extraction.reduce_tokens(torch.ones(3, 2, 2048), "mean")
        self.assertEqual(result.shape, (3, 2048))

    def test_regression_parameters_and_robot_stats(self):
        metadata = self.fixture("agibot")
        train_path = extraction.extract(self.args(metadata, "agibot"), MockEncoder())
        val_path = extraction.extract(
            self.args(metadata, "agibot", split="val"), MockEncoder()
        )
        lary = self.root / "official"
        (lary / "regression").mkdir(parents=True)
        (lary / "regression" / "main.py").write_text(
            "raise RuntimeError('must not import during dry run')"
        )
        args = evaluate.build_parser().parse_args(
            [
                "regress",
                "--lary-root",
                str(lary),
                "--train-report",
                str(train_path),
                "--val-report",
                str(val_path),
                "--output-root",
                str(self.root / "probes"),
                "--dry-run",
            ]
        )
        command = evaluate.regress(args)
        self.assertEqual(command[command.index("--dataset") + 1], "agibotbeta")
        self.assertEqual(command[command.index("--stride") + 1], "45")
        self.assertIn("--global_stats_json", command)
        self.assertNotIn("--feat_slice", command)
        self.assertNotIn("--pool", command)
        train_rows = evaluate.load_report(train_path)[1]
        stats = evaluate.robot_stats(train_rows, train_rows, 16)
        self.assertEqual(len(stats["robot_stats"]["robot_a"]["mean"]), 16)
        with self.assertRaisesRegex(ValueError, "lack train statistics"):
            evaluate.robot_stats(
                train_rows, [dict(train_rows[0], robot_type="unknown")], 16
            )

    def test_external_regression_requires_final_epoch_and_success(self):
        metadata = self.fixture()
        train = extraction.extract(self.args(metadata), MockEncoder())
        val = extraction.extract(self.args(metadata, split="val"), MockEncoder())
        lary = self.root / "lary"
        (lary / "regression").mkdir(parents=True)
        entry = lary / "regression" / "main.py"
        entry.write_text("print('Epoch 30 | Train Loss: 0.2 | Val Seen MSE: 0.125')")
        args = evaluate.build_parser().parse_args(
            [
                "regress",
                "--lary-root",
                str(lary),
                "--train-report",
                str(train),
                "--val-report",
                str(val),
                "--output-root",
                str(self.root / "probes"),
            ]
        )
        result = evaluate.regress(args)
        self.assertEqual(json.loads(result.read_text())["best_val_seen_mse"], 0.125)
        entry.write_text("print('Epoch 1 | Train Loss: 0.2 | Val Seen MSE: 0.1')")
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            evaluate.regress(args)
        entry.write_text(
            "print('Epoch 30 | Train Loss: 0.2 | Val Seen MSE: 0.1'); raise SystemExit(1)"
        )
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            evaluate.regress(args)

    def test_regression_rejects_corrupt_metrics_and_missing_final_validation(self):
        metadata = self.fixture()
        train = extraction.extract(self.args(metadata), MockEncoder())
        val = extraction.extract(self.args(metadata, split="val"), MockEncoder())
        lary = self.root / "lary"
        (lary / "regression").mkdir(parents=True)
        entry = lary / "regression" / "main.py"
        args = evaluate.build_parser().parse_args(
            [
                "regress",
                "--lary-root",
                str(lary),
                "--train-report",
                str(train),
                "--val-report",
                str(val),
                "--output-root",
                str(self.root / "probes"),
            ]
        )
        logs = [
            f"Epoch 1 | Train Loss: 0.2 | Val Seen MSE: 0.1\n"
            f"Epoch 30 | Train Loss: 0.2 | Val Seen MSE: {metric}"
            for metric in ("nan", "NaN", "inf", "-inf", "0.12oops", "1e", "")
        ]
        logs += [
            "Epoch 1 | Train Loss: 0.2 | Val Seen MSE: 0.1\nEpoch 30 | Train Loss: 0.2",
            "Epoch 1 | Train Loss: 0.2 | Val Seen MSE: nan\n"
            "Epoch 30 | Train Loss: 0.2 | Val Seen MSE: 0.1",
        ]
        for text in logs:
            with self.subTest(log=text):
                entry.write_text(f"print({text!r})")
                with self.assertRaisesRegex(RuntimeError, "incomplete"):
                    evaluate.regress(args)
        for result in (self.root / "probes").rglob("result.json"):
            self.assertEqual(json.loads(result.read_text())["status"], "failed")

    def test_regression_rejects_swapped_artifacts_changed_actions_and_shapes(self):
        metadata = self.fixture()
        train = extraction.extract(self.args(metadata), MockEncoder())
        val = extraction.extract(self.args(metadata, split="val"), MockEncoder())
        _, rows, _ = evaluate.load_report(train)
        token = Path(rows[0]["la_path"])
        action = Path(rows[0]["action"])
        original_token, original_action = token.read_bytes(), action.read_bytes()
        args = evaluate.build_parser().parse_args(
            [
                "regress",
                "--lary-root",
                str(self.root / "unused"),
                "--train-report",
                str(train),
                "--val-report",
                str(val),
                "--output-root",
                str(self.root / "probes"),
                "--dry-run",
            ]
        )
        token.write_bytes(Path(rows[1]["la_path"]).read_bytes())
        with self.assertRaisesRegex(ValueError, "binding mismatch"):
            evaluate.regress(args)
        token.write_bytes(Path(evaluate.load_report(val)[1][0]["la_path"]).read_bytes())
        with self.assertRaisesRegex(ValueError, "binding mismatch"):
            evaluate.regress(args)
        token.write_bytes(original_token)
        np.save(action, np.ones((5, 7), np.float32))
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            evaluate.regress(args)
        action.unlink()
        with self.assertRaisesRegex(ValueError, "Missing absolute"):
            evaluate.regress(args)
        action.write_bytes(original_action)
        with np.load(token) as data:
            changed = {key: data[key] for key in data.files}
        changed["tokens"] = np.ones((1, 4096), np.float32)
        np.savez(token, **changed)
        with self.assertRaisesRegex(ValueError, "Token shape"):
            evaluate.regress(args)

    def test_implementation_fingerprint_changes_extraction_protocol(self):
        metadata = self.fixture()
        args = self.args(metadata, resume=True)
        with mock.patch(
            "rynnlam.inference.implementation_sha256", return_value="c" * 64
        ):
            first = extraction.extract(args, MockEncoder())
        with mock.patch(
            "rynnlam.inference.implementation_sha256", return_value="d" * 64
        ):
            second = extraction.extract(args, MockEncoder())
        self.assertNotEqual(first, second)
        report = evaluate.load_report(second)[0]
        self.assertEqual(report["protocol"]["implementation_sha256"], "d" * 64)

    def test_compressed_extraction_preserves_metadata_order_and_numeric_outputs(self):
        metadata = self.fixture(count=3)
        with open(metadata, newline="") as handle:
            rows = list(csv.DictReader(handle))
        rows[2]["src_img"] = rows[0]["src_img"]
        extraction.write_csv(metadata, rows, list(rows[0]))
        packed = self.archives / "packed.tar.gz"
        with tarfile.open(packed, "w:gz") as target:
            for part in reversed(sorted(self.archives.glob("part*.tar.gz"))):
                with tarfile.open(part) as source:
                    for member in reversed(source.getmembers()):
                        target.addfile(member, source.extractfile(member))
        scratch = self.root / "scratch"
        scratch.mkdir()
        with mock.patch.object(
            extraction, "validate_action", wraps=extraction.validate_action
        ) as validate:
            report_path = extraction.extract(
                self.args(
                    metadata, tar_glob="packed.tar.gz", local_temp_dir=str(scratch)
                ),
                MockEncoder(),
            )
        observed = [
            np.load(io.BytesIO(call.args[0]))[0, 0] for call in validate.call_args_list
        ]
        self.assertEqual(observed, [0, 1, 2])
        report, output_rows, _ = evaluate.load_report(report_path)
        self.assertEqual(report["completed_rows"], [0, 1, 2])
        for i, row in enumerate(output_rows):
            self.assertEqual(int(row["rynnlam_row_id"]), i)
            np.testing.assert_array_equal(
                np.load(row["action"]), np.full((5, 7), i, np.float32)
            )
            with np.load(row["la_path"]) as data:
                np.testing.assert_array_equal(
                    data["tokens"], np.ones((2, 2048), np.float32)
                )
        self.assertEqual(list(scratch.iterdir()), [])

    def test_compressed_archives_use_one_local_scratch_and_safe_member_reads(self):
        scratch_dir = self.root / "scratch"
        scratch_dir.mkdir()
        for compression in ("gz", "bz2", "xz"):
            with self.subTest(compression=compression):
                path = self.archives / f"sample.{compression}"
                with tarfile.open(path, f"w:{compression}") as archive:
                    for name, raw in (("../../escape", b"first"), ("later", b"second")):
                        info = tarfile.TarInfo(name)
                        info.size = len(raw)
                        archive.addfile(info, io.BytesIO(raw))
                handles = []
                temporary_file = tempfile.TemporaryFile

                def track(*args, **kwargs):
                    self.assertEqual(kwargs["dir"], scratch_dir)
                    handle = temporary_file(*args, **kwargs)
                    handles.append(handle)
                    return handle

                with mock.patch.object(
                    extraction.tempfile, "TemporaryFile", side_effect=track
                ):
                    with extraction.indexed_archive(path, scratch_dir) as archive:
                        self.assertIs(archive.fileobj, handles[0])
                        members = archive.getmembers()
                        self.assertEqual(
                            archive.extractfile(members[1]).read(), b"second"
                        )
                        self.assertEqual(
                            archive.extractfile(members[0]).read(), b"first"
                        )
                self.assertEqual(len(handles), 1)
                self.assertTrue(handles[0].closed)
                self.assertEqual(list(scratch_dir.iterdir()), [])
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    with extraction.indexed_archive(path, scratch_dir):
                        raise RuntimeError("interrupted")
                self.assertEqual(list(scratch_dir.iterdir()), [])
                self.assertFalse((self.root / "escape").exists())

    def test_atomic_output_copies_from_seekable_scratch(self):
        output = self.root / "array.npz"
        extraction.atomic_write(output, lambda f: np.savez(f, tokens=np.ones((2, 3))))
        with np.load(output) as data:
            self.assertEqual(data["tokens"].shape, (2, 3))

        def fail(handle):
            handle.write(b"partial")
            raise RuntimeError("interrupted")

        before = output.read_bytes()
        with self.assertRaises(RuntimeError):
            extraction.atomic_write(output, fail)
        self.assertEqual(output.read_bytes(), before)
        self.assertFalse(list(self.root.glob("*.partial")))

    def test_cli_help_no_weights_or_lary(self):
        for command in (
            [sys.executable, str(SCRIPTS / "stream_extract_tar.py"), "--help"],
            [sys.executable, str(SCRIPTS / "evaluate_rynnlam.py"), "--help"],
            [sys.executable, str(SCRIPTS / "evaluate_rynnlam.py"), "extract", "--help"],
            ["bash", str(SCRIPTS / "evaluate_rynnlam.sh"), "regress", "--help"],
        ):
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("usage:", result.stdout)


if __name__ == "__main__":
    unittest.main()
