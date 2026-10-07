"""Unit tests for scripts/build_latent_stats.py, using synthetic latent npz files."""
import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_latent_stats.py"
SPEC = importlib.util.spec_from_file_location("build_latent_stats", SCRIPT)
stats = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stats)


def write_latent(path: Path, array: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, latent_action=array.astype(np.float16),
             pair_indices=np.zeros((array.shape[0], 2), np.int32))


def write_manifest(path: Path, latent_paths):
    """Manifest in the shape iter_latent_paths expects: views LIST with latent_path."""
    entries = []
    for i, latent_path in enumerate(latent_paths):
        entries.append({
            "dataset": "synthetic",
            "episode_id": f"ep{i:04d}",
            "caption": "synthetic caption",
            "views": [{"view": "head", "video_path": "/nonexistent.mp4",
                       "latent_path": str(latent_path), "num_latents": 10,
                       "fps": 20.0, "pair_stride": 4}],
        })
    path.write_text(json.dumps(entries))


class BuildLatentStatsTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _run(self, arrays, dim, workers=1, limit=None):
        paths = []
        for i, arr in enumerate(arrays):
            p = self.root / "latents" / f"ep{i:04d}" / "head" / "latent.npz"
            write_latent(p, arr)
            paths.append(p)
        manifest = self.root / "manifest.json"
        write_manifest(manifest, paths)
        out = self.root / "stats.json"
        argv = ["--manifest", str(manifest), "--out", str(out),
                "--latent-dim", str(dim), "--workers", str(workers)]
        if limit is not None:
            argv += ["--limit", str(limit)]
        code = stats.main(argv)
        self.assertEqual(code, 0)
        return json.loads(out.read_text())

    def test_moments_match_a_direct_numpy_reduction(self):
        rng = np.random.default_rng(0)
        arrays = [rng.normal(size=(12, 5)) for _ in range(4)]
        got = self._run(arrays, dim=5)

        # The files are stored as float16 (the real corpus format), so the reference must be
        # computed from the round-tripped values, not the original float64 -- otherwise the
        # comparison measures float16 rounding (~8e-4 relative) rather than the reduction.
        flat = np.concatenate([a.astype(np.float16) for a in arrays]).astype(np.float64)
        expected_mean = flat.mean(axis=0)
        # n-1 unbiased form, matching base.py:_finalize_leaf
        expected_std = flat.std(axis=0, ddof=1)
        np.testing.assert_allclose(got["mean"], expected_mean.astype(np.float32), rtol=1e-5)
        np.testing.assert_allclose(got["std"], expected_std.astype(np.float32), rtol=1e-5)
        np.testing.assert_allclose(got["min"], flat.min(axis=0).astype(np.float32), rtol=1e-5)
        np.testing.assert_allclose(got["max"], flat.max(axis=0).astype(np.float32), rtol=1e-5)
        self.assertEqual(got["rows"], flat.shape[0])
        self.assertEqual(got["files"], len(arrays))

    def test_parallel_reduction_is_identical_to_serial(self):
        """The thread pool must not change the numbers: partials are merged, not shared."""
        rng = np.random.default_rng(7)
        arrays = [rng.normal(size=(9, 4)) for _ in range(25)]
        serial = self._run(arrays, dim=4, workers=1)
        parallel_root = Path(self._tmp.name)
        # Re-run against the same files with a fresh output path.
        manifest = parallel_root / "manifest.json"
        out = parallel_root / "stats_parallel.json"
        self.assertEqual(stats.main([
            "--manifest", str(manifest), "--out", str(out),
            "--latent-dim", "4", "--workers", "16",
        ]), 0)
        parallel = json.loads(out.read_text())
        for key in ("mean", "std", "min", "max"):
            np.testing.assert_array_equal(np.array(serial[key]), np.array(parallel[key]),
                                          err_msg=f"{key} differs between serial and parallel")
        self.assertEqual(serial["rows"], parallel["rows"])
        self.assertEqual(serial["files"], parallel["files"])

    def test_wrong_width_refuses_to_write(self):
        """A corpus labelled at one width against a --latent-dim for another must fail loudly."""
        arrays = [np.zeros((8, 6), np.float32) for _ in range(3)]
        paths = []
        for i, arr in enumerate(arrays):
            p = self.root / "latents" / f"ep{i:04d}" / "head" / "latent.npz"
            write_latent(p, arr)
            paths.append(p)
        manifest = self.root / "manifest.json"
        write_manifest(manifest, paths)
        out = self.root / "stats.json"
        with self.assertRaises(SystemExit) as ctx:
            stats.main(["--manifest", str(manifest), "--out", str(out), "--latent-dim", "4"])
        self.assertIn("wrong width", str(ctx.exception))
        self.assertFalse(out.exists(), "no stats may be written on a width mismatch")

    def test_constant_channel_refuses_to_write(self):
        """std == 0 would divide by zero at normalization; that is a labeling problem."""
        arrays = [np.ones((6, 3), np.float32) for _ in range(2)]
        paths = []
        for i, arr in enumerate(arrays):
            p = self.root / "latents" / f"ep{i:04d}" / "head" / "latent.npz"
            write_latent(p, arr)
            paths.append(p)
        manifest = self.root / "manifest.json"
        write_manifest(manifest, paths)
        out = self.root / "stats.json"
        with self.assertRaises(SystemExit) as ctx:
            stats.main(["--manifest", str(manifest), "--out", str(out), "--latent-dim", "3"])
        self.assertIn("constant", str(ctx.exception))
        self.assertFalse(out.exists())

    def test_missing_files_are_counted_not_fatal(self):
        """A manifest referencing a latent that has not landed yet must not abort the run."""
        present = [np.random.default_rng(1).normal(size=(7, 3)) for _ in range(3)]
        paths = []
        for i, arr in enumerate(present):
            p = self.root / "latents" / f"ep{i:04d}" / "head" / "latent.npz"
            write_latent(p, arr)
            paths.append(p)
        paths.append(self.root / "latents" / "ep9999" / "head" / "latent.npz")  # never written
        manifest = self.root / "manifest.json"
        write_manifest(manifest, paths)
        out = self.root / "stats.json"
        self.assertEqual(stats.main([
            "--manifest", str(manifest), "--out", str(out), "--latent-dim", "3",
        ]), 0)
        got = json.loads(out.read_text())
        self.assertEqual(got["missing"], 1)
        self.assertEqual(got["files"], 3)

    def test_deduplicates_repeated_latent_paths(self):
        """Two views sharing one latent file must not double-count its rows."""
        arr = np.random.default_rng(2).normal(size=(5, 3))
        p = self.root / "latents" / "ep0000" / "head" / "latent.npz"
        write_latent(p, arr)
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps([{
            "dataset": "synthetic", "episode_id": "ep0000", "caption": "c",
            "views": [{"view": "head", "video_path": "/x.mp4", "latent_path": str(p),
                       "num_latents": 5, "fps": 20.0, "pair_stride": 4},
                      {"view": "head", "video_path": "/x.mp4", "latent_path": str(p),
                       "num_latents": 5, "fps": 20.0, "pair_stride": 4}],
        }]))
        out = self.root / "stats.json"
        self.assertEqual(stats.main([
            "--manifest", str(manifest), "--out", str(out), "--latent-dim", "3",
        ]), 0)
        got = json.loads(out.read_text())
        self.assertEqual(got["files"], 1, "the same latent path must be read once")
        self.assertEqual(got["rows"], 5)


if __name__ == "__main__":
    unittest.main()
