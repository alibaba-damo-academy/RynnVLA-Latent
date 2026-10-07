#!/usr/bin/env python3
"""Extract frozen RynnLAM features from LARYBench tar archives, without LARY.

Metadata must contain exact tar member names in src_img, tgt_img and action.
For multiple workers use the same output root and distinct --shard values;
merge with evaluate_rynnlam.py merge, which refuses incomplete or mixed protocols.
"""

import argparse
from contextlib import contextmanager
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
import uuid
import zipfile

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rynnlam.inference import file_sha256 as sha256_file

DATASETS = {
    "calvin": {"lary_name": "calvin", "stride": 5, "action_dim": 7},
    "vlabench": {"lary_name": "vlabench", "stride": 5, "action_dim": 7},
    "robocoin": {"lary_name": "robocoin", "stride": 10, "action_dim": 12},
    "agibot": {"lary_name": "agibotbeta", "stride": 45, "action_dim": 16},
}
REPRESENTATIONS = ("ktoken", "z", "zcam", "hints_pool", "full", "encoder")


@contextmanager
def indexed_archive(path, local_temp_dir=None):
    """Decompress once onto local scratch before any indexed member reads."""
    with open(path, "rb") as source:
        magic = source.read(6)
        source.seek(0)
        if magic.startswith(b"\x1f\x8b"):
            import gzip

            decompressor = gzip.open
        elif magic.startswith(b"BZh"):
            import bz2

            decompressor = bz2.BZ2File
        elif magic.startswith(b"\xfd7zXZ\x00"):
            import lzma

            decompressor = lzma.LZMAFile
        else:
            decompressor = None
        if decompressor is None:
            with tarfile.open(fileobj=source, mode="r:") as archive:
                yield archive
        else:
            with tempfile.TemporaryFile(dir=local_temp_dir) as scratch:
                with decompressor(source, mode="rb") as decoded:
                    shutil.copyfileobj(decoded, scratch, length=8 * 1024 * 1024)
                scratch.seek(0)
                with tarfile.open(fileobj=scratch, mode="r:") as archive:
                    yield archive


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def atomic_write(path, writer, local_temp_dir=None):
    """Seek only on local scratch; sequential copy to destination then rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        with tempfile.TemporaryFile(dir=local_temp_dir) as local:
            writer(local)
            local.seek(0)
            with open(pending, "wb") as target:
                shutil.copyfileobj(local, target)
        os.replace(pending, path)
    finally:
        pending.unlink(missing_ok=True)


def write_json(path, data, local_temp_dir=None):
    atomic_write(
        path,
        lambda f: f.write(json.dumps(data, indent=2, sort_keys=True).encode()),
        local_temp_dir,
    )


def write_csv(path, rows, fields, local_temp_dir=None):
    def writer(f):
        text = io.TextIOWrapper(f, encoding="utf-8", newline="", write_through=True)
        try:
            out = csv.DictWriter(text, fieldnames=fields)
            out.writeheader()
            out.writerows(rows)
            text.flush()
        finally:
            text.detach()

    atomic_write(path, writer, local_temp_dir)


def read_metadata(path, dataset):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        required = {"src_img", "tgt_img", "action"}
        if dataset in ("agibot", "robocoin"):
            required.add("robot_type")
        if not required.issubset(fields):
            raise ValueError(
                f"Metadata missing columns: {sorted(required - set(fields))}"
            )
        rows = list(reader)
    if not rows:
        raise ValueError("Metadata CSV is empty")
    for i, row in enumerate(rows):
        if any(not row.get(k, "").strip() for k in required):
            raise ValueError(f"Empty required metadata field at row {i}")
    return rows, fields


def decode_img(raw, height=238, width=322):
    from rynnlam.video import resize_crop

    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Image cannot be decoded")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return resize_crop(image, (height, width)).astype(np.float32) / 255.0


def validate_action(raw, dataset, stride):
    action = np.load(io.BytesIO(raw), allow_pickle=False)
    expected = stride * DATASETS[dataset]["action_dim"]
    if (
        action.size != expected
        or not np.issubdtype(action.dtype, np.number)
        or not np.isfinite(action).all()
    ):
        raise ValueError(
            f"Invalid action: expected {expected} finite numeric values; got {action.shape}"
        )
    return action


def validate_output(
    token_path, action_path, protocol_id, row_id, dataset, stride, token_shapes=None
):
    """Validate the row/protocol/action binding shared by resume and regression."""
    token_path, action_path = Path(token_path), Path(action_path)
    for path in (token_path, action_path):
        if not path.is_absolute() or not path.is_file():
            raise ValueError(f"Missing absolute extraction path: {path}")
    try:
        with np.load(token_path, allow_pickle=False) as data:
            token = data["tokens"]
            if (
                data["protocol_id"].item() != protocol_id
                or data["row_id"].item() != row_id
            ):
                raise ValueError("Token protocol/row binding mismatch")
            if (
                not token.size
                or not np.issubdtype(token.dtype, np.number)
                or not np.isfinite(token).all()
            ):
                raise ValueError("Non-finite, non-numeric or empty tokens")
            if token_shapes is not None and tuple(token.shape) not in {
                tuple(s) for s in token_shapes
            }:
                raise ValueError("Token shape differs from extraction report")
            validate_action(action_path.read_bytes(), dataset, stride)
            if data["action_sha256"].item() != sha256_file(action_path):
                raise ValueError("Action checksum mismatch")
            return token.shape
    # zipfile.BadZipFile subclasses Exception directly and is caught by none of the types
    # above. Folding it into the ValueError matters for resume: `resumed()` only catches
    # ValueError, so a truncated token npz would otherwise abort the whole shard instead of
    # being regenerated.
    except (OSError, ValueError, KeyError, EOFError, TypeError, zipfile.BadZipFile) as exc:
        raise ValueError(f"Invalid extraction output {token_path}: {exc}") from exc


def reduce_tokens(tokens, pooling):
    # Preserve the actual K x D for flat probes (feature-source D is 2048).
    if pooling == "mean" and tokens.ndim > 2:
        tokens = tokens.flatten(1, -2).mean(dim=1)
    return tokens.detach().float().cpu().numpy()


def extract(args, encoder=None):
    if args.batch_size < 1 or args.height < 1 or args.width < 1:
        raise ValueError("Batch size and geometry must be positive")
    if args.num_shards < 1 or not 0 <= args.shard < args.num_shards:
        raise ValueError("Require 0 <= shard < num_shards")
    rows, fields = read_metadata(args.csv, args.dataset)
    parts = sorted(Path(args.tar_dir).resolve().glob(args.tar_glob))
    if not parts:
        raise ValueError("No tar archives match --tar-dir / --tar-glob")
    if any(not p.is_file() for p in parts):
        raise ValueError("Archive glob matched a non-file")
    stride = args.stride or DATASETS[args.dataset]["stride"]
    if stride < 1:
        raise ValueError("Stride must be positive")
    if encoder is None:
        from rynnlam.inference import RynnLAMEncoder

        encoder = RynnLAMEncoder(
            args.checkpoint,
            device=args.device,
            normalize=args.normalization == "imagenet",
            precision=args.precision,
            trust_checkpoint=args.trust_checkpoint,
        )
    config = getattr(encoder, "config", None)
    if config is None:
        config = getattr(getattr(encoder, "model", None), "config", None)
    if hasattr(config, "to_dict"):
        config = config.to_dict()
    elif hasattr(config, "__dict__"):
        config = vars(config)
    # The full checkpoint hash also binds embedded config, even if not exposed.
    config = json.loads(json.dumps(config, default=str))
    from rynnlam.inference import implementation_sha256

    protocol = {
        "version": 1,
        "checkpoint_sha256": encoder.checkpoint_sha256,
        "extractor_sha256": sha256_file(__file__),
        "implementation_sha256": implementation_sha256(),
        "config": config,
        "config_source": "checkpoint (bound by checkpoint_sha256)",
        "dataset": args.dataset,
        "split": args.split,
        "stride": stride,
        "action_dim": DATASETS[args.dataset]["action_dim"],
        "action_mode": "absolute",
        "representation": args.representation,
        "pooling": args.pool,
        "normalization": args.normalization,
        "height": args.height,
        "width": args.width,
        "crop": "bicubic resize-to-cover then center crop",
        "precision": args.precision,
        "storage_dtype": "float16" if args.fp16 else "float32",
        "extractor_frozen": True,
        "metadata_sha256": sha256_file(args.csv),
        "num_shards": args.num_shards,
        "archives": [
            {"path": str(p), "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
            for p in parts
        ],
    }
    protocol_id = digest(protocol)
    tag = args.tag or "rynnlam"
    if (
        not tag
        or any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
            for c in tag
        )
        or tag in (".", "..")
    ):
        raise ValueError("Tag must contain only letters, digits, _, - and .")
    run_dir = (
        Path(args.output_root).resolve()
        / args.dataset
        / f"{tag}-{protocol_id[:16]}"
        / args.split
    )
    shard_dir = run_dir / f"shard-{args.shard:05d}-of-{args.num_shards:05d}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    lock = shard_dir / ".lock"
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise RuntimeError(
            f"Output locked: {lock}; remove only after confirming no live worker"
        ) from exc
    try:
        return _extract_locked(
            args, encoder, rows, fields, parts, stride, protocol, protocol_id, shard_dir
        )
    except Exception as exc:
        report_path = shard_dir / "report.json"
        try:
            report = json.loads(report_path.read_text())
        except (OSError, ValueError):
            report = {"protocol": protocol, "protocol_id": protocol_id}
        if report.get("status") != "failed":
            report.update(
                status="failed",
                errors=[{"error": str(exc), "type": type(exc).__name__}],
            )
            write_json(report_path, report, args.local_temp_dir)
        raise
    finally:
        lock.rmdir()


def _extract_locked(
    args, encoder, rows, fields, parts, stride, protocol, protocol_id, out
):
    csv_path = out / f"{args.split}_la.csv"
    # Never leave a previous successful CSV visible while rebuilding.
    csv_path.unlink(missing_ok=True)
    report_path = out / "report.json"
    write_json(
        report_path,
        {"status": "running", "protocol": protocol, "protocol_id": protocol_id},
        args.local_temp_dir,
    )
    src2rows = {}
    for i, row in enumerate(rows):
        src2rows.setdefault(row["src_img"], []).append(i)
    completed, owned, errors, shapes = {}, set(), [], set()
    batch = []

    def paths(i):
        return (
            out / "tokens" / f"latent_action_{i:08d}.npz",
            out / "actions" / f"action_{i:08d}.npy",
        )

    def record(i, shape):
        token_path, action_path = paths(i)
        row = dict(
            rows[i],
            la_path=str(token_path),
            action=str(action_path),
            rynnlam_row_id=str(i),
            rynnlam_protocol_id=protocol_id,
        )
        completed[i] = row
        shapes.add(tuple(shape))

    def resumed(i):
        token_path, action_path = paths(i)
        if not args.resume:
            return False
        try:
            shape = validate_output(
                token_path, action_path, protocol_id, i, args.dataset, stride
            )
            record(i, shape)
            return True
        except ValueError:
            return False

    def flush():
        if not batch:
            return
        images = torch.from_numpy(np.stack([entry[1] for entry in batch]))
        with torch.inference_mode():
            value = encoder(images, representation=args.representation)
        if (
            not isinstance(value, torch.Tensor)
            or value.shape[0] != len(batch)
            or value.ndim < 2
        ):
            raise ValueError("Encoder returned an invalid batch")
        tokens = reduce_tokens(value, args.pool)
        for (i, _, action_raw), token in zip(batch, tokens):
            token = token.astype(np.float16 if args.fp16 else np.float32)
            if not token.size or not np.isfinite(token).all():
                raise ValueError(f"Non-finite or empty features at row {i}")
            token_path, action_path = paths(i)
            atomic_write(
                action_path, lambda f, raw=action_raw: f.write(raw), args.local_temp_dir
            )
            atomic_write(
                token_path,
                lambda f, t=token, row_id=i, raw=action_raw: np.savez(
                    f,
                    tokens=t,
                    indices=np.array([], dtype=np.int64),
                    protocol_id=protocol_id,
                    row_id=row_id,
                    action_sha256=hashlib.sha256(raw).hexdigest(),
                ),
                args.local_temp_dir,
            )
            record(i, token.shape)
        batch.clear()

    # Index each selected archive once, then decode only referenced members.
    # The archive is never extracted to a dataset directory.
    for part in parts[args.shard :: args.num_shards]:
        try:
            with indexed_archive(part, args.local_temp_dir) as archive:
                members = {m.name: m for m in archive.getmembers() if m.isfile()}
                # Sort row IDs, including interleaved rows sharing a source.
                selected_rows = sorted(
                    i for src in src2rows if src in members for i in src2rows[src]
                )
                for i in selected_rows:
                    if i in owned:
                        errors.append(
                            {
                                "row": i,
                                "error": "Source occurs in multiple selected archives",
                            }
                        )
                        continue
                    owned.add(i)
                    row = rows[i]
                    missing = [
                        row[k] for k in ("tgt_img", "action") if row[k] not in members
                    ]
                    if missing:
                        errors.append(
                            {"row": i, "error": "Missing members", "members": missing}
                        )
                        continue
                    if resumed(i):
                        continue
                    try:
                        raw = archive.extractfile(members[row["action"]]).read()
                        validate_action(raw, args.dataset, stride)
                        images = [
                            decode_img(
                                archive.extractfile(members[row[k]]).read(),
                                args.height,
                                args.width,
                            )
                            for k in ("src_img", "tgt_img")
                        ]
                        batch.append((i, np.stack(images), raw))
                    except (OSError, ValueError, cv2.error) as exc:
                        errors.append({"row": i, "error": str(exc)})
                        continue
                    if len(batch) >= args.batch_size:
                        flush()
                flush()
        except (tarfile.TarError, OSError) as exc:
            errors.append({"archive": str(part), "error": str(exc)})
    missing_rows = sorted(
        (set(range(len(rows))) if args.num_shards == 1 else owned) - completed.keys()
    )
    if missing_rows:
        errors.append({"error": "Rows not extracted", "rows": missing_rows})
    if len(shapes) > 1:
        errors.append({"error": "Inconsistent token shapes", "shapes": sorted(shapes)})
    output_rows = [completed[i] for i in sorted(completed)]
    output_fields = list(
        dict.fromkeys(fields + ["la_path", "rynnlam_row_id", "rynnlam_protocol_id"])
    )
    report = {
        "status": (
            "failed"
            if errors
            else ("complete" if args.num_shards == 1 else "shard_complete")
        ),
        "protocol": protocol,
        "protocol_id": protocol_id,
        "shard": args.shard,
        "metadata_rows": len(rows),
        "owned_rows": sorted(owned),
        "completed_rows": sorted(completed),
        "token_shapes": sorted(shapes),
        "flat_feature_dim": (
            int(np.prod(next(iter(shapes)))) if len(shapes) == 1 else None
        ),
        "errors": errors,
        "csv": str(csv_path),
    }
    if errors:
        partial = out / f"{args.split}_la.partial.csv"
        write_csv(partial, output_rows, output_fields, args.local_temp_dir)
        write_json(report_path, report, args.local_temp_dir)
        raise RuntimeError(
            f"Incomplete extraction: {len(errors)} errors; see {report_path}"
        )
    write_csv(csv_path, output_rows, output_fields, args.local_temp_dir)
    report["csv_sha256"] = sha256_file(csv_path)
    write_json(report_path, report, args.local_temp_dir)
    print(f"{len(output_rows)} rows -> {csv_path}")
    if args.num_shards > 1:
        print(
            "Shard only: evaluate.py merge must validate full metadata coverage before regression."
        )
    return report_path


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--split", required=True, choices=("train", "val"))
    parser.add_argument(
        "--csv", required=True, help="Metadata CSV; exact archive member names"
    )
    parser.add_argument("--tar-dir", required=True)
    parser.add_argument(
        "--tar-glob", required=True, help="Quoted archive glob relative to --tar-dir"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--output-root",
        required=True,
        help="Dedicated output root; never the source dataset directory",
    )
    parser.add_argument("--tag", default="rynnlam")
    parser.add_argument("--representation", choices=REPRESENTATIONS, default="ktoken")
    parser.add_argument(
        "--pool",
        choices=("flat", "mean"),
        default="flat",
        help="Flat keeps K x D tokens; official regression flattens them",
    )
    parser.add_argument(
        "--normalization", choices=("imagenet", "none"), default="imagenet"
    )
    parser.add_argument("--height", type=int, default=238)
    parser.add_argument("--width", type=int, default=322)
    parser.add_argument(
        "--stride", type=int, help="Defaults: CALVIN/VLABench 5, RoboCOIN 10, AgiBot 45"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Store features in float16; independent of inference precision",
    )
    parser.add_argument("--trust-checkpoint", action="store_true")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--local-temp-dir",
        help="Local seekable scratch directory (default: system temporary directory)",
    )
    return parser


def main():
    args = build_parser().parse_args()
    try:
        extract(args)
    except (ValueError, RuntimeError, OSError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
