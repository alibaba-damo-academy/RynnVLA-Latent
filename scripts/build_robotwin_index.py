#!/usr/bin/env python3
"""Build the episode index that ``RoboTwinDataset`` requires.

``RoboTwinDataset`` reads a prebuilt JSON index rather than scanning the corpus at load time,
because the corpus is large and lives on shared storage: walking it once, offline, keeps every
training rank from doing the same walk. This script does that walk.

Expected on-disk layout (RoboTwin's own)::

    <data-root>/<task_name>/<variant>/data/episode<N>.hdf5

Each index entry is::

    {"path": <absolute hdf5 path under data-root>,
     "length": <frames>,              # first dim of the action dataset
     "robot_type": "aloha_agilex",    # a rynnvla.constants.RobotType value
     "instructions": [<str>, ...],    # language annotations; one is drawn per sample
     "variant": <str>}                # one of robotwin.SUPPORTED_VARIANTS

``instructions`` is the one field that cannot be derived from the corpus: RoboTwin's HDF5 files
carry endpose / joint_action / observation but no language. Supply them with ``--instructions``
as a JSON object mapping task name to a list of phrasings, taken from your own RoboTwin
checkout's language annotations. A single phrasing per task is fine -- ``load_instruction``
draws uniformly from the list, so more phrasings just means more instruction diversity.

    python scripts/build_robotwin_index.py \
        --data-root data/robotwin/robotwin_raw_hdf5 \
        --instructions data/robotwin/robotwin_instructions.json \
        --out data/robotwin/robotwin_index.json

Then point a mixture at the pair (see configs/data_robotwin_mixed.example.json). Record the
index's sha256 in the mixture as ``index_sha256`` to make a resumed run refuse an index that
was rebuilt underneath it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rynnvla.constants import RobotType  # noqa: E402
from rynnvla.datasets.vla_datasets.robotwin import SUPPORTED_VARIANTS  # noqa: E402

# The HDF5 datasets each action space reads, per rynnvla/datasets/vla_datasets/robotwin.py.
# Checked for presence and read for their first dimension only -- the pixel data is never
# touched, so the walk stays cheap on a large corpus.
REQUIRED_KEYS = {
    "ee": ("endpose/left_endpose", "endpose/left_gripper",
           "endpose/right_endpose", "endpose/right_gripper"),
    "qpos": ("joint_action/vector",),
}
CAMERA_KEYS = ("observation/head_camera/rgb", "observation/left_camera/rgb",
               "observation/right_camera/rgb")


def _robot_type(variant, override):
    """Derive ``aloha_agilex`` from ``aloha-agilex_clean_50``, RoboTwin's own spelling flip."""
    if override:
        return override
    stem = variant.split("_")[0] if "_" in variant else variant
    return stem.replace("-", "_")


def _scan(root, action_space, variants, instructions, robot_type_override, parser):
    import h5py

    required = REQUIRED_KEYS[action_space]
    files = sorted(root.rglob("*.hdf5"))
    if not files:
        parser.error(f"no .hdf5 files under {root}")

    entries, skipped, no_language = [], [], []
    for path in files:
        relative = path.relative_to(root).parts
        # <task_name>/<variant>/data/episode<N>.hdf5. The variant is matched against path
        # components RELATIVE to the root, not against the full path: matching the full path
        # would also fire when the corpus root itself is named after a variant. Every entry
        # written here carries an explicit "variant", which is what RoboTwinDataset reads
        # first, so its own path-inference fallback never has to guess.
        variant = next((v for v in SUPPORTED_VARIANTS if v in relative), None)
        if variant is None:
            skipped.append((str(path), f"no supported variant in {relative}"))
            continue
        if variants and variant not in variants:
            continue
        task = relative[0] if len(relative) > 1 else ""
        try:
            with h5py.File(path, "r", locking=False) as handle:
                missing = [key for key in required + CAMERA_KEYS if key not in handle]
                if missing:
                    skipped.append((str(path), f"missing {missing}"))
                    continue
                length = int(handle[required[0]].shape[0])
        except OSError as exc:
            skipped.append((str(path), f"unreadable: {exc}"))
            continue
        if length < 1:
            skipped.append((str(path), f"{length} frames"))
            continue
        robot_type = _robot_type(variant, robot_type_override)
        try:
            RobotType(robot_type)
        except ValueError:
            parser.error(f"{path}: derived robot_type {robot_type!r} is not a RobotType; "
                         "pass --robot-type explicitly")
        phrases = instructions.get(task, [])
        if not phrases:
            no_language.append(task)
        entries.append({"path": str(path), "length": length, "robot_type": robot_type,
                        "instructions": list(phrases), "variant": variant})

    for path, reason in skipped:
        print(f"[robotwin-index] skip {path}: {reason}", file=sys.stderr)
    return entries, skipped, sorted(set(no_language))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", required=True, type=Path,
                        help="RoboTwin HDF5 corpus root; becomes the mixture's data_path")
    parser.add_argument("--out", required=True, type=Path,
                        help="destination index JSON; must not exist yet")
    parser.add_argument("--instructions", type=Path, default=None,
                        help="JSON object mapping task name -> list of language phrasings")
    parser.add_argument("--variant", action="append", default=None,
                        help=f"keep only this variant (repeatable); one of {SUPPORTED_VARIANTS}")
    parser.add_argument("--action-space", choices=sorted(REQUIRED_KEYS), default="ee",
                        help="which action layout the run will read (default: ee)")
    parser.add_argument("--robot-type", default=None,
                        help="override the robot_type derived from the variant name")
    parser.add_argument("--allow-empty-instructions", action="store_true",
                        help="build the index even for tasks with no language annotation")
    args = parser.parse_args(argv)

    root = args.data_root.expanduser()
    if not root.is_absolute():
        root = Path.cwd() / root
    if not root.is_dir():
        parser.error(f"--data-root is not a directory: {root}")
    out = args.out.expanduser()
    if not out.is_absolute():
        out = Path.cwd() / out
    if out.exists():
        # Same reasoning as build_dataset_weights.py: this file's sha256 is what a pinned
        # mixture validates against, so silently replacing it would make a resumed run train
        # on a different corpus than the one its recipe names.
        parser.error(f"Refusing to overwrite existing index: {out}")
    if out.resolve() == root.resolve() or root in out.resolve().parents:
        parser.error("--out must not live under --data-root; it would be indexed as an episode")
    if args.variant:
        unknown = [v for v in args.variant if v not in SUPPORTED_VARIANTS]
        if unknown:
            parser.error(f"unknown --variant {unknown}; supported: {SUPPORTED_VARIANTS}")

    instructions = {}
    if args.instructions is not None:
        try:
            instructions = json.loads(args.instructions.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            parser.error(f"cannot read --instructions: {exc}")
        if not isinstance(instructions, dict):
            parser.error("--instructions must be a JSON object mapping task name -> [phrases]")

    entries, skipped, no_language = _scan(
        root, args.action_space, set(args.variant) if args.variant else None,
        instructions, args.robot_type, parser)
    if not entries:
        parser.error(f"no episodes survived the scan of {root} ({len(skipped)} skipped)")
    if no_language and not args.allow_empty_instructions:
        parser.error(
            f"{len(no_language)} task(s) have no entry in --instructions: {no_language[:8]}"
            f"{' ...' if len(no_language) > 8 else ''}. Those episodes would train with an "
            "EMPTY prompt, which is a different (and worse) model than a language-conditioned "
            "one, and nothing raises about it later. Add them, or pass "
            "--allow-empty-instructions to accept it deliberately.")

    out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(entries)
    out.write_text(payload, encoding="utf-8")
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    per_variant = {}
    for entry in entries:
        per_variant[entry["variant"]] = per_variant.get(entry["variant"], 0) + 1
    print(f"[robotwin-index] {len(entries)} episodes from {root} "
          f"({len(skipped)} skipped), action_space={args.action_space}")
    for variant in sorted(per_variant):
        print(f"[robotwin-index]   {variant:32s} {per_variant[variant]:6d} episodes")
    print(f"[robotwin-index] sha256 {digest}")
    print(f"[robotwin-index] wrote {out}")
    print(f'[robotwin-index] mixture entry: {{"data_type": "RoboTwinDataset", '
          f'"data_path": "{root}", "index_cache": "{out}", '
          f'"index_sha256": "{digest}", "action_space": "{args.action_space}"}}')
    return 0


if __name__ == "__main__":
    sys.exit(main())
