"""``scripts/build_robotwin_index.py`` must emit exactly what ``RoboTwinDataset`` will accept.

The index is built offline and pinned into a mixture by sha256, so a wrong field is not a crash
at build time -- it is a crash (or worse, a silently language-free run) much later, on every
rank, after the GPUs are allocated. These tests build a synthetic corpus with the real HDF5
layout and check the contract from both sides: what the script writes, and that the dataset
class will load it.
"""
import importlib.util
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from rynnvla.constants import RobotType
from rynnvla.datasets.vla_datasets.robotwin import SUPPORTED_VARIANTS

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "_build_robotwin_index_under_test", REPO_ROOT / "scripts/build_robotwin_index.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bri = _load_script()

VARIANT = SUPPORTED_VARIANTS[0]                     # "aloha-agilex_clean_50"
TASKS = ("handover_mic", "blocks_ranking_rgb")


def _write_episode(path, frames=12, action_space="ee", drop=()):
    path.parent.mkdir(parents=True, exist_ok=True)
    jpeg = np.array([b"\xff\xd8frame"] * frames)     # opaque bytes; only presence is checked
    with h5py.File(path, "w") as handle:
        if action_space == "ee":
            for side in ("left", "right"):
                handle.create_dataset(f"endpose/{side}_endpose",
                                      data=np.zeros((frames, 7), dtype=np.float64))
                handle.create_dataset(f"endpose/{side}_gripper",
                                      data=np.zeros(frames, dtype=np.float64))
        else:
            handle.create_dataset("joint_action/vector",
                                  data=np.zeros((frames, 14), dtype=np.float64))
        for camera in ("head", "left", "right"):
            key = f"observation/{camera}_camera/rgb"
            if key not in drop:
                handle.create_dataset(key, data=jpeg)


def _corpus(tmp_path, tasks=TASKS, frames=12, action_space="ee", drop=()):
    root = tmp_path / "robotwin_raw_hdf5"
    for task in tasks:
        _write_episode(root / task / VARIANT / "data" / "episode0.hdf5",
                       frames, action_space, drop)
    return root


def _instructions(tmp_path, tasks=TASKS):
    path = tmp_path / "instructions.json"
    path.write_text(json.dumps({task: [f"do the {task}", f"and again {task}"] for task in tasks}),
                    encoding="utf-8")
    return path


def _run(*argv):
    return bri.main([str(a) for a in argv])


def _entries(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ── the contract RoboTwinDataset reads ───────────────────────────────────────

def test_entry_carries_every_field_the_dataset_reads(tmp_path):
    root = _corpus(tmp_path)
    out = tmp_path / "index.json"
    assert _run("--data-root", root, "--instructions", _instructions(tmp_path), "--out", out) == 0
    entries = _entries(out)
    assert len(entries) == len(TASKS)
    for entry in entries:
        assert set(entry) == {"path", "length", "robot_type", "instructions", "variant"}
        assert entry["length"] == 12
        assert entry["variant"] == VARIANT
        assert Path(entry["path"]).is_file()
        assert Path(entry["path"]).is_relative_to(root)
        assert len(entry["instructions"]) == 2


def test_robot_type_derivation_flips_the_variant_spelling(tmp_path):
    """RoboTwin names the variant with a hyphen and the RobotType with an underscore.

    Getting this wrong is not a build-time error: RobotType(...) is only constructed when a
    sample is read, so every rank would fail at the first __getitem__ instead.
    """
    root = _corpus(tmp_path)
    out = tmp_path / "index.json"
    _run("--data-root", root, "--instructions", _instructions(tmp_path), "--out", out)
    for entry in _entries(out):
        assert entry["robot_type"] == "aloha_agilex"
        assert RobotType(entry["robot_type"]) == RobotType.ALOHA_AGILEX


def test_a_derived_robot_type_that_is_not_a_robottype_is_refused(tmp_path, monkeypatch):
    root = _corpus(tmp_path)
    monkeypatch.setattr(bri, "SUPPORTED_VARIANTS", ("madeup-robot_clean_50",))
    odd = tmp_path / "odd"
    _write_episode(odd / "task" / "madeup-robot_clean_50" / "data" / "episode0.hdf5")
    with pytest.raises(SystemExit):
        _run("--data-root", odd, "--instructions", _instructions(tmp_path, ("task",)),
             "--out", tmp_path / "index.json")


def test_dataset_loads_what_the_script_wrote(tmp_path):
    """The two-sided half: build the index, then actually construct the dataset from it."""
    from rynnvla.datasets.vla_datasets.robotwin import RoboTwinDataset

    root = _corpus(tmp_path, tasks=(TASKS[0],))
    index = tmp_path / "index.json"
    _run("--data-root", root, "--instructions", _instructions(tmp_path, (TASKS[0],)),
         "--out", index)
    dataset = RoboTwinDataset(
        data_path=str(root), index_cache=str(index), action_chunk_size=4,
        use_delta_action=False, processor=None, action_space="ee", chunk_overlap_ratio=0.99)
    assert len(dataset) > 0
    assert dataset.episode_lengths == [12]
    assert dataset.get_robot_type(0) == RobotType.ALOHA_AGILEX
    assert dataset.load_instruction(0, 0)[0] in _entries(index)[0]["instructions"]


# ── language conditioning must not be silently empty ─────────────────────────

def test_missing_language_is_refused_rather_than_written_empty(tmp_path):
    root = _corpus(tmp_path)
    partial = tmp_path / "partial.json"
    partial.write_text(json.dumps({TASKS[0]: ["only one task annotated"]}), encoding="utf-8")
    out = tmp_path / "index.json"
    with pytest.raises(SystemExit):
        _run("--data-root", root, "--instructions", partial, "--out", out)
    assert not out.exists()


def test_allow_empty_instructions_is_explicit(tmp_path):
    root = _corpus(tmp_path)
    out = tmp_path / "index.json"
    assert _run("--data-root", root, "--out", out, "--allow-empty-instructions") == 0
    assert all(entry["instructions"] == [] for entry in _entries(out))


# ── scanning ─────────────────────────────────────────────────────────────────

def test_variant_filter_selects_a_subset(tmp_path):
    root = _corpus(tmp_path, tasks=(TASKS[0],))
    other = SUPPORTED_VARIANTS[1]
    _write_episode(root / TASKS[0] / other / "data" / "episode0.hdf5")
    out = tmp_path / "index.json"
    _run("--data-root", root, "--instructions", _instructions(tmp_path, (TASKS[0],)),
         "--out", out, "--variant", VARIANT)
    entries = _entries(out)
    assert len(entries) == 1 and entries[0]["variant"] == VARIANT


def test_a_variant_named_in_the_root_path_does_not_leak_into_episodes(tmp_path):
    """The variant is matched against path components relative to --data-root.

    Matching the full path would tag every episode with the root's own name when a corpus is
    stored under a directory that happens to be named after a variant.
    """
    inner = tmp_path / VARIANT                       # root itself carries a variant name
    _write_episode(inner / "loose" / "data" / "episode0.hdf5")          # no variant component
    _write_episode(inner / TASKS[0] / VARIANT / "data" / "episode0.hdf5")
    out = tmp_path / "index.json"
    _run("--data-root", inner, "--out", out, "--allow-empty-instructions")
    entries = _entries(out)
    assert len(entries) == 1, "the loose episode must not inherit the root's variant name"
    assert Path(entries[0]["path"]).is_relative_to(inner / TASKS[0] / VARIANT)


def test_qpos_action_space_reads_the_joint_vector(tmp_path):
    root = _corpus(tmp_path, tasks=(TASKS[0],), frames=9, action_space="qpos")
    out = tmp_path / "index.json"
    assert _run("--data-root", root, "--instructions", _instructions(tmp_path, (TASKS[0],)),
                "--out", out, "--action-space", "qpos") == 0
    assert _entries(out)[0]["length"] == 9


def test_an_episode_missing_a_camera_is_skipped_not_fatal(tmp_path):
    root = _corpus(tmp_path, drop=("observation/left_camera/rgb",))
    _write_episode(root / TASKS[1] / VARIANT / "data" / "episode1.hdf5")   # a complete one
    out = tmp_path / "index.json"
    assert _run("--data-root", root, "--instructions", _instructions(tmp_path), "--out", out) == 0
    assert len(_entries(out)) == 1


def test_an_empty_corpus_is_an_error(tmp_path):
    root = tmp_path / "empty"
    root.mkdir()
    with pytest.raises(SystemExit):
        _run("--data-root", root, "--out", tmp_path / "index.json",
             "--allow-empty-instructions")


# ── refusing to be wrong quietly ─────────────────────────────────────────────

def test_refuses_to_overwrite_an_existing_index(tmp_path):
    root = _corpus(tmp_path, tasks=(TASKS[0],))
    out = tmp_path / "index.json"
    out.write_text("[]", encoding="utf-8")
    with pytest.raises(SystemExit):
        _run("--data-root", root, "--instructions", _instructions(tmp_path, (TASKS[0],)),
             "--out", out)
    assert out.read_text(encoding="utf-8") == "[]"


def test_refuses_to_write_the_index_inside_the_corpus(tmp_path):
    """It would then be picked up as an episode on the next rebuild."""
    root = _corpus(tmp_path, tasks=(TASKS[0],))
    with pytest.raises(SystemExit):
        _run("--data-root", root, "--instructions", _instructions(tmp_path, (TASKS[0],)),
             "--out", root / "index.json")


def test_reported_sha256_matches_the_written_bytes(tmp_path, capsys):
    """A mixture pins the index by this digest, so the printed value has to be the file's."""
    import hashlib
    root = _corpus(tmp_path, tasks=(TASKS[0],))
    out = tmp_path / "index.json"
    _run("--data-root", root, "--instructions", _instructions(tmp_path, (TASKS[0],)), "--out", out)
    logged = capsys.readouterr().out
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    assert digest in logged
