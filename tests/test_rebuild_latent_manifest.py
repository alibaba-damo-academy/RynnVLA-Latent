"""Contract tests for the manifest rebuild path: scripts/rebuild_latent_manifest.py,
rynnvla/utils/manifest_io.py, and the JSONL side of latent_pretrain.build_index /
build_latent_stats.

The rebuild exists because RynnLAM's index_shard*.json lost ~93% of its entries (each dataset
overwrote the shard's index), so the manifest has to be reconstructed from the source episode
JSONs plus the latent files themselves. These tests pin the contract that reconstruction
depends on:

* the tail read of a latent .npz returns exactly what ``np.load(...)['meta']`` returns, and
  falls back to numpy when the tail does not contain the member;
* every field build_index consumes (num_latents / fps / pair_stride / video_path) comes from
  the labeling protocol recorded in the file, not from an assumption;
* episodes are dropped loudly, with a counted reason, when a view is unlabeled, when the
  label was windowed (start_frame != 0), or when the width/stride is not the expected one;
* part files give crash-safe resume, shards partition without overlap, and combining parts
  from two different shardings is refused rather than silently duplicating episodes;
* JSONL and JSON-array manifests produce the same index.

Self-contained: synthetic source JSONs and latent npz files under tmp_path.

Usage:
    python -m pytest tests/test_rebuild_latent_manifest.py -v
"""
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _PROJECT_ROOT)

from rynnvla.constants import NUM_VIEW_SLOTS  # noqa: E402
from rynnvla.utils.manifest_io import iter_manifest, manifest_format  # noqa: E402


def _load_script(name):
    path = os.path.join(_PROJECT_ROOT, "scripts", f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rebuild = _load_script("rebuild_latent_manifest")
stats = _load_script("build_latent_stats")


# ── synthetic corpus ──────────────────────────────────────────────────────────

def write_latent(path, num_latents=21, dim=608, fps=30.0, pair_stride=4, gap=4,
                 start_frame=0, source=None, representation="ktoken_zcam", meta_last=True):
    """A latent npz shaped like RynnLAM's: latent_action / pair_indices / meta.

    ``meta_last`` mirrors the labeler's member order, which is what makes the single tail
    read sufficient; ``meta_last=False`` builds a file the fast path cannot serve.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    meta = json.dumps({
        "protocol": {"schema_version": 2, "gap": gap, "pair_stride": pair_stride,
                     "start_frame": start_frame, "end_frame": None,
                     "representation": representation,
                     "source": source or "/videos/unused.mp4"},
        "view": "head", "num_latents": num_latents, "num_frames": num_latents * pair_stride + gap,
        "unpaired_tail_frames": 0, "code_shape": [dim], "fps": fps,
    })
    arrays = {"latent_action": (np.random.default_rng(abs(hash(path)) % 2**32)
                                .standard_normal((num_latents, dim)) * 0.1).astype(np.float16),
              "pair_indices": np.zeros((num_latents, 2), np.int32),
              "meta": meta}
    np.savez(path, **arrays) if meta_last else np.savez(
        path, meta=meta, latent_action=arrays["latent_action"],
        pair_indices=arrays["pair_indices"])
    return path


def write_source_json(path, episodes):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(episodes, ensure_ascii=False), encoding="utf-8")
    return path


def episode(dataset, episode_id, views, caption=None):
    return {
        "dataset": dataset,
        "episode_id": episode_id,
        "caption": caption if caption is not None else [
            {"start_time": 3.0, "description": "grasp the mug"},
            {"start_time": 0.0, "description": "reach for the mug"},
        ],
        "views": {view: {"video_path": f"/videos/{episode_id.replace('/', '_')}_{view}.mp4",
                         "flow_path": "/nonexistent/flow.npz", "has_flow": False,
                         "depth_path": "/nonexistent/depth.npz", "has_depth": False}
                  for view in views},
    }


@pytest.fixture()
def corpus(tmp_path):
    """Two datasets: 'SyntheticBase' (3 episodes, one 2-view) and 'TinySet' (1 episode)."""
    data_root = tmp_path / "source"
    latent_dir = tmp_path / "latents"
    out_dir = tmp_path / "work"
    episodes = [
        episode("SyntheticBase", "chunk-000/ep0", ["head"]),
        episode("SyntheticBase", "chunk-000/ep1", ["head"]),
        episode("SyntheticBase", "chunk-000/ep2", ["global", "wrist_left"]),
    ]
    write_source_json(data_root / "SyntheticBase.json", episodes)
    write_source_json(data_root / "TinySet.json", [episode("TinySet", "ep0", ["head"])])
    # A caption artifact in the data root, skipped exactly like label_rynnvla_base.sh skips it.
    write_source_json(data_root / "caption_SyntheticBase.json", episodes)

    for ep in episodes:
        for view in ep["views"]:
            video = ep["views"][view]["video_path"]
            write_latent(latent_dir / ep["dataset"] / ep["episode_id"] / view / "latent.npz",
                         num_latents=21, fps=30.0, source=video)
    write_latent(latent_dir / "TinySet" / "ep0" / "head" / "latent.npz", num_latents=9, fps=20.0,
                 source="/videos/TinySet_ep0_head.mp4")
    return SimpleNamespace(data_root=str(data_root), latent_dir=str(latent_dir),
                           out_dir=str(out_dir), tmp=tmp_path)


def run_rebuild(corpus, **extra):
    argv = ["--data-root", corpus.data_root, "--latent-dir", corpus.latent_dir,
            "--out-dir", corpus.out_dir]
    for key, value in extra.items():
        argv += [f"--{key.replace('_', '-')}", str(value)]
    _run(argv)
    return _read_parts(corpus.out_dir)


def _run(argv):
    old = sys.argv
    sys.argv = ["rebuild_latent_manifest.py"] + argv
    try:
        rebuild.main()
    finally:
        sys.argv = old


def _read_parts(out_dir, subdir="parts"):
    parts_dir = Path(out_dir) / "manifests" / subdir
    entries = {}
    if not parts_dir.is_dir():
        return entries
    for path in sorted(parts_dir.glob("*.jsonl")):
        entries[path.name] = [json.loads(line) for line in
                              path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return entries


# ── npz tail read ─────────────────────────────────────────────────────────────

def test_tail_read_equals_numpy_load(tmp_path):
    path = write_latent(tmp_path / "ds" / "ep" / "head" / "latent.npz", num_latents=17,
                        fps=29.97, pair_stride=4, gap=4)
    with np.load(path, allow_pickle=False) as archive:
        expected = json.loads(str(archive["meta"]))
    assert rebuild.read_npz_meta(path) == expected
    # A tail too small to hold the member must fall back to numpy, not guess.
    meta, how = rebuild.probe_latent(path, tail_bytes=64)
    assert how == "npz" and meta == expected


def test_probe_reports_missing_and_unreadable(tmp_path):
    assert rebuild.probe_latent(str(tmp_path / "gone.npz"), 1 << 16) == (None, "missing_latent")
    broken = tmp_path / "broken.npz"
    broken.write_bytes(b"PK\x03\x04 not really a zip")
    assert rebuild.probe_latent(str(broken), 1 << 16) == (None, "unreadable_latent")


# ── rebuild contract ──────────────────────────────────────────────────────────

def test_rebuild_reads_every_field_from_the_labeling_protocol(corpus):
    parts = run_rebuild(corpus, datasets="SyntheticBase")
    entries = parts["SyntheticBase.n1.shard0000.jsonl"]
    assert [e["episode_id"] for e in entries] == ["chunk-000/ep0", "chunk-000/ep1", "chunk-000/ep2"]
    first = entries[0]
    assert first["dataset"] == "SyntheticBase"
    assert first["gap"] == 4 and first["latent_dim"] == 608
    assert first["caption"] == "reach for the mug grasp the mug"  # ordered + deduplicated
    view = first["views"][0]
    assert view == {"view": "head",
                    "video_path": "/videos/chunk-000_ep0_head.mp4",  # protocol.source, not the JSON
                    "latent_path": os.path.join(corpus.latent_dir, "SyntheticBase",
                                                "chunk-000/ep0", "head", "latent.npz"),
                    "num_latents": 21, "fps": 30.0, "pair_stride": 4, "start_frame": 0}
    # Canonical camera order, not alphabetical: global (role 3) before wrist_left (role 1).
    assert [v["view"] for v in entries[2]["views"]] == ["wrist_left", "global"]


def test_non_dataset_jsons_are_not_rebuilt(corpus):
    parts = run_rebuild(corpus)
    assert sorted(parts) == ["SyntheticBase.n1.shard0000.jsonl", "TinySet.n1.shard0000.jsonl"]


def test_unlabeled_and_partially_labeled_episodes_are_dropped_with_a_reason(corpus):
    latent_dir = Path(corpus.latent_dir)
    os.unlink(latent_dir / "SyntheticBase" / "chunk-000/ep0" / "head" / "latent.npz")
    os.unlink(latent_dir / "SyntheticBase" / "chunk-000/ep2" / "wrist_left" / "latent.npz")
    parts = run_rebuild(corpus, datasets="SyntheticBase")
    entries = parts["SyntheticBase.n1.shard0000.jsonl"]
    assert [e["episode_id"] for e in entries] == ["chunk-000/ep1"]
    report = json.loads((Path(corpus.out_dir) / "rebuild_report.n1.shard0000.json").read_text())
    skips = report["datasets"][0]["skips"]
    # ep0 lost its only view, ep2 lost one of two: both are a missing latent AND an episode
    # that cannot be trained with a camera silently absent.
    assert skips["missing_latent"] == 2 and skips["incomplete_views"] == 2


def test_windowed_labels_carry_their_offset(corpus):
    """Some corpora are consecutive windows of one long recording rather than standalone
    episodes. Latent 0 of such an episode is not source frame 0, so the offset has to travel to
    the index."""
    write_latent(Path(corpus.latent_dir) / "TinySet" / "ep0" / "head" / "latent.npz",
                 num_latents=9, fps=20.0, start_frame=1956)
    parts = run_rebuild(corpus, datasets="TinySet")
    entry = parts["TinySet.n1.shard0000.jsonl"][0]
    assert entry["views"][0]["start_frame"] == 1956
    report = json.loads((Path(corpus.out_dir) / "rebuild_report.n1.shard0000.json").read_text())
    counters = report["datasets"][0]["counters"]
    assert counters["windowed_start_frame:TinySet"] == 1
    assert "windowed_start_frame:TinySet" not in report["datasets"][0]["skips"]  # not a drop

    # A negative offset cannot be a frame index; that one is refused rather than clamped.
    write_latent(Path(corpus.latent_dir) / "TinySet" / "ep0" / "head" / "latent.npz",
                 num_latents=9, fps=20.0, start_frame=-4)
    _run(["--data-root", corpus.data_root, "--latent-dir", corpus.latent_dir,
          "--out-dir", corpus.out_dir, "--datasets", "TinySet", "--force"])
    report = json.loads((Path(corpus.out_dir) / "rebuild_report.n1.shard0000.json").read_text())
    assert report["datasets"][0]["skips"]["negative_start_frame:-4"] == 1


def test_index_carries_start_frame_and_the_dataset_decodes_the_offset(tmp_path):
    """End to end: manifest -> index v4 -> the frame the dataset actually requests.

    The requested frame is recorded rather than inferred from pixels, so this does not depend
    on what a decoder does with a flat frame.
    """
    from rynnvla.datasets.vla_datasets import latent_pretrain as lp

    entry = {"dataset": "Windowed", "episode_id": "ep0", "caption": "pour the water",
             "gap": 4, "latent_dim": 4,
             "views": [{"view": "head", "video_path": str(tmp_path / "v.mp4"),
                        "latent_path": str(tmp_path / "l.npz"), "num_latents": 40,
                        "fps": 20.0, "pair_stride": 4, "start_frame": 1956}]}
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(json.dumps(entry) + "\n", encoding="utf-8")
    index = tmp_path / "idx" / "i.npz"
    meta = lp.build_index(str(manifest), str(index), 6, 0.25)
    assert meta["kept"] == {"Windowed": 1}, meta
    with np.load(index) as z:
        assert int(z["version"][0]) == 4
        assert z["ep_start_frame"].tolist() == [1956]
        assert z["view_start_frame"].tolist() == [1956]

    ds = lp.LatentPretrainDataset(data_path=str(index), action_chunk_size=30,
                                  use_delta_action=False, processor=None,
                                  num_view_slots=NUM_VIEW_SLOTS, latent_chunk=6,
                                  latent_step_seconds=0.25, latent_action_dim=4)
    calls = []
    real = lp._read_frame
    monkey = lambda video, frame, view=None: calls.append(frame) or torch.zeros(8, 8, 3, dtype=torch.uint8)
    lp._read_frame = monkey
    try:
        ds.load_images(0, 7)  # latent start 7 -> source frame 1956 + 7*4
    finally:
        lp._read_frame = real
    assert calls == [1956 + 7 * 4], calls

    # A v2 index has no offset column; reading it must yield zeros, not an error. Both offset
    # columns have to go, because a real v2 index predates windowed labeling entirely.
    with np.load(index) as z:
        arrays = {k: z[k] for k in z.files
                  if k not in ("ep_start_frame", "view_start_frame")}
    arrays["version"] = np.asarray([2])
    v2 = tmp_path / "idx" / "v2.npz"
    np.savez(v2, **arrays)
    ds2 = lp.LatentPretrainDataset(data_path=str(v2), action_chunk_size=30,
                                   use_delta_action=False, processor=None,
                                   num_view_slots=NUM_VIEW_SLOTS, latent_chunk=6,
                                   latent_step_seconds=0.25, latent_action_dim=4)
    assert ds2._ep_start_frame.tolist() == [0]
    assert ds2._view_start_frame.tolist() == [0]
    calls.clear()
    lp._read_frame = monkey
    try:
        ds2.load_images(0, 7)
    finally:
        lp._read_frame = real
    assert calls == [7 * 4], calls


def test_index_carries_a_start_frame_per_view_not_one_per_episode(tmp_path):
    """Regression for the v3 index: one episode-level start_frame applied to every view.

    The views of one episode are concatenated into different files at different offsets, so
    collapsing them to views[0]'s value asks one view for a frame past the end of its own file
    (IndexError -> 8 exhausted retries -> the rank dies) and another for a frame that decodes
    fine but belongs to a different minute of the recording, silently pairing the image with
    the wrong latents. Measured over the real manifest the per-view offset disagrees within an
    episode for 97.0% of Droid and 93.8% of BEHAVIOR-1K multi-view records.

    The offsets below are a real Droid episode (episode_000405).
    """
    from rynnvla.datasets.vla_datasets import latent_pretrain as lp

    views = [("wrist_left", 119962), ("global", 23779), ("side", 23779)]
    entry = {"dataset": "Windowed", "episode_id": "ep0", "caption": "pour the water",
             "gap": 4, "latent_dim": 4,
             "views": [{"view": name, "video_path": str(tmp_path / f"{name}.mp4"),
                        "latent_path": str(tmp_path / f"{name}.npz"), "num_latents": 40,
                        "fps": 20.0, "pair_stride": 4, "start_frame": sf}
                       for name, sf in views]}
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(json.dumps(entry) + "\n", encoding="utf-8")
    index = tmp_path / "idx" / "i.npz"
    lp.build_index(str(manifest), str(index), 6, 0.25)

    with np.load(index) as z:
        # views are written in manifest order, so the column lines up with `views` above
        assert z["view_start_frame"].tolist() == [sf for _, sf in views]
        # ep_start_frame is retained as views[0]'s labeling-window offset for provenance (it is
        # what a manifest rebuild cross-checks against the source JSON); the decode path must NOT
        # read it -- per-view offsets live in view_start_frame, and using the episode-level column
        # would misalign every view after the first.
        assert z["ep_start_frame"].tolist() == [views[0][1]]

    ds = lp.LatentPretrainDataset(data_path=str(index), action_chunk_size=30,
                                  use_delta_action=False, processor=None,
                                  num_view_slots=NUM_VIEW_SLOTS, latent_chunk=6,
                                  latent_step_seconds=0.25, latent_action_dim=4)
    calls = []
    real = lp._read_frame
    monkey = lambda video, frame, view=None: calls.append(
        (os.path.basename(video), frame)) or torch.zeros(8, 8, 3, dtype=torch.uint8)
    lp._read_frame = monkey
    try:
        ds.load_images(0, 7)  # latent start 7 -> each view's own offset + 7*4
    finally:
        lp._read_frame = real
    assert sorted(calls) == sorted((f"{name}.mp4", sf + 7 * 4) for name, sf in views), calls


def test_wrong_width_and_stride_are_refused(corpus):
    write_latent(Path(corpus.latent_dir) / "TinySet" / "ep0" / "head" / "latent.npz",
                 num_latents=9, dim=256, fps=20.0)
    run_rebuild(corpus, datasets="TinySet")
    report = json.loads((Path(corpus.out_dir) / "rebuild_report.n1.shard0000.json").read_text())
    assert report["datasets"][0]["skips"]["latent_dim_mismatch:[256]"] == 1

    write_latent(Path(corpus.latent_dir) / "TinySet" / "ep0" / "head" / "latent.npz",
                 num_latents=9, fps=20.0, pair_stride=1)
    _run(["--data-root", corpus.data_root, "--latent-dir", corpus.latent_dir,
          "--out-dir", corpus.out_dir, "--datasets", "TinySet", "--force",
          "--expected-pair-stride", "4"])
    report = json.loads((Path(corpus.out_dir) / "rebuild_report.n1.shard0000.json").read_text())
    assert report["datasets"][0]["skips"]["pair_stride_mismatch:1"] == 1


def test_unsafe_names_cannot_escape_the_latent_root(corpus):
    write_source_json(Path(corpus.data_root) / "Escape.json",
                      [episode("Escape", "../TinySet/ep0", ["head"])])
    parts = run_rebuild(corpus, datasets="Escape")
    assert parts == {"Escape.n1.shard0000.jsonl": []}
    report = json.loads((Path(corpus.out_dir) / "rebuild_report.n1.shard0000.json").read_text())
    assert report["datasets"][0]["skips"]["unsafe_name"] == 1


def test_unmapped_camera_is_fatal(corpus):
    write_source_json(Path(corpus.data_root) / "Odd.json", [episode("Odd", "ep0", ["ceiling_cam"])])
    write_latent(Path(corpus.latent_dir) / "Odd" / "ep0" / "ceiling_cam" / "latent.npz")
    with pytest.raises(SystemExit) as excinfo:
        _run(["--data-root", corpus.data_root, "--latent-dir", corpus.latent_dir,
              "--out-dir", corpus.out_dir, "--datasets", "Odd"])
    assert excinfo.value.code == 2


def test_duplicate_episodes_are_counted_once(corpus):
    episodes = [episode("SyntheticBase", "chunk-000/ep0", ["head"])] * 2
    write_source_json(Path(corpus.data_root) / "Dupes.json", episodes)
    write_latent(Path(corpus.latent_dir) / "SyntheticBase" / "chunk-000/ep0" / "head" / "latent.npz")
    parts = run_rebuild(corpus, datasets="Dupes")
    # dataset field inside the entry says SyntheticBase, so the part is named after the JSON
    assert len(parts["Dupes.n1.shard0000.jsonl"]) == 1
    report = json.loads((Path(corpus.out_dir) / "rebuild_report.n1.shard0000.json").read_text())
    assert report["datasets"][0]["skips"]["duplicate_episode"] == 1


def test_unsupported_sources_are_left_out_of_the_manifest(corpus):
    """A zarr/frame directory cannot be decoded by latent_pretrain._read_frame, so indexing it
    would only produce samples the __getitem__ retry loop silently throws away."""
    src = Path(corpus.data_root) / "Zarrish.json"
    ep = episode("Zarrish", "ep0", ["head"])
    ep["views"]["head"]["video_path"] = "/datasets/Zarrish/2025-09-20-17-42-51-000000"  # no suffix
    write_source_json(src, [ep])
    write_latent(Path(corpus.latent_dir) / "Zarrish" / "ep0" / "head" / "latent.npz", fps=0.0)
    parts = run_rebuild(corpus, datasets="Zarrish")
    assert parts == {"Zarrish.n1.shard0000.jsonl": []}
    report = json.loads((Path(corpus.out_dir) / "rebuild_report.n1.shard0000.json").read_text())
    assert report["datasets"][0]["skips"]["unsupported_source:dir"] == 1
    # The same episode with a decodable suffix is kept, so the rule is about the source kind.
    ep["views"]["head"]["video_path"] = "/datasets/Zarrish/clip.mp4"
    write_source_json(src, [ep])
    _run(["--data-root", corpus.data_root, "--latent-dir", corpus.latent_dir,
          "--out-dir", corpus.out_dir, "--datasets", "Zarrish", "--force"])
    assert len(_read_parts(corpus.out_dir)["Zarrish.n1.shard0000.jsonl"]) == 1


def test_missing_fps_is_measured_or_filled_not_left_at_zero(corpus):
    """fps 0 in the latent meta would let build_index apply its own default and rescale every
    chunk of that corpus: RoboMIND2.0's real rate is ~14 Hz (median of 1096 values measured
    from HDF5 timestamps), so a default of 30 gives stride 2 instead of 1 and each chunk spans
    2.86 s of video instead of 1.43 s -- 2x too long, silently."""
    src = Path(corpus.data_root) / "NoFps.json"
    write_source_json(src, [episode("NoFps", "ep0", ["head"])])
    write_latent(Path(corpus.latent_dir) / "NoFps" / "ep0" / "head" / "latent.npz", fps=0.0)
    parts = run_rebuild(corpus, datasets="NoFps")
    entry = parts["NoFps.n1.shard0000.jsonl"][0]
    assert entry["views"][0]["fps"] == 30.0  # --default-fps, and counted
    report = json.loads((Path(corpus.out_dir) / "rebuild_report.n1.shard0000.json").read_text())
    assert report["datasets"][0]["skips"]["fps_fallback:NoFps:30"] == 1

    # A dataset in STATIC_FPS gets the table value instead of the default.
    write_source_json(Path(corpus.data_root) / "RoboMIND2.0.json",
                      [episode("RoboMIND2.0", "ep0", ["head"])])
    write_latent(Path(corpus.latent_dir) / "RoboMIND2.0" / "ep0" / "head" / "latent.npz", fps=0.0)
    parts = run_rebuild(corpus, datasets="RoboMIND2.0")
    entry = parts["RoboMIND2.0.n1.shard0000.jsonl"][0]
    assert entry["views"][0]["fps"] == rebuild.STATIC_FPS["RoboMIND2.0"] == 14.0


def test_hdf5_fps_comes_from_the_trajectory_timestamps(tmp_path, monkeypatch):
    """RoboMIND2.0 latents carry fps 0; the frame rate is recoverable from the source file."""
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "trajectory.hdf5"
    with h5py.File(path, "w") as f:
        # 2225 frames over 22 s, the shape and scale of a real RoboMIND2.0 episode
        f.create_dataset("camera_observations/timestamp",
                         data=np.linspace(1747982392, 1747982414, 2225).astype(np.int64))
    assert abs(rebuild.hdf5_fps(str(path)) - 2224 / 22) < 1e-6
    assert rebuild.hdf5_fps(str(tmp_path / "absent.hdf5")) is None

    with h5py.File(tmp_path / "short.hdf5", "w") as f:
        f.create_dataset("camera_observations/timestamp", data=np.array([7], np.int64))
    assert rebuild.hdf5_fps(str(tmp_path / "short.hdf5")) is None


def test_existing_part_is_reused_and_force_rebuilds_it(corpus):
    first = run_rebuild(corpus, datasets="TinySet")
    assert len(first["TinySet.n1.shard0000.jsonl"]) == 1
    _run(["--data-root", corpus.data_root, "--latent-dir", corpus.latent_dir,
          "--out-dir", corpus.out_dir, "--datasets", "TinySet"])
    report = json.loads((Path(corpus.out_dir) / "rebuild_report.n1.shard0000.json").read_text())
    assert report["datasets"][0]["reused"] is True
    # A latent that vanished since the last run must be reflected by --force, not by resume:
    # the part stays (the slice was scanned) but no longer claims that episode.
    os.unlink(Path(corpus.latent_dir) / "TinySet" / "ep0" / "head" / "latent.npz")
    _run(["--data-root", corpus.data_root, "--latent-dir", corpus.latent_dir,
          "--out-dir", corpus.out_dir, "--datasets", "TinySet", "--force"])
    assert _read_parts(corpus.out_dir) == {"TinySet.n1.shard0000.jsonl": []}


# ── sharding and combining ────────────────────────────────────────────────────

def test_shards_partition_the_corpus_and_combine_merges_them(corpus):
    seen = []
    for shard in (0, 1):
        _run(["--data-root", corpus.data_root, "--latent-dir", corpus.latent_dir,
              "--out-dir", corpus.out_dir, "--datasets", "SyntheticBase",
              "--shard", str(shard), "--num-shards", "2"])
        parts = _read_parts(corpus.out_dir)
        seen += [e["episode_id"] for e in parts[f"SyntheticBase.n2.shard{shard:04d}.jsonl"]]
    assert sorted(seen) == ["chunk-000/ep0", "chunk-000/ep1", "chunk-000/ep2"]

    _run(["--out-dir", corpus.out_dir, "--num-shards", "2", "--expect-shards", "2",
          "--combine-only", "--name", "combined"])
    combined = Path(corpus.out_dir) / "manifests" / "combined.jsonl"
    lines = [json.loads(x) for x in combined.read_text(encoding="utf-8").splitlines()]
    assert sorted(e["episode_id"] for e in lines) == sorted(seen)
    assert manifest_format(str(combined)) == "jsonl"


def test_an_empty_slice_still_counts_as_a_finished_shard(corpus):
    """A shard whose slice holds no labeled episode must not block the whole combine.

    At --num-shards 32 the smallest datasets hand some shards
    nothing, and labeling has not reached other slices at all. If "kept nothing" published no
    part, --expect-shards could not distinguish that from a job that never ran, and preparation
    would refuse a complete sweep.
    """
    for shard in (0, 1):
        _run(["--data-root", corpus.data_root, "--latent-dir", corpus.latent_dir,
              "--out-dir", corpus.out_dir, "--datasets", "TinySet",
              "--shard", str(shard), "--num-shards", "2"])
    # TinySet has one episode, so exactly one shard kept it and the other published an empty part.
    parts = _read_parts(corpus.out_dir)
    assert sorted(parts) == ["TinySet.n2.shard0000.jsonl", "TinySet.n2.shard0001.jsonl"]
    assert sorted(len(v) for v in parts.values()) == [0, 1]

    _run(["--out-dir", corpus.out_dir, "--num-shards", "2", "--expect-shards", "2",
          "--combine-only", "--name", "tiny"])
    combined = (Path(corpus.out_dir) / "manifests" / "tiny.jsonl").read_text(encoding="utf-8")
    assert len([line for line in combined.splitlines() if line.strip()]) == 1


def test_combine_refuses_a_sharding_whose_shards_are_missing(corpus):
    """And it must refuse BEFORE writing anything: a partial manifest at the path preparation
    reads is indistinguishable from a complete one, so build_index and stats would train on
    whatever fraction happened to be there. This fired for real on 2026-09-16, when a fuse
    disconnect killed 28 of 32 shards and the combine still published ~12% of the episodes."""
    _run(["--data-root", corpus.data_root, "--latent-dir", corpus.latent_dir,
          "--out-dir", corpus.out_dir, "--datasets", "SyntheticBase",
          "--shard", "0", "--num-shards", "3"])
    manifests = Path(corpus.out_dir) / "manifests"
    with pytest.raises(SystemExit, match="missing shards"):
        _run(["--out-dir", corpus.out_dir, "--num-shards", "3", "--expect-shards", "3",
              "--combine-only", "--name", "combined"])
    assert not (manifests / "combined.jsonl").exists()
    assert [p.name for p in manifests.iterdir() if p.name != "parts"] == []


def test_parts_from_different_shardings_are_never_combined(corpus):
    """n1 and n2 parts hold overlapping episodes; mixing them would double-sample silently."""
    run_rebuild(corpus, datasets="SyntheticBase")
    _run(["--data-root", corpus.data_root, "--latent-dir", corpus.latent_dir,
          "--out-dir", corpus.out_dir, "--datasets", "SyntheticBase",
          "--shard", "0", "--num-shards", "2"])
    _run(["--out-dir", corpus.out_dir, "--num-shards", "2", "--combine-only", "--name", "half"])
    lines = (Path(corpus.out_dir) / "manifests" / "half.jsonl").read_text().splitlines()
    assert len(lines) == 2  # only shard 0 of the n2 sharding, never the n1 part


# ── manifest_io ───────────────────────────────────────────────────────────────

def test_iter_manifest_handles_both_formats(tmp_path):
    entries = [{"dataset": "d", "episode_id": f"ep{i}"} for i in range(5)]
    array = tmp_path / "a.json"
    array.write_text(json.dumps(entries, indent=1), encoding="utf-8")
    jsonl = tmp_path / "b.jsonl"
    jsonl.write_text("".join(json.dumps(e) + "\n" for e in entries), encoding="utf-8")
    assert manifest_format(str(array)) == "array"
    assert manifest_format(str(jsonl)) == "jsonl"
    assert list(iter_manifest(str(array))) == entries
    assert list(iter_manifest(str(jsonl))) == entries

    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n\n", encoding="utf-8")
    assert manifest_format(str(empty)) == "empty"
    assert list(iter_manifest(str(empty))) == []

    garbage = tmp_path / "garbage.json"
    garbage.write_text("dataset,episode_id\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not a JSON array or JSONL"):
        manifest_format(str(garbage))


def test_iter_jsonl_names_the_broken_line(tmp_path):
    path = tmp_path / "broken.jsonl"
    path.write_text('{"episode_id": "a"}\n{"episode_id": \n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"broken\.jsonl:2: malformed manifest line"):
        list(iter_manifest(str(path)))


# ── build_index and stats over JSONL ─────────────────────────────────────────

def test_build_index_matches_between_jsonl_and_array_manifests(corpus, tmp_path):
    from rynnvla.datasets.vla_datasets.latent_pretrain import build_index

    parts = run_rebuild(corpus)
    _run(["--out-dir", corpus.out_dir, "--combine-only", "--name", "all"])
    jsonl = str(Path(corpus.out_dir) / "manifests" / "all.jsonl")
    array = str(tmp_path / "all.json")
    with open(array, "w", encoding="utf-8") as handle:
        json.dump([json.loads(line) for line in open(jsonl, encoding="utf-8")], handle)

    stream_meta = build_index(jsonl, str(tmp_path / "idx" / "stream.npz"), 6, 0.25)
    loaded_meta = build_index(array, str(tmp_path / "idx" / "array.npz"), 6, 0.25)
    assert stream_meta["sorted"] is False and loaded_meta["sorted"] is True
    assert stream_meta["kept"] == loaded_meta["kept"] == {"SyntheticBase": 3, "TinySet": 1}
    with np.load(tmp_path / "idx" / "stream.npz") as streamed, \
            np.load(tmp_path / "idx" / "array.npz") as loaded:
        assert set(streamed.files) == set(loaded.files)
        assert int(streamed["version"][0]) == int(loaded["version"][0]) == 4
        for key in ("ep_ds", "ep_nlat", "ep_fps", "ep_stride", "ep_pair_stride",
                    "ep_start_frame", "view_off", "view_role", "view_start_frame",
                    "caption_blob", "caption_off",
                    "video_blob", "video_pos", "latent_blob", "latent_pos"):
            np.testing.assert_array_equal(streamed[key], loaded[key], err_msg=key)
        assert sorted(str(x) for x in streamed["ds_names"]) == ["SyntheticBase", "TinySet"]


def test_build_index_streams_without_materializing_the_manifest(corpus, tmp_path, monkeypatch):
    """The point of JSONL: build_index must not json.load it (parsing costs ~6x the text size)."""
    from rynnvla.datasets.vla_datasets import latent_pretrain as lp

    parts = run_rebuild(corpus)
    _run(["--out-dir", corpus.out_dir, "--combine-only", "--name", "all"])
    jsonl = str(Path(corpus.out_dir) / "manifests" / "all.jsonl")

    def explode(*_args, **_kwargs):
        raise AssertionError("build_index must stream a JSONL manifest, not json.load it")

    monkeypatch.setattr(lp.json, "load", explode)
    meta = lp.build_index(jsonl, str(tmp_path / "idx" / "streamed.npz"), 6, 0.25)
    assert sum(meta["kept"].values()) == 4


def test_stats_over_jsonl_and_sampling_is_deterministic(corpus, tmp_path):
    run_rebuild(corpus)
    _run(["--out-dir", corpus.out_dir, "--combine-only", "--name", "all"])
    manifest = str(Path(corpus.out_dir) / "manifests" / "all.jsonl")

    def run(out_name, *extra):
        out = tmp_path / out_name
        assert stats.main(["--manifest", manifest, "--out", str(out),
                           "--latent-dim", "608", "--workers", "4", *extra]) == 0
        return json.loads(out.read_text())

    full = run("full.json")
    assert full["files"] == 5 and full["episodes"] == 4 and full["episodes_sampled"] == 4
    assert full["sample_rate"] == 1.0

    # The same rate must select the same episodes in every run and in every process: at 0.5
    # blake2b picks SyntheticBase ep0 and ep2 (3 views, 21 latents each) and skips the rest.
    a = run("sample_a.json", "--sample-rate", "0.5")
    b = run("sample_b.json", "--sample-rate", "0.5")
    for key in ("files", "rows", "episodes_sampled", "mean", "std", "min", "max"):
        np.testing.assert_array_equal(np.array(a[key]), np.array(b[key]), err_msg=key)
    assert (a["episodes_sampled"], a["files"], a["rows"]) == (2, 3, 63)
    assert a["episodes"] == 4 and a["sample_rate"] == 0.5
    assert stats.episode_selected("SyntheticBase", "chunk-000/ep1", 0.5) is False
    assert stats.episode_selected("TinySet", "ep0", 0.5) is False

    # argparse writes the rejection to stderr and exits 2, before any file is read.
    with pytest.raises(SystemExit) as excinfo:
        stats.main(["--manifest", manifest, "--out", str(tmp_path / "bad.json")])
    assert excinfo.value.code == 2
    assert not (tmp_path / "bad.json").exists()
