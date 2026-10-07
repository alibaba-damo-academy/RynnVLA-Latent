"""Local safetensors loading and distributed bucket/role sampling; no network."""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from rynnlam.dataset_epic import LAMEpicDataset, WeightedMultiDatasetBatchSampler


def write_scene(path, prefix, flow_dims=3):
    n, h, w = 5, 4, 6
    tensors = {
        "rgb": np.full((n, h, w, 3), 128, np.uint8),
        "depth": np.ones((n, h, w), np.float16),
        "mask": np.ones((n, h, w), np.uint8),
        "flow": np.zeros((n - 1, h, w, flow_dims), np.float16),
        "intrinsics": np.tile(np.eye(3, dtype=np.float32), (n, 1, 1)),
        "extrinsics": np.tile(np.eye(4, dtype=np.float32), (n, 1, 1)),
    }
    save_file({prefix + "__" + k: v for k, v in tensors.items()}, str(path))


@pytest.mark.parametrize("flow_dims", [2, 3])
def test_local_read_uses_file_stem_and_user_cache(tmp_path, monkeypatch, flow_dims):
    root = tmp_path / "data"
    root.mkdir()
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    file = root / "old_camera_name.safetensors"
    write_scene(file, file.stem, flow_dims)
    entry = dict(
        scene_id="Demo/scene/global",
        file=file.name,
        num_frames=5,
        num_flows=4,
        height=4,
        width=6,
        dataset="Demo",
        role="global",
    )
    (root / "manifest_0000.json").write_text(json.dumps([entry]))
    dataset = LAMEpicDataset(
        safetensors_root=str(root),
        max_frame_stride=1,
        min_frame_stride=1,
        max_sample_stride=1,
    )
    sample = dataset[0]
    assert tuple(sample["images"].shape) == (2, 4, 6, 3)
    assert sample["dataset"] == "Demo"
    assert list((tmp_path / "cache" / "rynnlam" / "frame_counts").glob("*.json"))
    assert len(dataset) == 4
    # Changed manifest counts must not reuse the previous cache.
    entry["num_frames"] = 3
    (root / "manifest_0000.json").write_text(json.dumps([entry]))
    assert len(LAMEpicDataset(safetensors_root=str(root), max_frame_stride=1)) == 2


def test_trainer_explicit_manifest(tmp_path):
    from rynnlam.config import RynnLAMConfig
    from _scripts import load_script

    build_data = load_script("train_rynnlam").build_data

    scene = tmp_path / "camera.safetensors"
    write_scene(scene, scene.stem)
    manifest = tmp_path / "selected.json"
    manifest.write_text(
        json.dumps(
            [
                dict(
                    scene_id="Demo/episode/head",
                    file=scene.name,
                    num_frames=5,
                    num_flows=4,
                    height=4,
                    width=6,
                    dataset="Demo",
                )
            ]
        )
    )
    config = RynnLAMConfig(
        safetensors_root=str(tmp_path),
        manifest_path=str(manifest),
        cache_dir=str(tmp_path / "cache"),
        num_workers=0,
        batch_size=1,
        min_frame_stride=1,
        max_frame_stride=1,
        max_sample_stride=1,
        dataset_sampling_temperature=1.0,
    )
    dataset, _, loader = build_data(config)
    assert len(dataset) == 4
    assert next(iter(loader))["images"].shape == (1, 2, 4, 6, 3)
    with pytest.raises(FileNotFoundError):
        LAMEpicDataset(manifest_path=str(tmp_path / "absent.json"))


class FakeDataset:
    def __init__(self):
        self.ids = ["Robot/a", "Robot/b", "Human/c"]
        self._st_meta = {
            self.ids[0]: dict(dataset="Robot", height=4, width=6, role="head"),
            self.ids[1]: dict(dataset="Robot", height=6, width=4, role="wrist_left"),
            self.ids[2]: dict(dataset="Human", height=4, width=6, role="head"),
        }

    def __len__(self):
        return 120

    def _idx_to_scene_frame(self, idx):
        return self.ids[idx // 40], idx % 40


def test_distributed_bucket_role_sampling():
    data = FakeDataset()
    kwargs = dict(
        dataset=data,
        batch_size=2,
        num_replicas=2,
        seed=13,
        total_samples=60,
        role_multipliers={"wrist_left": 2.0},
    )
    ranks = [WeightedMultiDatasetBatchSampler(rank=r, **kwargs) for r in range(2)]
    batches = [list(s) for s in ranks]
    assert batches[0] == list(ranks[0])
    assert len(batches[0]) == len(batches[1])
    indices = [set(i for batch in rank for i in batch) for rank in batches]
    assert indices[0].isdisjoint(indices[1])  # no replacement in this fixture
    counts = Counter()
    for rank in batches:
        for batch in rank:
            scenes = [data._idx_to_scene_frame(i)[0] for i in batch]
            assert (
                len(
                    {
                        (data._st_meta[s]["height"], data._st_meta[s]["width"])
                        for s in scenes
                    }
                )
                == 1
            )
            counts.update(
                data._st_meta[s]["role"] for s in scenes if s.startswith("Robot/")
            )
    assert counts["wrist_left"] > counts["head"]


# The upstream RynnLAM suite also pinned an object-store reader here (credential discovery from
# the environment, bucket-keyed header cache). This release reads local paths only and ships no
# such reader, so those tests have nothing to cover. tests/rynnlam/test_config.py pins the
# absence of the corresponding config fields instead.


@pytest.mark.parametrize("flow_dims", [2, 3])
def test_local_flow_rejects_truncated_interval(tmp_path, flow_dims):
    from safetensors.numpy import load_file

    path = tmp_path / "short.safetensors"
    write_scene(path, "short", flow_dims)
    tensors = load_file(str(path))
    tensors["short__flow"] = tensors["short__flow"][:2].copy()
    save_file(tensors, str(path))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            [dict(scene_id="short", file=path.name, height=4, width=6, num_frames=5)]
        )
    )
    data = LAMEpicDataset(
        manifest_path=str(manifest),
        cache_dir=str(tmp_path / "cache"),
        min_frame_stride=3,
        max_frame_stride=3,
        max_sample_stride=3,
    )
    with pytest.raises(ValueError, match="Incomplete flow interval"):
        data._load_from_safetensors("short", 0)


@pytest.mark.parametrize(
    "kwargs",
    [{"temperature": x} for x in [0, -1, float("nan"), float("inf")]]
    + [
        {field: {"unused": x}}
        for field in ["dataset_multipliers", "role_multipliers"]
        for x in [-1, float("nan"), float("inf")]
    ],
)
def test_sampler_rejects_invalid_weights(kwargs):
    with pytest.raises(ValueError):
        WeightedMultiDatasetBatchSampler(FakeDataset(), 2, 1, 0, **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dataset_multipliers": {"Robot": 0, "Human": 0}},
        {"role_multipliers": {"head": 0, "wrist_left": 0}},
        {"dataset_multipliers": {"Robot": 0}, "role_multipliers": {"head": 0}},
    ],
)
def test_sampler_rejects_no_positive_groups(kwargs):
    with pytest.raises(ValueError, match="No positive weighted groups"):
        WeightedMultiDatasetBatchSampler(FakeDataset(), 2, 1, 0, **kwargs)


@pytest.mark.parametrize("drop_last", [False, True])
@pytest.mark.parametrize(
    "kwargs,allowed",
    [
        ({"dataset_multipliers": {"Human": 0}}, {0, 1}),
        ({"role_multipliers": {"head": 0}}, {1}),
        ({"dataset_multipliers": {"Human": 0}, "role_multipliers": {"head": 0}}, {1}),
    ],
)
def test_zero_weights_never_sample(kwargs, allowed, drop_last):
    samplers = [
        WeightedMultiDatasetBatchSampler(
            FakeDataset(), 3, 2, rank, total_samples=31, drop_last=drop_last, **kwargs
        )
        for rank in range(2)
    ]
    for sampler in samplers:
        batches = list(sampler)
        assert batches and len(batches) == len(sampler) == len(samplers[0])
        assert {i // 40 for b in batches for i in b} <= allowed
        assert all(isinstance(v, np.ndarray) for v in sampler.group_to_indices.values())
        if kwargs.get("role_multipliers", {}).get("head") == 0:
            assert sampler._per_dataset_n["Human"] == 0


@pytest.mark.parametrize("skip_scan", [False, True])
def test_manifest_metadata_needs_no_executor_and_preserves_cache_hash(
    tmp_path, monkeypatch, skip_scan
):
    import hashlib
    import rynnlam.dataset_epic as module

    entries = [
        dict(
            scene_id=f"demo/{i}",
            file="not-read.safetensors",
            num_frames=5,
            height=4,
            width=6,
        )
        for i in range(25)
    ]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(entries))
    monkeypatch.setattr(
        module, "ThreadPoolExecutor", lambda **kw: pytest.fail("No raw scans needed")
    )
    data = LAMEpicDataset(
        manifest_path=str(manifest),
        cache_dir=str(tmp_path / "cache"),
        max_frame_stride=1,
        skip_frame_scan=skip_scan,
    )
    identity = [(sid, data._st_meta[sid]) for sid in data.scene_list]
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[
        :20
    ]
    assert (tmp_path / "cache" / f"frame_counts_{digest}_25.json").is_file()
    assert len(data) == 100 and data.scene_meta["demo/0"]["height"] == 4


def test_mixed_manifest_only_scans_missing_raw_metadata(tmp_path, monkeypatch):
    import rynnlam.dataset_epic as module
    from concurrent.futures import ThreadPoolExecutor

    raw = tmp_path / "depth.npz"
    np.savez(raw, depth=np.ones((7, 4, 6), np.float32))
    meta = tmp_path / "meta.json"
    meta.write_text(
        json.dumps(
            [
                dict(scene_id=sid, rgb="unused", depth_npz=str(raw))
                for sid in ["ready", "raw"]
            ]
        )
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            [
                dict(
                    scene_id="ready",
                    file="unused.safetensors",
                    height=4,
                    width=6,
                    num_frames=5,
                )
            ]
        )
    )
    submitted = []

    class TrackingPool(ThreadPoolExecutor):
        def submit(self, fn, sid):
            submitted.append(sid)
            return super().submit(fn, sid)

    monkeypatch.setattr(module, "ThreadPoolExecutor", TrackingPool)
    data = LAMEpicDataset(
        meta_json=str(meta),
        manifest_path=str(manifest),
        cache_dir=str(tmp_path / "cache"),
        max_frame_stride=1,
    )
    assert submitted == ["raw"]
    assert data.scene_frame_counts == {"ready": 4, "raw": 6}


def test_scan_futures_are_bounded():
    from concurrent.futures import Future
    from rynnlam.dataset_epic import _bounded_scan_futures

    class Pool:
        outstanding = 0
        peak = 0

        def submit(self, fn, item):
            self.outstanding += 1
            self.peak = max(self.peak, self.outstanding)
            future = Future()
            future.set_result(fn(item))
            return future

    pool = Pool()
    results = []
    for future in _bounded_scan_futures(pool, lambda x: x, list(range(27)), 4):
        results.append(future.result())
        pool.outstanding -= 1
    assert sorted(results) == list(range(27)) and pool.peak == 4


@pytest.mark.parametrize("total", [60, 240])
@pytest.mark.parametrize("drop_last", [False, True])
def test_positive_sampling_matches_legacy_rng(total, drop_last):
    import torch

    data = FakeDataset()
    sampler = WeightedMultiDatasetBatchSampler(
        data,
        3,
        2,
        0,
        total_samples=total,
        seed=13,
        temperature=1.5,
        role_multipliers={"wrist_left": 2},
        drop_last=drop_last,
    )
    g = torch.Generator().manual_seed(13)
    batches = []
    weights = {"Robot": 80 ** (1 / 1.5), "Human": 40 ** (1 / 1.5)}
    for ds in weights:
        budget = max(1, round(total * weights[ds] / sum(weights.values())))
        groups = [
            (k, list(v)) for k, v in sampler.group_to_indices.items() if k[0] == ds
        ]
        denominator = sum(
            len(v) * (2 if k[2] == "wrist_left" else 1) for k, v in groups
        )
        for key, indices in groups:
            n = max(
                1,
                round(
                    budget
                    * len(indices)
                    * (2 if key[2] == "wrist_left" else 1)
                    / denominator
                ),
            )
            pos = (
                torch.randint(len(indices), (n,), generator=g)
                if n > len(indices)
                else torch.randperm(len(indices), generator=g)[:n]
            ).tolist()
            drawn = [indices[p] for p in pos]
            batches.extend(
                drawn[start : start + 3]
                for start in range(0, len(drawn), 3)
                if not drop_last or len(drawn[start : start + 3]) == 3
            )
    batches = [batches[i] for i in torch.randperm(len(batches), generator=g).tolist()]
    assert list(sampler) == batches[::2][: len(batches) // 2]


def test_compact_bucket_uses_scene_spans_and_legacy_order():
    import torch
    from rynnlam.dataset_epic import BucketedDistributedBatchSampler

    data = FakeDataset()
    data.scene_list = data.ids
    data.scene_frame_counts = dict.fromkeys(data.ids, 40)
    data._idx_to_scene_frame = lambda _: pytest.fail("Must index by scene spans")
    sampler = BucketedDistributedBatchSampler(
        data, batch_size=3, num_replicas=2, rank=0, seed=13
    )
    g = torch.Generator().manual_seed(13)
    batches = []
    for indices in (list(range(40)) + list(range(80, 120)), list(range(40, 80))):
        drawn = [indices[i] for i in torch.randperm(len(indices), generator=g).tolist()]
        batches.extend(drawn[i : i + 3] for i in range(0, len(drawn), 3))
    batches = [batches[i] for i in torch.randperm(len(batches), generator=g).tolist()]
    assert list(sampler) == batches[::2][: len(batches) // 2]
    assert all(isinstance(v, np.ndarray) for v in sampler.bucket_to_indices.values())
