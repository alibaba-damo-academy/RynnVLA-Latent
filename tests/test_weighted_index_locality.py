"""The weighted (temperature-sampled) Stage-1 index must stay compatible with the locality shuffle.

Regression cover for a defect that cost the alpha=0.7 arm 5.2x wall clock. The chain:

``_build_weighted_index`` laid the virtual index out DATASET-major -- one contiguous segment per
dataset, remapped inside by ``size // target`` -- and per-element ``round`` made its total come
out one short of the natural total (121,058,083 vs 121,058,084 on the real corpus).
``DistributedBatchSampler._episode_sample_counts`` compares ``sum(counts)`` to ``len(dataset)``
and refuses on any mismatch, so the locality permutation silently degraded to a plain randperm.
The randperm destroyed the latent LRU's reuse, nearly every sample re-read a whole npz to take 6
of its rows, and in a synchronous 64-rank run one rank's cold read stalls every rank at the
gradient all-reduce -- which the step probe books as ``fwd_bwd``, not ``data``, so the run looks
compute-bound when it is waiting. Measured: 0.841 -> 4.13 s/step, slow-sample events 0.12 ->
9.07 per step.

Two independent reasons the old layout could not be rescued by aligning the totals, both pinned
below: per-element rounding does not conserve the total (test_largest_remainder_*), and a
dataset-major layout is not tiled by per-episode counts at all, so feeding those counts to the
sampler would produce a permutation that is not a permutation
(test_dataset_major_layout_is_not_tiled_by_episode_counts).

The fixture is deliberately shaped so the two rounding schemes disagree: "Big" needs 28 slots
over 4 episodes whose exact shares are [10.84, 10.84, 4.52, 1.81]. Per-element rounding gives
[11, 11, 5, 2] = 29; largest remainder gives [11, 11, 4, 2] = 28.
"""
import importlib.util
import json
import math
import os

import numpy as np
import pytest

from rynnvla.constants import NUM_VIEW_SLOTS
from rynnvla.datasets.vla_datasets import latent_pretrain as lp

try:
    from rynnvla.training.sampler import DistributedBatchSampler
except ImportError:  # pragma: no cover - dev pods without the training extras
    # Same by-path load tests/test_sampler_locality.py uses: rynnvla.training.__init__ pulls in
    # trainer -> deepspeed, which the sampler itself does not need.
    def _load(name, rel):
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), rel)
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    DistributedBatchSampler = _load("_sampler_wi", "rynnvla/training/sampler.py").DistributedBatchSampler


CHUNK, STEP_SECONDS, FPS, PAIR_STRIDE, LATENT_DIM = 6, 0.25, 20.0, 4, 4
# stride = round(fps * latent_step_seconds / pair_stride) = 1, so start_stride = 3 and
# span = 6: windows(nlat) = max(0, (nlat - 6) // 3 + 1).
SPEC = {
    # name: [num_latents per episode] -> windows per episode -> dataset size
    "Big": [40, 40, 20, 10],      # 12, 12, 5, 2 -> 31  (downsampled at alpha=0.7)
    "Small": [10, 6],             #  2, 1        ->  3  (upsampled at alpha=0.7)
}
NATURAL_TOTAL = 34


def _windows(nlat):
    stride = int(round(FPS * STEP_SECONDS / PAIR_STRIDE))
    start_stride = max(1, (stride * CHUNK) // 2)
    span = (CHUNK - 1) * stride + 1
    return max(0, (nlat - span) // start_stride + 1)


def _build(tmp_path, spec=SPEC, latent_chunk=CHUNK):
    lines = []
    for name, nlats in spec.items():
        for i, nlat in enumerate(nlats):
            lines.append(json.dumps({
                "dataset": name, "episode_id": f"ep{i}", "caption": f"{name} ep{i}",
                "gap": 4, "latent_dim": LATENT_DIM,
                "views": [{"view": "head", "video_path": str(tmp_path / f"{name}{i}.mp4"),
                           "latent_path": str(tmp_path / f"{name}{i}.npz"),
                           "num_latents": nlat, "fps": FPS,
                           "pair_stride": PAIR_STRIDE, "start_frame": 0}],
            }))
    manifest = tmp_path / "m.jsonl"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    index = tmp_path / "i.npz"
    lp.build_index(str(manifest), str(index), latent_chunk, STEP_SECONDS)
    return str(index)


def _dataset(index, weights=None):
    return lp.LatentPretrainDataset(
        data_path=index, action_chunk_size=30, use_delta_action=False, processor=None,
        num_view_slots=NUM_VIEW_SLOTS, latent_chunk=CHUNK,
        latent_step_seconds=STEP_SECONDS, latent_action_dim=LATENT_DIM,
        dataset_weights=weights)


def _alpha_weights(ds, alpha=0.7):
    """Exactly what scripts/build_dataset_weights.py writes into the mixture: windows ** alpha."""
    windows = np.bincount(ds._ep_ds.astype(np.int64),
                          weights=ds._ep_starts.astype(np.float64),
                          minlength=len(ds._ds_names))
    return {ds._ds_names[i]: float(windows[i]) ** alpha for i in range(len(ds._ds_names))}


def _legacy_targets(ds, weights):
    """The dataset-major per-dataset targets, i.e. what len(dataset) was before the relayout."""
    total = int(ds._cum_lengths[-1])
    wsum = sum(float(weights[n]) for n in ds._ds_names)
    sizes = np.bincount(ds._ep_ds.astype(np.int64), weights=ds._ep_starts.astype(np.float64),
                        minlength=len(ds._ds_names))
    return {ds._ds_names[i]: max(1, round(float(weights[ds._ds_names[i]]) / wsum * total))
            for i in range(len(ds._ds_names))}, sizes.astype(np.int64), total


# ── largest remainder ────────────────────────────────────────────────────────

def test_largest_remainder_conserves_the_total_where_naive_rounding_does_not():
    exact = np.array([10.838709677, 10.838709677, 4.516129032, 1.806451613])
    got = lp._largest_remainder(exact, 28)
    assert got.tolist() == [11, 11, 4, 2]
    assert int(got.sum()) == 28
    # The point of the helper: per-element rounding overshoots and would move len(dataset).
    assert int(np.round(exact).sum()) == 29


def test_largest_remainder_is_deterministic_and_ties_break_by_position():
    exact = np.array([1.5, 1.5, 1.5, 1.5])                       # sums to 6
    a = lp._largest_remainder(exact, 6)
    b = lp._largest_remainder(exact, 6)
    assert a.tolist() == b.tolist() == [2, 2, 1, 1]
    # All four remainders tie at 0.5, so the two extra units go to the first two by position.
    # Every rank derives the same allocation from the same index without communicating.
    tied = np.array([1.5, 0.5, 1.0])                              # sums to 3
    assert lp._largest_remainder(tied, 3).tolist() == [2, 0, 1]


def test_largest_remainder_refuses_proportions_that_do_not_sum_to_total():
    with pytest.raises(ValueError, match="do not sum to total"):
        lp._largest_remainder(np.array([1.0, 1.0, 1.0]), 99)


# ── the guard that was tripping ──────────────────────────────────────────────

def test_slot_counts_tile_the_weighted_index_so_the_sampler_guard_passes(tmp_path):
    ds = _dataset(_build(tmp_path), weights=None)
    ds._build_weighted_index(_alpha_weights(ds))
    counts = np.asarray(ds.sampler_slot_counts, dtype=np.int64)
    # sampler.py: int(counts.sum()) != len(dataset) -> refuse -> randperm -> no locality.
    assert int(counts.sum()) == len(ds)
    assert counts.ndim == 1 and counts.size > 0 and not (counts < 0).any()


def test_dataset_major_layout_is_not_tiled_by_episode_counts(tmp_path):
    """Why aligning the totals alone could never have fixed it.

    Under the old layout, dataset "Big" owned one contiguous virtual segment and positions
    inside it were remapped by ``size // target``, so virtual adjacency jumped across episodes.
    The sampler's permutation emits ``cumsum(counts)[e] + r``, which is only a permutation when
    ``[base[e], base[e] + counts[e])`` all belong to episode e. This asserts the property the
    new layout has and the old one did not: every episode's virtual positions are contiguous
    and map inside that episode.
    """
    ds = _dataset(_build(tmp_path), weights=None)
    ds._build_weighted_index(_alpha_weights(ds))
    slots = np.asarray(ds.sampler_slot_counts, dtype=np.int64)
    vbase = np.concatenate([[0], np.cumsum(slots)[:-1]])
    for e in range(len(slots)):
        hit = [v for v in range(len(ds)) if ds._resolve_index(ds._map_virtual_index(v))[0] == e]
        assert hit == list(range(int(vbase[e]), int(vbase[e]) + int(slots[e]))), e


def test_len_is_unchanged_by_the_relayout_so_max_steps_stays_put(tmp_path):
    """max_steps is derived from len(dataset) at submit time and sits in RESUME_CRITICAL_FIELDS.

    If the relayout moved len(dataset) by even one, max_steps would move with it and every
    existing checkpoint of the arm would be rejected on resume.
    """
    ds = _dataset(_build(tmp_path), weights=None)
    weights = _alpha_weights(ds)
    legacy, _, _ = _legacy_targets(ds, weights)
    ds._build_weighted_index(weights)
    assert len(ds) == sum(legacy.values()) == NATURAL_TOTAL
    global_batch = 8
    assert math.ceil(len(ds) / global_batch) == math.ceil(sum(legacy.values()) / global_batch)


def test_per_dataset_slots_reproduce_the_legacy_targets_exactly(tmp_path):
    ds = _dataset(_build(tmp_path), weights=None)
    weights = _alpha_weights(ds)
    legacy, sizes, _ = _legacy_targets(ds, weights)
    ds._build_weighted_index(weights)
    slots = np.asarray(ds.sampler_slot_counts, dtype=np.int64)
    for di, name in enumerate(ds._ds_names):
        got = int(slots[ds._ep_ds == di].sum())
        assert got == legacy[name], (name, got, legacy[name])
    # The fixture's shape: Big is downsampled 31 -> 28, Small is upsampled 3 -> 6.
    assert legacy["Big"] == 28 and legacy["Small"] == 6


def test_alpha_one_reproduces_natural_counts_exactly(tmp_path):
    """alpha=1 must be a no-op, otherwise the natural arm is not the control it claims to be."""
    ds = _dataset(_build(tmp_path), weights=None)
    natural = np.asarray(ds.episode_lengths, dtype=np.int64)
    ds._build_weighted_index(_alpha_weights(ds, alpha=1.0))
    assert np.asarray(ds.sampler_slot_counts, dtype=np.int64).tolist() == natural.tolist()
    assert len(ds) == int(natural.sum()) == NATURAL_TOTAL


def test_zero_weight_still_buys_one_slot(tmp_path):
    """Carried over from the dataset-major layout on purpose: weighting a source 0 does NOT
    exclude it. Recorded here so nobody rediscovered that as a way to drop a corpus."""
    ds = _dataset(_build(tmp_path), weights=None)
    ds._build_weighted_index({"Big": 0.0, "Small": 1.0})
    slots = np.asarray(ds.sampler_slot_counts, dtype=np.int64)
    di = ds._ds_names.index("Big")
    assert int(slots[ds._ep_ds == di].sum()) == 1


# ── virtual -> real mapping ──────────────────────────────────────────────────

def test_every_virtual_index_maps_to_a_valid_window_of_the_right_episode(tmp_path):
    ds = _dataset(_build(tmp_path), weights=None)
    ds._build_weighted_index(_alpha_weights(ds))
    counts = np.asarray(ds.episode_lengths, dtype=np.int64)
    for v in range(len(ds)):
        real = ds._map_virtual_index(v)
        assert 0 <= real < int(ds._cum_lengths[-1])
        e, ordinal, _ = ds._resolve_index(real)
        assert 0 <= ordinal < int(counts[e]), (v, real, e, ordinal, counts[e])


def test_distinct_window_coverage_matches_the_legacy_layout(tmp_path):
    """The relayout must not change how much of the corpus an epoch touches.

    On the real corpus both layouts visit the same number of distinct windows per epoch; the
    check here is the same identity on the fixture, per dataset, for a downsampled and an
    upsampled source.
    """
    ds = _dataset(_build(tmp_path), weights=None)
    weights = _alpha_weights(ds)
    legacy, sizes, _ = _legacy_targets(ds, weights)
    ds._build_weighted_index(weights)
    slots = np.asarray(ds.sampler_slot_counts, dtype=np.int64)
    counts = np.asarray(ds.episode_lengths, dtype=np.int64)
    for di, name in enumerate(ds._ds_names):
        m = ds._ep_ds == di
        got = int(np.minimum(counts[m], slots[m]).sum())
        assert got == min(int(sizes[di]), legacy[name]), (name, got)


def test_downsampled_episodes_keep_their_windows_spread_not_front_loaded(tmp_path):
    """A downsampled episode must contribute windows from across its own duration.

    Taking the first c_e windows instead of striding would silently bias every downsampled
    corpus toward episode beginnings.
    """
    ds = _dataset(_build(tmp_path), weights=None)
    ds._build_weighted_index(_alpha_weights(ds))
    slots = np.asarray(ds.sampler_slot_counts, dtype=np.int64)
    counts = np.asarray(ds.episode_lengths, dtype=np.int64)
    down = np.nonzero(counts > slots)[0]
    assert down.size, "fixture must downsample at least one episode"
    vbase = np.concatenate([[0], np.cumsum(slots)[:-1]])
    for e in down.tolist():
        c, s = int(counts[e]), int(slots[e])
        ordinals = [ds._resolve_index(ds._map_virtual_index(int(vbase[e]) + k))[1]
                    for k in range(s)]
        assert ordinals == [(k * c) // s for k in range(s)], (e, ordinals)
        assert len(set(ordinals)) == s
        # The stride reaches to within one step of the episode's last window; taking the first
        # s windows instead would bias every downsampled corpus toward episode beginnings.
        assert max(ordinals) >= c - math.ceil(c / s), (e, ordinals, c, s)


# ── locality actually comes back ─────────────────────────────────────────────

def test_sampler_builds_a_locality_permutation_over_the_weighted_index(tmp_path):
    ds = _dataset(_build(tmp_path), weights=None)
    ds._build_weighted_index(_alpha_weights(ds))
    sampler = DistributedBatchSampler(
        ds, sequence_lengths=None, num_replicas=1, rank=0, micro_batch_size=1,
        gradient_accumulation_steps=1, shuffle=True, seed=0, drop_last=False, num_workers=1)
    counts = sampler._episode_sample_counts()
    assert counts is not None, "sampler refused the weighted index -> randperm -> no locality"
    perm = sampler._locality_permutation(0)
    assert perm is not None
    assert sorted(perm.tolist()) == list(range(len(ds))), "not a permutation"


def test_adjacent_virtual_slots_usually_share_an_episode(tmp_path):
    """The property that makes the latent LRU work, measured on the fixture."""
    ds = _dataset(_build(tmp_path), weights=None)
    ds._build_weighted_index(_alpha_weights(ds))
    eps = [ds._resolve_index(ds._map_virtual_index(v))[0] for v in range(len(ds))]
    same = sum(1 for a, b in zip(eps, eps[1:]) if a == b) / (len(eps) - 1)
    ceiling = 1.0 - len(set(eps)) / len(eps)
    assert same >= 0.8 * ceiling, (same, ceiling)


class _TwoSpaceDataset:
    """The sampler must read sampler_slot_counts, not episode_lengths, when both exist."""

    def __init__(self, virtual, real):
        self._virtual = np.asarray(virtual, dtype=np.int64)
        self._real = np.asarray(real, dtype=np.int64)

    def __len__(self):
        return int(self._virtual.sum())

    @property
    def sampler_slot_counts(self):
        return self._virtual

    @property
    def episode_lengths(self):
        # Deliberately does NOT tile len(): this is the dataset-major shape that used to make
        # the sampler fall back to randperm.
        return self._real


def test_sampler_prefers_sampler_slot_counts_over_episode_lengths():
    ds = _TwoSpaceDataset(virtual=[4, 4, 4], real=[7, 2, 3])
    sampler = DistributedBatchSampler(
        ds, sequence_lengths=None, num_replicas=1, rank=0, micro_batch_size=1,
        gradient_accumulation_steps=1, shuffle=True, seed=0, drop_last=False, num_workers=1)
    counts = sampler._episode_sample_counts()
    assert counts is not None
    assert counts.tolist() == [4, 4, 4]


def test_sampler_falls_back_to_episode_lengths_without_the_hook():
    class _Legacy:
        def __len__(self):
            return 12

        @property
        def episode_lengths(self):
            return [4, 4, 4]

    sampler = DistributedBatchSampler(
        _Legacy(), sequence_lengths=None, num_replicas=1, rank=0, micro_batch_size=1,
        gradient_accumulation_steps=1, shuffle=True, seed=0, drop_last=False, num_workers=1)
    assert sampler._episode_sample_counts().tolist() == [4, 4, 4]


# ── the unweighted path is untouched ─────────────────────────────────────────

def test_unweighted_dataset_is_byte_identical_in_behaviour(tmp_path):
    """The 4B natural arm was mid-run when this changed; it must not be affected.

    dataset_weights is None -> _build_weighted_index never runs -> _map_virtual_index is the
    identity, len() comes from the base class, and sampler_slot_counts is episode_lengths.
    """
    ds = _dataset(_build(tmp_path), weights=None)
    assert ds._weight_slots is None and ds._weight_vbase is None and ds._weight_total is None
    assert len(ds) == NATURAL_TOTAL
    assert ds.sampler_slot_counts == ds.episode_lengths
    for v in range(len(ds)):
        assert ds._map_virtual_index(v) == v
