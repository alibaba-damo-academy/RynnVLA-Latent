"""Regression tests for the sampler's locality-preserving shuffle and the latent cache width.

Both exist because a uniform `randperm` over an episode-contiguous index defeats the latent
LRU completely: measured on the real corpus the hit rate was 0.0%, so every sample re-read a
whole npz to take 6 of its rows, amplifying IO ~120x over the bytes the corpus occupies). The
sampler now emits a round-robin over windows of W episodes, bucketed by episode sample count.

The window is derived, not chosen, and the two constraints it must satisfy are asserted at
construction. A W that merely happens to divide today's worker stride degrades to near-zero
reuse the day num_workers, micro_batch_size or the rank count moves, and it does so silently --
the epoch still trains, just slowly. These tests pin the derivation and the failure modes.
"""
import numpy as np
import pytest

try:
    from rynnvla.datasets.vla_datasets.latent_loader import LatentSlotLoader
    from rynnvla.training.sampler import DistributedBatchSampler
except ImportError:  # pragma: no cover - dev pods without the training extras
    # rynnvla.training.__init__ pulls in trainer -> deepspeed, and rynnvla.arguments pulls in
    # a transformers new enough to still export PreTrainedConfig. Neither is needed to test
    # the sampler, which depends only on math/typing/numpy/torch, so load the two modules by
    # path. Without this the file cannot be collected on a dev pod at all, and a test nobody
    # has ever seen run is not a test.
    import importlib.util
    import os

    def _load(name, rel):
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), rel)
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    DistributedBatchSampler = _load("_sampler_under_test", "rynnvla/training/sampler.py").DistributedBatchSampler
    LatentSlotLoader = _load("_latent_loader_under_test",
                             "rynnvla/datasets/vla_datasets/latent_loader.py").LatentSlotLoader

NRANKS, MICRO, GA, NWORKERS, LRU = 64, 8, 1, 12, 64
STRIDE = MICRO * GA * NWORKERS * NRANKS          # 6144: one worker's batch-to-batch distance


class _StubDataset:
    """Duck-types the only two things the sampler reads."""

    def __init__(self, counts):
        self._counts = np.asarray(counts, dtype=np.int64)

    def __len__(self):
        return int(self._counts.sum())

    @property
    def episode_lengths(self):
        # The real dataset returns a list (latent_pretrain.py); the sampler wraps it in
        # np.asarray either way. Handing back the array keeps a 2M-episode fixture cheap.
        return self._counts


class _OpaqueDataset:
    """A dataset with no episode_lengths: the sampler must fall back, not crash."""

    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n


def _sampler(counts, seed=0, shuffle=True, num_workers=NWORKERS, **kw):
    ds = counts if isinstance(counts, _OpaqueDataset) else _StubDataset(counts)
    return DistributedBatchSampler(
        ds, sequence_lengths=None, num_replicas=NRANKS, rank=0, micro_batch_size=MICRO,
        gradient_accumulation_steps=GA, shuffle=shuffle, seed=seed, drop_last=False,
        num_workers=num_workers, **kw)


def _unstride(per_rank):
    """Invert the sampler's `indices[r::num_replicas]` split back into epoch order."""
    out = np.empty(len(per_rank[0]) * NRANKS, dtype=np.int64)
    for r in range(NRANKS):
        out[r::NRANKS] = per_rank[r]
    return out


@pytest.fixture(scope="module")
def mixed_counts():
    """A production-SHAPED corpus at 1/6 scale: median S_e 4, mean ~11, tail to 1900.

    Scale is not a free parameter here, and getting it wrong silently inverts the verdict.
    The locality window is W = 6144 episodes, cut from a list bucketed by S_e, so the number
    of length buckets is ~episodes/W and bucket purity -- hence whether a window stays dense
    -- improves with corpus size. Measured on random subsamples of the real index:

        episodes    buckets   periodicity ep[p]==ep[p+W]   min distinct eps/batch
         250,000       514            0.566                        3 / 8
        2,000,000      953            0.782                        8 / 8
        6,000,000    1,169            0.824                        8 / 8
        13,049,971   ~2,125          (production, ~0.87)            8 / 8

    A 250k fixture therefore FAILS properties that hold in production, and a bimodal fixture
    (a few hundred very long episodes among tens of thousands of short ones) fails them harder
    still -- both were mistaken for real defects during development. The ideal ceiling is
    1 - 1/mean(S_e) = 0.89. Assertions below are floored at the 2M measurement, i.e.
    deliberately conservative relative to production.
    """
    rng = np.random.RandomState(0)
    n = 2_000_000
    u = rng.random_sample(n)
    counts = np.empty(n, dtype=np.int64)
    counts[u < 0.90] = rng.randint(1, 9, size=int((u < 0.90).sum()))
    m1 = (u >= 0.90) & (u < 0.98)
    counts[m1] = rng.randint(9, 41, size=int(m1.sum()))
    m2 = (u >= 0.98) & (u < 0.997)
    counts[m2] = rng.randint(41, 251, size=int(m2.sum()))
    counts[u >= 0.997] = rng.randint(251, 1901, size=int((u >= 0.997).sum()))
    rng.shuffle(counts)
    return counts


def test_the_window_is_derived_from_the_worker_stride(mixed_counts):
    assert _sampler(mixed_counts)._locality_window() == STRIDE == 6144


def test_the_window_tracks_num_workers_instead_of_being_hardcoded(mixed_counts):
    # The whole point of deriving it: change num_workers and the window must follow, otherwise
    # it stops dividing the stride and reuse collapses to ~0 without any error.
    assert _sampler(mixed_counts, num_workers=8)._locality_window() == MICRO * GA * 8 * NRANKS
    assert _sampler(mixed_counts, num_workers=16)._locality_window() == MICRO * GA * 16 * NRANKS


def test_a_window_that_cannot_divide_the_stride_is_refused(mixed_counts):
    with pytest.raises(ValueError, match="must divide the worker batch stride"):
        _sampler(mixed_counts, locality_window=1000)._locality_window()


def test_a_window_small_enough_to_collide_inside_a_batch_is_refused(mixed_counts):
    # W=64 makes 64*k mod W identical for every k, so all 8 samples of a micro-batch would come
    # from one episode -- the gradient correlation the round-robin exists to prevent.
    with pytest.raises(ValueError, match="several samples of one episode"):
        _sampler(mixed_counts, locality_window=64)._locality_window()


def test_the_emitted_epoch_is_a_true_permutation(mixed_counts):
    n = len(_StubDataset(mixed_counts))
    perm = _sampler(mixed_counts)._locality_permutation(0)
    assert perm is not None
    assert len(perm) == n
    assert np.array_equal(np.sort(perm), np.arange(n))


def test_every_micro_batch_draws_distinct_episodes(mixed_counts):
    """Batch diversity: the reason for round-robin over a window instead of episode-contiguous.

    Walks EVERY batch. An earlier version stepped every 7th batch of a small synthetic corpus
    and reported a clean 8/8; on the real S_e distribution (median 4) the true minimum is 4/8,
    because a block whose episodes have unequal lengths goes ragged, so a batch can straddle
    two rounds and meet one episode twice. Measured on a 260k-episode unbiased subsample of
    the real index: mean 7.99, 99.6% of batches full. The floor is that measurement.
    """
    perm = _sampler(mixed_counts)._locality_permutation(0)
    bounds = np.concatenate([[0], np.cumsum(mixed_counts)])
    nb = (len(perm) - NRANKS) // (MICRO * NRANKS)
    distinct = np.array([
        len(set(np.searchsorted(bounds, perm[j * MICRO * NRANKS + np.arange(MICRO) * NRANKS],
                                side="right").tolist()))
        for j in range(nb)
    ])
    assert distinct.mean() > 7.9, f"mean distinct episodes per batch fell to {distinct.mean():.2f}"
    # 0.98, not 1.0: measured 98.7% on the bimodal fixture below and 99.6% on a 260k-episode
    # unbiased subsample of the real index. Blocks that straddle two episode lengths go ragged
    # and a batch can then meet one episode twice.
    assert (distinct == MICRO).mean() > 0.98, f"only {100*(distinct == MICRO).mean():.1f}% of batches are full"


def test_a_worker_revisits_the_same_episodes(mixed_counts):
    """The property the latent LRU actually depends on -- batch diversity is not enough.

    Reuse needs worker w's batch j and batch j+num_workers to draw from the SAME episodes,
    because the LRU is per-worker. randperm scores 0.000 here, which is the entire problem
    this module exists to fix. Floors are the measured 2M-episode values (mean 0.800,
    median 1.000); production sits higher -- see the scale table on the fixture.
    """
    perm = _sampler(mixed_counts)._locality_permutation(0)
    bounds = np.concatenate([[0], np.cumsum(mixed_counts)])
    nb = (len(perm) - NRANKS) // (MICRO * NRANKS)
    if nb <= 2 * NWORKERS:
        pytest.skip("corpus too small for a worker chain")

    def batch(j):
        pos = j * MICRO * NRANKS + np.arange(MICRO) * NRANKS
        return set(np.searchsorted(bounds, perm[pos], side="right").tolist())

    overlaps = [
        len(a & batch(j + NWORKERS)) / len(a)
        for w in range(NWORKERS)
        for j in range(w, nb - NWORKERS, NWORKERS)
        for a in [batch(j)]
    ]
    assert np.median(overlaps) > 0.9, f"median same-worker overlap {np.median(overlaps):.3f}"
    # 0.6, not the 0.705 measured here: the fixture spreads ~1900 distinct lengths over far
    # fewer episodes per bucket than a production corpus does, so its windows are less pure and
    # its floor sits below the 0.82 measured on a real subsample. The discriminating power is
    # the contrast with randperm's 0.000, not the exact floor.
    assert np.mean(overlaps) > 0.6, f"mean same-worker overlap {np.mean(overlaps):.3f}"


def test_long_episode_reuse_actually_beats_randperm(mixed_counts):
    """The reason this code exists. Simulates the per-worker LRU over both orders."""
    perm = _sampler(mixed_counts)._locality_permutation(0)
    bounds = np.concatenate([[0], np.cumsum(mixed_counts)])
    rng = np.random.RandomState(1)
    random_order = rng.permutation(len(_StubDataset(mixed_counts)))

    def hit_rate(order, rounds=400):
        caches = [{"order": [], "set": set()} for _ in range(NWORKERS)]
        hits = total = 0
        for j in range(rounds):
            c = caches[j % NWORKERS]
            for k in range(MICRO):
                p = j * MICRO * NRANKS + k * NRANKS
                if p >= len(order):
                    return hits / max(total, 1)
                e = int(np.searchsorted(bounds, int(order[p]), side="right"))
                if e in c["set"]:
                    hits += 1
                    c["order"].remove(e)
                else:
                    c["set"].add(e)
                    if len(c["order"]) >= LRU:
                        c["set"].discard(c["order"].pop(0))
                c["order"].append(e)
                total += 1
        return hits / max(total, 1)

    locality, uniform = hit_rate(perm), hit_rate(random_order)
    assert uniform < 0.05, f"baseline should miss almost everything, got {uniform:.3f}"
    # Floor is well under the measured value for the same reason as the overlap test above:
    # window purity scales with corpus size. What this asserts is the contrast -- randperm is
    # ~0.00 and locality must be an order of magnitude above it.
    assert locality > 0.4, f"locality order should reuse heavily, got {locality:.3f}"
    assert locality > 8 * uniform


def test_shuffle_false_is_the_identity_order(mixed_counts):
    n = len(_StubDataset(mixed_counts))
    per_rank = _sampler(mixed_counts, shuffle=False)._get_global_batch_indices()
    assert np.array_equal(_unstride(per_rank)[:n], np.arange(n))


def test_same_seed_and_epoch_reproduce_and_a_new_epoch_differs(mixed_counts):
    a = _sampler(mixed_counts, seed=5)._locality_permutation(5)
    b = _sampler(mixed_counts, seed=5)._locality_permutation(5)
    c = _sampler(mixed_counts, seed=5)._locality_permutation(6)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_ranks_are_disjoint_and_cover_the_epoch(mixed_counts):
    per_rank = _sampler(mixed_counts)._get_global_batch_indices()
    assert len(per_rank) == NRANKS
    assert all(len(p) == len(per_rank[0]) for p in per_rank)
    # The split is indices[r::num_replicas], so walking the ranks round-robin must rebuild the
    # padded epoch exactly -- no dropped, duplicated or reordered sample.
    rebuilt = np.empty(len(per_rank[0]) * NRANKS, dtype=np.int64)
    for i in range(len(per_rank[0])):
        for r in range(NRANKS):
            rebuilt[i * NRANKS + r] = per_rank[r][i]
    n = len(_StubDataset(mixed_counts))
    perm = _sampler(mixed_counts)._locality_permutation(0)
    padded = np.concatenate([perm, perm[: len(rebuilt) - n]]) if len(rebuilt) > n else perm[: len(rebuilt)]
    assert np.array_equal(rebuilt, padded)


def test_a_dataset_without_episode_lengths_falls_back_to_randperm():
    s = _sampler(_OpaqueDataset(64 * STRIDE), shuffle=True)
    assert s._locality_permutation(0) is None
    per_rank = s._get_global_batch_indices()          # must not raise
    assert len(per_rank) == NRANKS


def test_too_few_episodes_for_one_window_falls_back():
    counts = np.full(100, 50)                          # 100 episodes < W=6144
    s = _sampler(counts)
    assert s._locality_permutation(0) is None


def test_production_scale_periodicity_meets_the_ceiling():
    """The real acceptance bar, on the real episode-length distribution.

    Deliberately separate from the fast suite and skipped without the index, because the bar is
    scale-dependent and cannot be asserted on a small fixture. Periodicity tracks episodes per
    distinct length, not episode count:

        corpus                   episodes/distinct S_e   periodicity
        small synthetic (above)           ~1,050             0.635
        real index, subsample             ~2,100             0.782
        real index, full corpus           ~3x that           0.8725   <- production

    So 0.85 is a production-scale number; asserting it on the small fixture would fail on a correct
    implementation. The ceiling 1 - episodes/samples is a bound for ANY ordering, since every
    (episode, view) npz must be read at least once.

    Needs a production-scale index npz, which is not bundled; set RYNNVLA_LATENT_INDEX to point
    at one built by scripts/build_latent_pretrain_index.py over your own corpus.
    """
    import os

    path = os.environ.get("RYNNVLA_LATENT_INDEX")
    if not path or not os.path.exists(path):
        pytest.skip(f"set RYNNVLA_LATENT_INDEX to a production-scale latent index (got {path!r})")

    z = np.load(path)
    nlat = z["ep_nlat"].astype(np.int64)
    stride = z["ep_stride"].astype(np.int64)
    chunk = 6
    counts = np.maximum(((nlat - ((chunk - 1) * stride + 1)) // np.maximum(1, (stride * chunk) // 2)) + 1, 0)

    sampler = _sampler(counts, seed=7)
    perm = sampler._locality_permutation(7)
    window = sampler._locality_window()

    # A wrong S_e formula here would silently invalidate every number above, so pin it to the
    # sample count the index itself reports.
    assert len(perm) == int(counts.sum())
    bounds = np.concatenate([[0], np.cumsum(counts)])
    pos = np.arange(0, len(perm) - window, max(1, len(perm) // 2_000_000))
    ep = np.searchsorted(bounds, perm[np.concatenate([pos, pos + window])], side="right")
    k = pos.size
    periodicity = float((ep[:k] == ep[k:]).mean())
    ceiling = 1.0 - len(counts) / len(perm)

    assert np.array_equal(np.sort(perm), np.arange(len(perm))), "not a true permutation"
    assert periodicity >= 0.85, f"periodicity {periodicity:.4f} below the 0.85 production bar"
    assert periodicity >= 0.95 * ceiling, f"periodicity {periodicity:.4f} under 95% of ceiling {ceiling:.4f}"


def test_a_ragged_corpus_warns_instead_of_silently_losing_reuse(caplog):
    """Pin the startup self-check.

    A bimodal length distribution makes windows ragged and drops reuse to roughly randperm's,
    with nothing else observably wrong -- the symptom is just "training is slow". Production
    measures 0.869 (97% of its 0.892 ceiling) and stays quiet; this fixture measures 0.561
    (67%) and must warn. Without this test a refactor could drop the guard and the next corpus
    change would cost days of GPU time before anyone connected it to the sampler.
    """
    import logging

    rng = np.random.RandomState(0)
    counts = np.concatenate([rng.randint(1, 9, size=2_000_000), rng.randint(400, 1900, size=3000)])
    rng.shuffle(counts)

    with caplog.at_level(logging.WARNING):
        perm = _sampler(counts)._locality_permutation(0)
    assert perm is not None
    assert any("locality periodicity" in r.getMessage() for r in caplog.records), (
        "a ragged corpus must warn; messages were " + repr([r.getMessage()[:60] for r in caplog.records])
    )


def test_latent_cache_keeps_the_on_disk_width_and_output_is_unchanged(tmp_path):
    """Widening after the slice must be bit-identical to widening before it."""
    dim, T = 8, 40
    rng = np.random.RandomState(3)
    arr = rng.randn(T, dim).astype(np.float16)
    path = tmp_path / "latent.npz"
    np.savez(path, latent_action=arr)

    loader = LatentSlotLoader(latent_action_dim=dim)
    got, mask = loader.load_slots({"cam": str(path)}, start_frame=5, chunk=6, stride=2)

    expected = ((arr.astype(np.float32))[np.clip(5 + np.arange(6) * 2, 0, T - 1)]
                - loader.mean) / loader.std
    import torch
    assert torch.equal(got[0], torch.from_numpy(expected))
    assert bool(mask.all())

    cached = loader._cache[str(path)]
    assert cached.dtype == np.float16, "the cache must not hold a widened copy"
    assert cached.nbytes == arr.nbytes


def test_the_dim_check_still_fires_on_a_width_mismatch(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez(path, latent_action=np.zeros((10, 7), dtype=np.float16))
    loader = LatentSlotLoader(latent_action_dim=8)
    with pytest.raises(ValueError, match="latent_action has shape"):
        loader.load_slots({"cam": str(path)}, start_frame=0, chunk=2)
