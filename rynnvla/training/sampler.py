import math
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DistributedSampler

# Absolute, unlike the relative imports in sibling modules: this file depends only on
# math/typing/numpy/torch, and keeping it that way lets tests load it by path on a dev pod
# where rynnvla.training.__init__ cannot be imported (it pulls in deepspeed).
from rynnvla.utils.logging import get_logger

logger = get_logger(__name__)


class DistributedBatchSampler(DistributedSampler):
    def __init__(
        self,
        dataset: Dataset,
        sequence_lengths: List[int],
        num_replicas: int,
        rank: int,
        micro_batch_size: int,
        gradient_accumulation_steps: int,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        decoder_load_balancing: bool = False,
        dynamic_batching: bool = False,
        dynamic_batching_window_size: int = 128,
        model_max_length: int = 16384,
        num_workers: int = 0,
        locality_window: Optional[int] = None,
        shuffle_mode: str = "auto",
    ):
        self.dataset = dataset
        self.sequence_lengths = sequence_lengths
        self.num_replicas = num_replicas
        self.rank = rank
        self.micro_batch_size = micro_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.decoder_load_balancing = decoder_load_balancing
        self.dynamic_batching = dynamic_batching
        self.num_workers = max(1, int(num_workers))
        self.locality_window = locality_window
        if shuffle_mode not in ("auto", "global"):
            raise ValueError(f"Unknown sampler shuffle mode: {shuffle_mode!r}")
        # Keep auto for existing checkpoints: resuming must reproduce their order.
        # Direct-action recipes can explicitly request the historical global shuffle;
        # episode_lengths alone does not imply that temporal locality is appropriate.
        self.shuffle_mode = shuffle_mode

        self.epoch = 0
        self.num_skipped_batches = 0

        assert not (decoder_load_balancing or dynamic_batching) or sequence_lengths is not None

        if self.dynamic_batching:
            raise NotImplementedError

        else:
            local_batch_size = micro_batch_size * gradient_accumulation_steps
            global_batch_size = local_batch_size * self.num_replicas

            # If the dataset length is evenly divisible by # of replicas, then there
            # is no need to drop any data, since the dataset will be split equally.
            if self.drop_last and len(self.dataset) % global_batch_size != 0:  # type: ignore[arg-type]
                # Split to nearest available length that is evenly divisible.
                # This is to ensure each rank receives the same amount of data when
                # using this Sampler.
                self.num_batches = math.ceil(
                    (len(self.dataset) - global_batch_size) / global_batch_size  # type: ignore[arg-type]
                )
            else:
                self.num_batches = math.ceil(len(self.dataset) / global_batch_size)  # type: ignore[arg-type]

            # A dataset smaller than one global batch makes the ceil above go negative and
            # land on 0. That is not a harmless empty epoch: indices[:0] still satisfies the
            # len(indices) == total_size assert in _get_global_batch_indices, so the loader
            # would iterate zero batches and training would silently run zero steps.
            if self.num_batches <= 0:
                raise ValueError(
                    f"Dataset of {len(self.dataset)} samples cannot fill one global batch of "  # type: ignore[arg-type]
                    f"{global_batch_size} (micro_batch_size={micro_batch_size} x "
                    f"gradient_accumulation_steps={gradient_accumulation_steps} x "
                    f"num_replicas={self.num_replicas}). Add data, lower the batch, or set "
                    "dataloader_drop_last=False."
                )

    # ── locality-preserving shuffle ────────────────────────────────────────────
    # randperm(len(dataset)) is uniform, and that is exactly the problem. The latent LRU
    # (latent_loader.py, cache_size=64, keyed by per-view npz path) can only amortise a read
    # if the SAME file returns to the SAME worker before 64 other files evict it. Index space
    # is episode-contiguous, so a uniform permutation puts two samples of one episode millions
    # of positions apart: measured hit rate 0.0%, and every sample re-reads a whole npz to
    # take 6 of its rows. Over one epoch that amplifies IO ~120x above the bytes the corpus
    # actually occupies, because long-episode corpora -- a small share of episodes but a large
    # share of samples, in npz files of ~190 MiB -- are re-read once per sample instead of once
    # per episode.
    #
    # This keeps the epoch a true permutation but emits it as a round-robin over windows of
    # W episodes: position p inside a window belongs to episode (p mod W) at round (p // W).
    # Two constraints follow from how this sampler is consumed, and both are asserted rather
    # than left to chance:
    #
    #   W | (local_batch_size * num_workers * num_replicas)
    #       A worker owns batches w, w+num_workers, ..., so its consecutive batches are
    #       exactly that many positions apart. W dividing it gives episode(p) ==
    #       episode(p + stride): the worker revisits the same W-offsets and the LRU hits.
    #       W = stride satisfies this by construction, and stays correct if num_workers,
    #       micro_batch_size or the rank count changes -- which is why it is derived, not
    #       hardcoded. A hardcoded W that merely happens to divide today's stride silently
    #       degrades to near-zero reuse the day one of them moves.
    #   W > num_replicas * (local_batch_size - 1)
    #       One micro-batch spans positions num_replicas*k apart for k = 0..lbs-1. At or
    #       below this bound those collide mod W and a batch draws several samples from ONE
    #       episode -- the gradient correlation this exists to avoid.
    #
    # Windows are cut from a list bucketed by per-episode sample count. That is not cosmetic:
    # a window mixing 4-sample episodes with a 438-sample one goes ragged after round 4, so
    # position p stops identifying episode p mod W and measured hit rate collapses from 91%
    # to 6%. Bucketing keeps the round-robin dense. It costs in-batch source diversity,
    # measured on a full-corpus index against a uniform randperm of the same samples
    # (uniform matches the analytic baseline sum_i(1-(1-p_i)^k) to 0.2%). Quoted as ratios so
    # the trade-off transfers to a corpus of any size:
    #   micro-batch, 8 samples at stride num_replicas: distinct corpora per batch fall to ~65%
    #     of uniform, and ~5% of micro-batches come from a single corpus (0% uniform).
    #   global batch, 512 consecutive positions: distinct corpora per step fall to ~59% of
    #     uniform, and the largest single corpus takes ~2.4x the share of the step it would
    #     under uniform sampling.
    # So a step is ~40% less source-diverse and is often dominated by one corpus. That is a
    # variance increase, not a bias: block order is reshuffled per epoch, so which corpus
    # dominates which step is random over the run. It is bought deliberately -- the uniform
    # order costs >100x read amplification through the per-view latent LRU.
    def _episode_sample_counts(self) -> Optional[np.ndarray]:
        # sampler_slot_counts is the space THIS sampler permutes. It differs from
        # episode_lengths only for a dataset whose __len__ enumerates a virtual index that is
        # a reweighting of a real one (LatentPretrainDataset under dataset_weights): there
        # episode_lengths must keep describing the real space because _cum_lengths and
        # _resolve_index are built from it. Datasets with a single index space do not define
        # the hook and fall through to episode_lengths, unchanged.
        lengths = getattr(self.dataset, "sampler_slot_counts", None)
        if lengths is None:
            lengths = getattr(self.dataset, "episode_lengths", None)
        if lengths is None:
            return None
        counts = np.asarray(lengths, dtype=np.int64)
        # A wrapper dataset whose counts do not tile its own index space would silently produce
        # a permutation that is not a permutation. Refuse and fall back.
        #
        # This guard is load-bearing, not defensive: it is what caught the weighted index
        # laying its virtual space out dataset-major, where per-episode counts cannot tile it.
        # The fallback is silent by design and lands as "training is slow" -- see
        # _log_locality_quality -- so a dataset that expects locality and never logs the
        # locality line is failing this check.
        if counts.ndim != 1 or counts.size == 0 or (counts < 0).any():
            return None
        if int(counts.sum()) != len(self.dataset):  # type: ignore[arg-type]
            return None
        return counts

    def _locality_window(self) -> int:
        local_batch_size = self.micro_batch_size * self.gradient_accumulation_steps
        stride = local_batch_size * self.num_workers * self.num_replicas
        window = stride if self.locality_window is None else int(self.locality_window)
        minimum = self.num_replicas * (local_batch_size - 1) + 1
        if window < minimum:
            raise ValueError(
                f"locality_window={window} would put several samples of one episode in a single "
                f"micro-batch; it must exceed num_replicas * (local_batch_size - 1) = "
                f"{self.num_replicas} * {local_batch_size - 1} = {minimum - 1}."
            )
        if stride % window != 0:
            raise ValueError(
                f"locality_window={window} must divide the worker batch stride "
                f"local_batch_size * num_workers * num_replicas = {local_batch_size} * "
                f"{self.num_workers} * {self.num_replicas} = {stride}, or a worker never "
                "revisits the same episodes and the latent LRU gets no reuse."
            )
        return window

    def _locality_permutation(self, seed: int) -> Optional[np.ndarray]:
        counts = self._episode_sample_counts()
        if counts is None:
            return None
        window = self._locality_window()
        if len(counts) < window:
            return None

        rng = np.random.RandomState(seed & 0x7FFFFFFF)
        # Bucket by episode length (random within a bucket), then shuffle the bucket order so
        # the epoch still visits short- and long-episode corpora in a random sequence.
        order = np.lexsort((rng.permutation(len(counts)), counts))
        blocks = rng.permutation((len(counts) + window - 1) // window)
        base = np.zeros(len(counts), dtype=np.int64)
        np.cumsum(counts[:-1], out=base[1:])

        out = np.empty(int(counts.sum()), dtype=np.int64)
        filled = 0
        for block in blocks:
            win = order[block * window:(block + 1) * window]
            lengths = counts[win]
            starts = base[win]
            for r in range(int(lengths.max())):
                live = lengths > r
                if not live.any():
                    break
                emit = starts[live] + r
                out[filled:filled + emit.size] = emit
                filled += emit.size
        perm = out[:filled]
        self._log_locality_quality(perm, counts, window)
        return perm

    def _log_locality_quality(self, perm: np.ndarray, counts: np.ndarray, window: int) -> None:
        """Report how much reuse the permutation actually bought, once per epoch.

        The round-robin only stays dense while a window's episodes have similar sample counts.
        On a real corpus that holds -- episodes per distinct length exceeds the window size, so
        a window is nearly always one length and measured periodicity is 0.872 against a 0.892
        ceiling. A future corpus whose
        lengths are bimodal (a few very long episodes among many short ones) makes windows
        ragged: position p stops identifying episode p mod W, and reuse collapses to randperm's
        ~1x with nothing else visibly wrong. The failure is silent and lands as "training is
        slow", which is exactly the misattribution this line exists to prevent.
        """
        n = len(perm)
        if n <= 2 * window:
            return
        bounds = np.concatenate([[0], np.cumsum(counts)])
        sample = np.arange(0, n - window, max(1, n // 1_000_000))
        episode = np.searchsorted(bounds, perm[np.concatenate([sample, sample + window])], side="right")
        k = sample.size
        periodicity = float((episode[:k] == episode[k:]).mean())
        # Hard ceiling for ANY order: each (episode, view) file must be read at least once, so
        # the miss fraction cannot go below episodes/samples.
        ceiling = 1.0 - len(counts) / n
        logger.info(
            f"DistributedBatchSampler locality: window={window} periodicity={periodicity:.3f} "
            f"= {100 * periodicity / max(ceiling, 1e-9):.0f}% of the {ceiling:.3f} ceiling "
            f"({n:,} samples, {len(counts):,} episodes)"
        )
        if periodicity < 0.8 * ceiling:
            logger.warning(
                f"DistributedBatchSampler: locality periodicity {periodicity:.3f} is under 80% of its "
                f"{ceiling:.3f} ceiling. The latent LRU will get little reuse, so nearly every sample "
                "re-reads a whole npz to take 6 of its rows and the run stays I/O bound. Check the "
                "episode-length distribution before blaming throughput on anything else."
            )

    def _get_global_batch_indices(self):
        local_batch_size = self.micro_batch_size * self.gradient_accumulation_steps
        global_batch_size = local_batch_size * self.num_replicas

        indices = None
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            if self.shuffle_mode == "auto":
                indices = self._locality_permutation(self.seed + self.epoch)
            if indices is None:
                g = torch.Generator()
                g.manual_seed(self.seed + self.epoch)
                indices = torch.randperm(len(self.dataset), generator=g).numpy()  # type: ignore[arg-type]
                logger.info(f"DistributedBatchSampler global shuffle: {len(indices):,} samples, "
                            f"seed={self.seed + self.epoch}, mode={self.shuffle_mode}")
        else:
            indices = np.arange(len(self.dataset), dtype=np.int64)  # type: ignore[arg-size]

        total_size = self.num_batches * global_batch_size
        if not self.drop_last:
            # Add extra samples to make it evenly divisible. np.resize tiles `indices` and
            # truncates to exactly total_size, which reproduces the previous one-pass-then-
            # prefix order bit for bit whenever the old expression was correct -- so existing
            # checkpoints resume on the same order -- and stays correct when it was not:
            # `indices[:padding_size]` silently clamps to len(indices) once a global batch is
            # more than twice the dataset, overshooting total_size and tripping the assert
            # below. That needs a dataset smaller than ~half a global batch, so it is
            # unreachable on a production corpus but immediate on the bundled sample data run
            # with a real recipe's batch geometry.
            if total_size > len(indices):
                indices = np.resize(indices, total_size)
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:total_size]
        assert len(indices) == total_size

        per_rank = [indices[i :: self.num_replicas] for i in range(self.num_replicas)]

        assert len(per_rank) == self.num_replicas
        assert all(len(local) == self.num_batches * local_batch_size for local in per_rank)

        return per_rank

    def _longest_first_partition(self, data_indices: List[int]):
        partitions = [[] for _ in range(self.num_replicas)]
        batch_seqlens = [0 for _ in range(self.num_replicas)]

        seqlen_list = [self.sequence_lengths[i] for i in data_indices]
        sorted_seqlen_list = sorted(
            [(seqlen, i) for i, seqlen in enumerate(seqlen_list)],
            key=lambda x: x[0],
            reverse=True,
        )

        for i, (seqlen, idx) in enumerate(sorted_seqlen_list):
            if i < self.num_replicas:
                partition_id = i
            else:
                partition_id = min(list(range(self.num_replicas)), key=lambda x: batch_seqlens[x])
            partitions[partition_id].append(idx)
            batch_seqlens[partition_id] += seqlen

        new_data_indices = [[data_indices[i] for i in batch] for batch in partitions]

        return new_data_indices

    def __iter__(self):
        per_rank = self._get_global_batch_indices()
        local_batch_size = self.micro_batch_size * self.gradient_accumulation_steps
        for i in range(0, self.num_batches):
            for j in range(self.gradient_accumulation_steps):
                if i * self.gradient_accumulation_steps + j < self.num_skipped_batches:
                    continue
                offset = i * local_batch_size + j * self.micro_batch_size
                if self.decoder_load_balancing:
                    all_sample_indices = sum(
                        [
                            per_rank[k][offset : offset + self.micro_batch_size].tolist()
                            for k in range(self.num_replicas)
                        ],
                        [],
                    )
                    batch_indices = self._longest_first_partition(all_sample_indices)
                    yield batch_indices[self.rank]
                else:
                    # .tolist() here rather than over the whole epoch: the previous code
                    # materialised all 121M indices as Python ints per rank (~4.4 GB each,
                    # 8 ranks per node). Converting one micro-batch at a time keeps the
                    # epoch in a single numpy array instead.
                    yield per_rank[self.rank][offset : offset + self.micro_batch_size].tolist()

    def __len__(self) -> int:
        return self.num_batches * self.gradient_accumulation_steps

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self.num_skipped_batches = 0

    def skip_first_batches(self, num_batches: int):
        self.num_skipped_batches = num_batches


if __name__ == "__main__":
    sampler = DistributedBatchSampler()
