"""Keep an explicit global-shuffle arm reproducible without changing existing resumes.

``sampler_shuffle`` is in RESUME_CRITICAL_FIELDS, so a checkpoint trained under one mode can
never be resumed under the other. These tests pin both halves: that "global" reproduces the
historical rank-strided order exactly, and that "auto" (the default) still gets episode
locality -- plus the resume guards that make a mode switch loud instead of silent.
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from rynnvla.api.train import resolve_resume, validate_options

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("_shuffle_sampler", ROOT / "rynnvla/training/sampler.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
DistributedBatchSampler = module.DistributedBatchSampler


class EpisodeDataset:
    episode_lengths = np.arange(20, 84)

    def __len__(self):
        return int(self.episode_lengths.sum())


def sampler(rank=0, **kwargs):
    return DistributedBatchSampler(
        EpisodeDataset(), None, 4, rank, 2, 2, seed=42, num_workers=4,
        drop_last=True, **kwargs,
    )


def test_global_matches_legacy_order_on_every_rank_after_resume():
    size = len(EpisodeDataset())
    for epoch in (0, 1):
        order = torch.randperm(size, generator=torch.Generator().manual_seed(42 + epoch)).numpy()
        for rank in range(4):
            value = sampler(rank, shuffle_mode="global")
            value.set_epoch(epoch)
            # Historical sampler: rank-stride, then accumulation's micro-batches.
            expected = order[:value.num_batches * 16][rank::4].reshape(-1, 2).tolist()
            assert list(value) == expected
            value.skip_first_batches(17)
            assert list(value) == expected[17:]


def test_auto_preserves_locality_and_global_does_not_use_it():
    old = sampler()
    assert np.array_equal(old._get_global_batch_indices()[0], old._locality_permutation(42)[::4])
    value = sampler(shuffle_mode="global")
    value._locality_permutation = lambda _: pytest.fail("global must never enter locality")
    list(value)


def test_invalid_shuffle_fails():
    with pytest.raises(ValueError, match="shuffle"):
        sampler(shuffle_mode="typo")
    with pytest.raises(ValueError, match="sampler_shuffle"):
        validate_options({"sampler_shuffle": "typo"})


@pytest.mark.parametrize("saved", [None, "auto", "global"])
def test_resume_pins_sampling_mode_including_pre_option_checkpoints(tmp_path, saved):
    checkpoint = tmp_path / "checkpoint-5000"
    checkpoint.mkdir()
    for name in ("config.json", "processor_config.json"):
        (checkpoint / name).write_text("{}")
    metadata = {} if saved is None else {"sampler_shuffle": saved}
    (checkpoint / "resume_args.json").write_text(json.dumps(metadata))
    current = "auto" if saved is None else saved
    values = {"output_dir": str(tmp_path), "sampler_shuffle": current}
    assert resolve_resume(values, True, lambda _: str(checkpoint)) == str(checkpoint)
    if saved is None:
        assert resolve_resume({"output_dir": str(tmp_path)}, True, lambda _: str(checkpoint)) == str(checkpoint)
    other = "global" if current == "auto" else "auto"
    with pytest.raises(ValueError, match="sampler_shuffle"):
        resolve_resume({**values, "sampler_shuffle": other}, True, lambda _: str(checkpoint))


def test_checkpoint_without_any_metadata_cannot_switch_to_global(tmp_path):
    for name in ("config.json", "processor_config.json"):
        (tmp_path / name).write_text("{}")
    with pytest.raises(ValueError, match="sampler_shuffle"):
        resolve_resume({"output_dir": str(tmp_path), "sampler_shuffle": "global"}, True,
                       lambda _: str(tmp_path))
