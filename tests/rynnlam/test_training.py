"""CPU-only trainer regression tests, without DA3 weights or dataset scans."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from _scripts import load_script

train = load_script("train_rynnlam")
from rynnlam.config import RynnLAMConfig


class TinyModel(torch.nn.Module):
    patch_size = 14

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.2))

    def forward(self, images, **kwargs):
        return {"prediction": self.weight * images.mean()}


def sample():
    return {
        k: torch.ones(1)
        for k in ("images", "extrinsics", "intrinsics", "depths", "flow", "mask")
    }


@pytest.fixture
def trainer_factory(monkeypatch, tmp_path):
    calls = []
    manifest = tmp_path / "manifest.json"
    manifest.write_text('[{"scene_id": "sample", "file": "sample.safetensors"}]')
    monkeypatch.setattr(train, "build_model", lambda config: TinyModel())

    def loss(output, target, loss_config, global_step=0, **kwargs):
        calls.append(global_step)
        value = output["prediction"].square()
        return {"total_loss": value, "flow_loss": value}

    monkeypatch.setattr(train, "compute_lam_v5_losses", loss)

    def make(n=5, **overrides):
        settings = dict(
            device="cpu",
            output_dir=str(tmp_path),
            batch_size=1,
            gradient_accumulation_steps=2,
            lr_warmup_steps=1,
            vis_interval=0,
            save_interval_steps=0,
            eval_interval=0,
            save_interval=0,
            log_interval=100,
            num_workers=0,
            manifest_path=str(manifest),
        )
        settings.update(overrides)
        trainer = train.DDPTrainer(RynnLAMConfig(**settings), 0)
        trainer.sampler = SimpleNamespace(
            set_epoch=lambda epoch: None, seed=0, shuffle=True, drop_last=False
        )
        trainer.train_loader = [sample() for _ in range(n)]
        trainer.dataset = SimpleNamespace(
            scene_list=["sample"], get_scene=lambda scene: [sample()]
        )
        trainer.setup_data = lambda: None
        return trainer

    return make, calls


def test_partial_accumulation_and_budget_no_extra_step(trainer_factory):
    make, calls = trainer_factory
    trainer = make(max_steps=3)
    trainer.train_epoch(0)
    assert trainer.global_step == 3 and trainer.optimizer_step == 2
    assert trainer.scheduler.last_epoch == 2
    assert calls == [0, 1, 2]
    weights = trainer.model_raw.weight.detach().clone()
    assert trainer.train_epoch(0) == {}
    trainer.train()
    assert trainer.optimizer_step == 2 and trainer.scheduler.last_epoch == 2
    assert torch.equal(weights, trainer.model_raw.weight)


def test_optimizer_budget_counts_updates(trainer_factory):
    make, _ = trainer_factory
    trainer = make(n=20, target_effective_batch=2, max_steps=2)
    trainer.train_epoch(0)
    assert trainer.global_step == 4 and trainer.optimizer_step == 2


def test_last_partial_backward_syncs(trainer_factory, monkeypatch):
    make, _ = trainer_factory
    trainer = make(n=3)

    class FakeDDP(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.module = model
            self.unsynced = 0

        def forward(self, **kwargs):
            return self.module(**kwargs)

        def no_sync(self):
            self.unsynced += 1
            return nullcontext()

    monkeypatch.setattr(train, "DDP", FakeDDP)
    trainer.model = FakeDDP(trainer.model_raw)
    trainer.train_epoch(0)
    assert trainer.model.unsynced == 1  # batch 2 AND final batch 3 synchronize
    assert trainer.optimizer_step == 2


def test_rank_zero_eval_uses_raw_model_and_actual_step(trainer_factory):
    make, calls = trainer_factory
    trainer = make()
    trainer.global_step = 17
    trainer.model = Mock()
    trainer.model.side_effect = AssertionError(
        "DDP wrapper must not be called in evaluation"
    )
    trainer.evaluate(0)
    assert calls == [17]
    trainer.model.assert_not_called()


def test_checkpoint_resume_and_weights_only(trainer_factory):
    make, _ = trainer_factory
    original = make(max_steps=2)
    original.train_epoch(0)
    original.save_step_checkpoint(0)
    path = original.save_dir / "latest.pt"
    restored = make(max_steps=4)
    restored.load_checkpoint(path)
    assert restored.global_step == 2 and restored.optimizer_step == 1
    assert restored.batches_in_epoch == 2
    restored.train_epoch(0)
    assert restored.global_step == 4 and restored.optimizer_step == 2
    reset = make(reset_optimizer=True)
    reset.load_checkpoint(path)
    assert (
        reset.global_step == 0
        and reset.optimizer_step == 0
        and reset.batches_in_epoch == 0
    )
    assert not reset.optimizer.state
    assert torch.equal(reset.model_raw.weight, original.model_raw.weight)


def test_empty_loader_has_no_unbound_batch_index(trainer_factory):
    make, _ = trainer_factory
    trainer = make(n=0)
    assert trainer.train_epoch(0) == {}
    assert trainer.optimizer_step == 0


def test_unrepresentable_effective_batch_fails(trainer_factory):
    make, _ = trainer_factory
    with pytest.raises(ValueError, match="positive multiple"):
        make(batch_size=2, target_effective_batch=3)


def test_nonfinite_loss_discards_accumulated_gradients(trainer_factory, monkeypatch):
    make, _ = trainer_factory
    trainer = make(n=3)

    def loss(output, global_step, **kwargs):
        value = output["prediction"].square()
        if global_step == 1:
            value = value * float("nan")
        return {"total_loss": value, "flow_loss": value}

    monkeypatch.setattr(train, "compute_lam_v5_losses", loss)
    trainer.train_epoch(0)
    assert trainer.global_step == 3 and trainer.optimizer_step == 1
    assert torch.isfinite(trainer.model_raw.weight)


def test_checkpoint_deferred_until_gradient_boundary(trainer_factory):
    make, _ = trainer_factory
    trainer = make(n=3, save_interval_steps=1)
    seen = []
    trainer.save_step_checkpoint = lambda epoch: seen.append(trainer.global_step)
    trainer.train_epoch(0)
    assert seen == [2, 3]


@pytest.mark.parametrize(
    "change, field",
    [
        ({"batch_size": 2}, "batch_size"),
        ({"gradient_accumulation_steps": 1}, "accum_steps"),
        ({"min_frame_stride": 3}, "min_frame_stride"),
        ({"max_frame_stride": 8}, "max_frame_stride"),
        ({"max_sample_stride": 20}, "max_sample_stride"),
        ({"dataset_sampling_temperature": 1.5}, "dataset_sampling_temperature"),
        ({"dataset_sampling_weights": {"robot": 2.0}}, "dataset_sampling_weights"),
        (
            {"dataset_role_sampling_weights": {"wrist": 1.8}},
            "dataset_role_sampling_weights",
        ),
    ],
)
def test_resume_rejects_changed_layout(trainer_factory, change, field):
    make, _ = trainer_factory
    original = make(max_steps=2)
    original.train_epoch(0)
    original.save_step_checkpoint(0)
    restored = make(**change)
    with pytest.raises(ValueError, match=field + ".*reset_optimizer"):
        restored.load_checkpoint(original.save_dir / "latest.pt")
    assert restored.global_step == 0 and not restored.optimizer.state


def test_resume_rejects_changed_world_size(trainer_factory, monkeypatch):
    make, _ = trainer_factory
    original = make(max_steps=2)
    original.train_epoch(0)
    original.save_step_checkpoint(0)
    monkeypatch.setattr(train, "get_world_size", lambda: 2)
    restored = make()
    with pytest.raises(ValueError, match="world_size.*reset_optimizer"):
        restored.load_checkpoint(original.save_dir / "latest.pt")


def test_manifest_content_change_rejected_and_hash_cached(trainer_factory, monkeypatch):
    from pathlib import Path

    make, _ = trainer_factory
    original = make(max_steps=2)
    original.train_epoch(0)
    original.save_step_checkpoint(0)
    manifest = Path(original.config.manifest_path)
    manifest.write_text('[{"scene_id": "sample", "file": "changed.safetensors"}]')
    restored = make()
    with pytest.raises(ValueError, match="manifest_fingerprints.*reset_optimizer"):
        restored.load_checkpoint(original.save_dir / "latest.pt")
    # Existing data setups retain their snapshot; checkpoint saves do not rehash.
    monkeypatch.setattr(
        Path, "open", Mock(side_effect=AssertionError("unexpected file read"))
    )
    assert original._resume_data_layout() is original._data_layout
    assert restored._resume_data_layout() is restored._data_layout


@pytest.mark.parametrize("position", [0, 2])
def test_historical_checkpoint_layout_policy(trainer_factory, monkeypatch, position):
    make, _ = trainer_factory
    original = make(max_steps=2)
    original.train_epoch(0)
    original.save_step_checkpoint(0)
    path = original.save_dir / "latest.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    del checkpoint["resume_data_layout"]
    checkpoint["batches_in_epoch"] = position
    torch.save(checkpoint, path)
    restored = make()
    if position:
        with pytest.raises(ValueError, match="no resume data-layout.*reset_optimizer"):
            restored.load_checkpoint(path)
    else:
        warning = Mock()
        monkeypatch.setattr(train.logger, "warning", warning)
        restored.load_checkpoint(path)
        assert any(
            "sample-exact resume" in str(call) for call in warning.call_args_list
        )
    reset = make(reset_optimizer=True, batch_size=3)
    reset.load_checkpoint(path)
    assert reset.global_step == reset.optimizer_step == reset.batches_in_epoch == 0
    assert not reset.optimizer.state


@pytest.mark.parametrize("reset_optimizer", [False, True])
def test_checkpoint_deserializes_on_cpu(trainer_factory, monkeypatch, reset_optimizer):
    make, _ = trainer_factory
    original = make(max_steps=2)
    original.train_epoch(0)
    original.save_step_checkpoint(0)
    load = Mock(wraps=torch.load)
    monkeypatch.setattr(torch, "load", load)
    restored = make(reset_optimizer=reset_optimizer)
    restored.load_checkpoint(original.save_dir / "latest.pt")
    assert load.call_args.kwargs == {"map_location": "cpu", "weights_only": True}
    assert bool(restored.optimizer.state) is not reset_optimizer


def test_same_process_resume_replays_only_remaining_samples(trainer_factory):
    make, _ = trainer_factory
    original = make(max_steps=2)
    original.train_loader = [
        dict(sample(), images=torch.tensor([float(i + 1)])) for i in range(5)
    ]
    original.train_epoch(0)
    original.save_step_checkpoint(0)
    restored = make(max_steps=5)
    restored.train_loader = original.train_loader
    restored.load_checkpoint(original.save_dir / "latest.pt")
    seen = []
    restored.model_raw.register_forward_pre_hook(
        lambda module, args, kwargs: seen.append(kwargs["images"].item()),
        with_kwargs=True,
    )
    restored.train()
    assert seen == [3, 4, 5]
    assert restored.global_step == 5 and restored.optimizer_step == 3


@pytest.mark.parametrize("target_effective_batch, budget", [(0, 2), (2, 1)])
def test_resumed_complete_budget_never_steps(
    trainer_factory, target_effective_batch, budget
):
    make, _ = trainer_factory
    original = make(max_steps=budget, target_effective_batch=target_effective_batch)
    original.train_epoch(0)
    original.save_step_checkpoint(0)
    restored = make(
        max_steps=budget,
        target_effective_batch=target_effective_batch,
        resume=str(original.save_dir / "latest.pt"),
    )
    restored.optimizer.step = Mock(side_effect=AssertionError("extra optimizer step"))
    restored.scheduler.step = Mock(side_effect=AssertionError("extra scheduler step"))
    restored.train()
    assert restored.global_step == original.global_step
    assert restored.scheduler.last_epoch == original.scheduler.last_epoch
    restored.optimizer.step.assert_not_called()
    restored.scheduler.step.assert_not_called()


def test_epoch_checkpoint_preserves_layout(trainer_factory):
    make, _ = trainer_factory
    original = make(n=2)
    original.train_epoch(0)
    original.batches_in_epoch = 0
    original.save_checkpoint(0, {})
    restored = make(n=2)
    restored.load_checkpoint(original.save_dir / "latest.pt")
    assert restored.current_epoch == 1 and restored.batches_in_epoch == 0
    assert restored._resume_data_layout() == original._resume_data_layout()


def test_manifest_directory_fingerprints_all_selected_files(trainer_factory, tmp_path):
    make, _ = trainer_factory
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    (manifests / "manifest.json").write_text("[]")
    (manifests / "manifest_first.json").write_text("[]")
    child = manifests / "child"
    child.mkdir()
    nested = child / "manifest_second.json"
    nested.write_text("[]")
    original = make(manifest_path=str(manifests))
    original.save_step_checkpoint(0)
    assert len(original._resume_data_layout()["manifest_fingerprints"]) == 3
    nested.write_text('[{"scene_id": "new"}]')
    restored = make(manifest_path=str(manifests))
    with pytest.raises(ValueError, match="manifest_fingerprints"):
        restored.load_checkpoint(original.save_dir / "latest.pt")
