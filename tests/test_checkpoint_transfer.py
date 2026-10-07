from types import SimpleNamespace

import pytest
import torch

from rynnvla import models


@pytest.fixture(autouse=True)
def local_parallel_state(monkeypatch):
    monkeypatch.setattr(models.mpu, "get_expert_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(models.mpu, "get_expert_model_parallel_group", lambda: None)
    monkeypatch.setattr(models.mpu, "get_expert_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(models.mpu, "get_expert_data_parallel_rank", lambda: 0)
    monkeypatch.setattr(models.mpu, "get_data_parallel_rank", lambda: 0)


def test_checkpoint_prefix_remapping():
    source = {"model.visual.weight": torch.ones(2), "other": torch.zeros(1)}
    target = {"visual.weight": torch.empty(2)}
    mapped = models._remap_checkpoint_keys(source, target, "model")
    assert torch.equal(mapped["visual.weight"], source["model.visual.weight"])
    assert "other" in mapped


def test_checkpoint_shape_mismatch_does_not_broadcast(tmp_path):
    torch.save({"weight": torch.ones(1)}, tmp_path / "pytorch_model.bin")
    target = {"weight": torch.zeros(2)}
    with pytest.raises(ValueError, match="has shape"):
        models._load_checkpoint_file(str(tmp_path), "pytorch_model.bin", {}, target, {"weight"})
    assert torch.equal(target["weight"], torch.zeros(2))


def test_checkpoint_copies_values_and_reports_missing(tmp_path):
    expected = torch.arange(6).reshape(2, 3).float()
    torch.save({"model.weight": expected}, tmp_path / "pytorch_model.bin")
    target = {"weight": torch.zeros_like(expected), "bias": torch.zeros(2)}
    missing = set(target)
    models._load_checkpoint_file(str(tmp_path), "pytorch_model.bin", {}, target, missing, prefix="model")
    assert torch.equal(target["weight"], expected)
    assert missing == {"bias"}


def _transfer_model():
    model = torch.nn.Linear(2, 2)
    model.base_model_prefix = "model"
    model.config = SimpleNamespace(use_latent_head_readout=True)
    return model


@pytest.mark.parametrize("missing", [[], ["latent_readout_proj.weight", "latent_readout_proj.bias"]])
def test_stage_transfer_allows_only_new_readout(monkeypatch, missing):
    monkeypatch.setattr(models.AutoConfig, "from_pretrained", lambda path: SimpleNamespace(
        model_type="rynn_brain_vla", tie_word_embeddings=False, use_latent_actions=True
    ))
    monkeypatch.setattr(models, "_load_checkpoint_files", lambda *args, **kwargs: missing)
    model = _transfer_model()
    assert models._load_pretrained_weights(model, model.state_dict(), "stage1") == missing


@pytest.mark.parametrize("missing", ["action_expert.layers.0.weight", "latent_action_head.0.weight", "slot_seed_proj.weight"])
def test_stage_transfer_rejects_missing_pretrained_modules(monkeypatch, missing):
    monkeypatch.setattr(models.AutoConfig, "from_pretrained", lambda path: SimpleNamespace(
        model_type="rynn_brain_vla", tie_word_embeddings=False, use_latent_actions=True
    ))
    monkeypatch.setattr(models, "_load_checkpoint_files", lambda *args, **kwargs: [missing])
    model = _transfer_model()
    with pytest.raises(ValueError, match="Incomplete RynnVLA checkpoint transfer"):
        models._load_pretrained_weights(model, model.state_dict(), "stage1")
