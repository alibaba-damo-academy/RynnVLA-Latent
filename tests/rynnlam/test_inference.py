from dataclasses import asdict

import numpy as np
import pytest
import torch

from rynnlam.config import RynnLAMConfig
from rynnlam.inference import RynnLAMEncoder
from rynnlam.modules import lam_v5
from rynnlam.model import build_model
from test_model import SyntheticEncoder, tiny_config


def test_checkpoint_roundtrip_and_strict_load(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(lam_v5, "DA3ViTLargeEncoder", SyntheticEncoder)
    config = RynnLAMConfig(**tiny_config(mode="freeze"))
    model = build_model(config).eval()
    path = tmp_path / "checkpoint.pt"
    torch.save({"config": asdict(config), "model_state_dict": model.state_dict()}, path)
    encoder = RynnLAMEncoder(path, device="cpu", precision="fp32")
    images = np.random.default_rng(7).random((1, 2, 28, 28, 3), dtype=np.float32)
    expected = model.encode_pair(torch.from_numpy(images))["k_tokens"]
    torch.testing.assert_close(encoder(images), expected)
    assert encoder(images, "z").shape == (1, 64)
    assert encoder(images, "zcam").shape == (1, 96)
    assert encoder(images, "hints_pool").shape == (1, 128)
    assert encoder(images, "full").shape == (1, 4, 224)
    with pytest.raises(ValueError):
        encoder(images * 2)
    with pytest.raises(ValueError):
        encoder(images[:, 0])
    state = model.state_dict()
    del state["hint_compressor.queries"]
    torch.save({"config": asdict(config), "model_state_dict": state}, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        RynnLAMEncoder(path, device="cpu", precision="fp32")
