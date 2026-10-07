"""CPU tests; set RYNNLAM_BASELINE_ROOT to enable historical source parity."""

from dataclasses import make_dataclass
import importlib.util
import os
from pathlib import Path
import sys
import types

import pytest
import torch
from torch import nn

from rynnlam.model import RynnLAM, build_model
from rynnlam.modules import lam_v5
from rynnlam.modules.losses_v5 import LAMv5LossConfig, compute_lam_v5_losses


class SyntheticEncoder(nn.Module):
    def __init__(self, finetune_mode="freeze", **kwargs):
        super().__init__()
        self.scale = nn.Parameter(
            torch.ones(2048), requires_grad=finetune_mode == "full"
        )
        self.last_images = None

    def forward(self, images):
        self.last_images = images
        values = images[:, :, ::14, ::14].mean(-1).flatten(2)
        channels = torch.arange(2048, device=images.device, dtype=images.dtype) / 2048
        features = torch.sin(values.unsqueeze(-1) + channels) * self.scale
        return [(features, None)], None


def tiny_config(source="features", k=2, mode="full"):
    return dict(
        latent_encoder_model_dim=32,
        latent_encoder_num_heads=4,
        latent_encoder_num_blocks=1,
        flow_decoder_model_dim=32,
        flow_decoder_num_heads=4,
        flow_decoder_dec_blocks=1,
        recon_decoder_model_dim=32,
        recon_decoder_num_heads=4,
        recon_decoder_num_blocks=1,
        cam_dec_hidden_dim=16,
        k_token_source=source,
        num_k_tokens=k,
        encoder_finetune_mode=mode,
    )


@pytest.fixture(autouse=True)
def cpu_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.fixture
def synthetic(monkeypatch):
    monkeypatch.setattr(lam_v5, "DA3ViTLargeEncoder", SyntheticEncoder)


def inputs():
    images = torch.rand(2, 2, 28, 28, 3)
    extrinsics = torch.eye(4).repeat(2, 2, 1, 1)
    extrinsics[:, 1, :3, 3] = 0.02
    intrinsics = torch.eye(3).repeat(2, 2, 1, 1)
    intrinsics[:, :, 0, 0] = 20
    intrinsics[:, :, 1, 1] = 20
    return images, extrinsics, intrinsics, torch.ones(2, 2, 28, 28)


@pytest.mark.parametrize(
    "source,k", [("features", 2), ("features", 4), ("features", 8), ("hints", 2)]
)
def test_encode_and_gradients(synthetic, source, k):
    model = build_model(tiny_config(source, k))
    images, extrinsics, intrinsics, depths = inputs()
    encoded = model.encode_pair(images)
    assert encoded["latent_action"].shape == (2, 64)
    assert encoded["camera_pose_latent"].shape == (2, 32)
    assert encoded["motion_hints"].shape == (2, 4, 128)
    assert encoded["k_tokens"].shape == (2, k, 2048 if source == "features" else 128)
    torch.testing.assert_close(
        model.encoder.last_images,
        (images - images.new_tensor([0.485, 0.456, 0.406]))
        / images.new_tensor([0.229, 0.224, 0.225]),
    )
    raw = model.encode_pair(images, normalize=False)
    torch.testing.assert_close(model.encoder.last_images, images)
    assert not torch.equal(raw["target_features"], encoded["target_features"])
    with torch.no_grad():
        model.recon_decoder_ktoken.out_proj.weight.normal_(std=0.01)
    output = model(images, extrinsics, intrinsics, depths)
    assert not output["source_features"].requires_grad
    assert not output["target_features"].requires_grad
    losses = compute_lam_v5_losses(
        output,
        dict(flows=torch.rand_like(output["flow_3d"]) * 0.05, extrinsics=extrinsics),
        LAMv5LossConfig(
            lambda_recon=5, use_z_residual_loss=True, use_contrastive_feat_loss=True
        ),
    )
    assert "z_gain_recon" in losses
    losses["total_loss"].backward()
    assert model.hint_compressor.queries.grad is not None
    assert model.hint_compressor.queries.grad.abs().sum() > 0
    assert model.latent_encoder.out_proj.weight.grad is not None
    assert model.encoder.scale.grad is not None


def test_builder_and_validation(synthetic):
    cfg = tiny_config(mode="freeze")
    Config = make_dataclass(
        "Config", [(key, type(value), value) for key, value in cfg.items()]
    )
    model = build_model(Config())
    assert model.hint_compressor is not None
    assert not model.encoder.scale.requires_grad
    with pytest.raises(ValueError):
        build_model(dict(cfg, k_token_source="invalid"))
    with pytest.raises(ValueError):
        model.encode_pair(torch.rand(2, 3, 28, 28, 3))
    with pytest.raises(TypeError):
        model.encode_pair(torch.zeros(2, 2, 28, 28, 3, dtype=torch.uint8))


def test_backbone_checkpoint_loading(monkeypatch, tmp_path):
    from rynnlam.modules import lam
    from safetensors.torch import save_file

    class TinyBackbone(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(2, 2))

    monkeypatch.setattr(lam, "DinoV2", TinyBackbone)
    path = tmp_path / "model.safetensors"
    save_file(
        {
            "model.backbone.weight": torch.ones(2, 2),
            "model.depth_head.weight": torch.zeros(1),
        },
        str(path),
    )
    encoder = lam.DA3ViTLargeEncoder(checkpoint_path=str(path))
    torch.testing.assert_close(encoder.model.weight, torch.ones(2, 2))
    assert not encoder.model.weight.requires_grad
    save_file({"model.backbone.unexpected": torch.ones(2, 2)}, str(path))
    with pytest.raises(RuntimeError):
        lam.DA3ViTLargeEncoder(checkpoint_path=str(path))
    with pytest.raises(ValueError):
        lam.DA3ViTLargeEncoder(finetune_mode="lora")


def baseline_modules():
    root = os.environ.get("RYNNLAM_BASELINE_ROOT")
    if not root:
        pytest.skip("set RYNNLAM_BASELINE_ROOT for source parity")
    directory = Path(root) / "lam" / "modules"
    package = "_rynnlam_baseline"
    if package not in sys.modules:
        for name, path in [
            (package, directory.parent),
            (package + ".modules", directory),
        ]:
            mod = types.ModuleType(name)
            mod.__path__ = [str(path)]
            sys.modules[name] = mod
        logger = types.ModuleType(package + ".logger")
        import logging

        logger.logger = logging.getLogger("baseline")
        sys.modules[logger.__name__] = logger

    def load(name):
        fullname = package + ".modules." + name
        spec = importlib.util.spec_from_file_location(
            fullname, directory / (name + ".py")
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[fullname] = module
        spec.loader.exec_module(module)
        return module

    return load("lam_v5"), load("losses_v5")


@pytest.mark.parametrize(
    "source,k,mode",
    [
        ("features", 2, "freeze"),
        ("features", 4, "full"),
        ("features", 8, "full"),
        ("hints", 2, "full"),
    ],
)
def test_historical_full_graph_parity(synthetic, monkeypatch, source, k, mode):
    baseline, old_losses = baseline_modules()
    monkeypatch.setattr(baseline, "DA3ViTLargeEncoder", SyntheticEncoder)
    cfg = tiny_config(source, k, mode)
    cfg.update(
        latent_dim=64,
        camera_pose_latent_dim=32,
        motion_hint_dim=128,
        motion_hints_dropout=0.8,
        flow_mask_ratio=0.5,
        warp_num_sample_points=8,
        adversarial_alpha=2.0,
    )
    torch.manual_seed(41)
    old = baseline.LAMv5(**cfg, use_k_token_recon=True)
    old.log_z_utilization = True
    torch.manual_seed(41)
    new = build_model(cfg)
    assert list(old.state_dict()) == list(new.state_dict())
    assert list(dict(old.named_parameters())) == list(dict(new.named_parameters()))
    for key, value in old.state_dict().items():
        torch.testing.assert_close(value, new.state_dict()[key], rtol=0, atol=0)
    with torch.no_grad():
        old.recon_decoder_ktoken.out_proj.weight.normal_(std=0.01)
    new.load_state_dict(old.state_dict(), strict=True)
    args = inputs()
    torch.manual_seed(57)
    expected = old(*args)
    torch.manual_seed(57)
    actual = new(*args)
    for key, value in actual.items():
        if key != "k_tokens" and value is not None:
            torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
    target = dict(flows=torch.rand_like(actual["flow_3d"]) * 0.1, extrinsics=args[1])
    loss_kwargs = dict(
        lambda_pose=5,
        lambda_flow=5,
        lambda_feat=3,
        lambda_adversarial=2,
        use_contrastive_feat_loss=True,
        contrast_temperature=0.2,
        lambda_recon=5,
        use_z_residual_loss=True,
        z_recon_warmup_steps=800,
    )
    old_l = old_losses.compute_lam_v5_losses(
        expected,
        target,
        old_losses.LAMv5LossConfig(**loss_kwargs),
        global_step=400,
        flow_warmup_steps=2000,
    )
    new_l = compute_lam_v5_losses(
        actual,
        target,
        LAMv5LossConfig(**loss_kwargs),
        global_step=400,
        flow_warmup_steps=2000,
    )
    assert old_l.keys() == new_l.keys()
    for key in old_l:
        torch.testing.assert_close(old_l[key], new_l[key], rtol=0, atol=0)
    old_l["total_loss"].backward()
    new_l["total_loss"].backward()
    for (name, p), (_, q) in zip(old.named_parameters(), new.named_parameters()):
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, rtol=1e-5, atol=1e-6, msg=name)
    old.eval()
    new.eval()
    with torch.no_grad():
        expected = old(*args)
        actual = new(*args)
        for key in actual:
            if key != "k_tokens" and actual[key] is not None:
                torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
