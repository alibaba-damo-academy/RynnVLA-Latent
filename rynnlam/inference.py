"""Checkpoint loading and frame-pair encoding without benchmark dependencies."""

from contextlib import nullcontext
from dataclasses import asdict, is_dataclass
import hashlib
from pathlib import Path

import torch


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_sha256():
    import cv2
    import numpy as np

    digest = hashlib.sha256()
    root = Path(__file__).parent
    for source in sorted(root.rglob("*.py")):
        digest.update(str(source.relative_to(root)).encode())
        digest.update(source.read_bytes())
    digest.update(f"{torch.__version__}|{np.__version__}|{cv2.__version__}".encode())
    return digest.hexdigest()


def load_checkpoint(path, *, device="cpu", trust_checkpoint=False):
    from .model import build_model

    checkpoint = torch.load(path, map_location="cpu", weights_only=not trust_checkpoint)
    if (
        not isinstance(checkpoint, dict)
        or not {"config", "model_state_dict"} <= checkpoint.keys()
    ):
        raise ValueError(
            "Expected a checkpoint with config and model_state_dict fields"
        )
    config = checkpoint["config"]
    if is_dataclass(config):
        config = asdict(config)
    if not isinstance(config, dict):
        raise ValueError("Checkpoint config must be a flat dictionary")
    config = dict(config)
    # Full checkpoints already contain the backbone; do not load its old local path.
    config["encoder_checkpoint_path"] = None
    config["encoder_finetune_mode"] = "freeze"
    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.requires_grad_(False).eval().to(device)
    return model, config


class RynnLAMEncoder:
    """Encode RGB frame pairs into K tokens or diagnostic representations."""

    def __init__(
        self,
        checkpoint,
        device="cuda",
        normalize=True,
        precision="bf16",
        trust_checkpoint=False,
        checkpoint_sha256=None,
    ):
        if precision not in {"fp32", "bf16"}:
            raise ValueError("precision must be fp32 or bf16")
        self.checkpoint = Path(checkpoint).expanduser().resolve(strict=True)
        self.device = torch.device(device)
        self.normalize = bool(normalize)
        self.precision = precision
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; select device='cpu'")
        if (
            self.device.type == "cuda"
            and precision == "bf16"
            and not torch.cuda.is_bf16_supported()
        ):
            raise RuntimeError(
                "This GPU does not support bfloat16; select precision='fp32'"
            )
        self.checkpoint_sha256 = checkpoint_sha256 or file_sha256(self.checkpoint)
        self.model, self.config = load_checkpoint(
            self.checkpoint, device=self.device, trust_checkpoint=trust_checkpoint
        )

    @torch.inference_mode()
    def __call__(self, images, representation="ktoken"):
        images = torch.as_tensor(images, dtype=torch.float32, device=self.device)
        if images.ndim != 5 or images.shape[1] != 2 or images.shape[-1] != 3:
            raise ValueError("Expected RGB images [batch, 2, height, width, 3]")
        if not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1:
            raise ValueError("RGB images must be finite floats in [0, 1]")
        context = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if self.device.type == "cuda" and self.precision == "bf16"
            else nullcontext()
        )
        with context:
            features = self.model.encode_pair(images, normalize=self.normalize)
        z, camera, hints = (
            features["latent_action"],
            features["camera_pose_latent"],
            features["motion_hints"],
        )
        if representation == "ktoken":
            tokens = features["k_tokens"]
            if tokens is None:
                raise ValueError("This checkpoint has no K-token compressor")
        elif representation == "z":
            tokens = z
        elif representation == "zcam":
            tokens = torch.cat([z, camera], dim=-1)
        elif representation == "ktoken_zcam":
            # Combined latent action: k_token (K*d, flattened) + z + camera.
            # For the b512 arm this is 8*64 + 64 + 32 = 608 per frame pair.
            k_tokens = features["k_tokens"]
            if k_tokens is None:
                raise ValueError("This checkpoint has no K-token compressor")
            tokens = torch.cat([k_tokens.flatten(1), z, camera], dim=-1)
        elif representation == "hints_pool":
            tokens = hints.mean(dim=1)
        elif representation == "full":
            n = hints.shape[1]
            tokens = torch.cat(
                [
                    z[:, None].expand(-1, n, -1),
                    camera[:, None].expand(-1, n, -1),
                    hints,
                ],
                dim=-1,
            )
        elif representation == "encoder":
            tokens = torch.stack(
                [features["source_features"], features["target_features"]], dim=1
            )
        else:
            raise ValueError(f"Unknown representation: {representation}")
        return tokens.float().detach()
