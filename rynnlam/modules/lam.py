"""DA3-Large encoder wrapper."""

from pathlib import Path
from typing import Literal, Optional

import torch
import torch.nn as nn
from safetensors.torch import load_file

from .dinov2.dinov2 import DinoV2
from ..logger import logger


class DA3ViTLargeEncoder(nn.Module):
    def __init__(
        self,
        out_layers=(11, 15, 19, 23),
        finetune_mode: Literal["freeze", "full"] = "freeze",
        checkpoint_path: Optional[str] = None,
    ):
        super().__init__()
        if finetune_mode not in ("freeze", "full"):
            raise ValueError("finetune_mode must be freeze or full")
        self.out_layers = list(out_layers)
        self.finetune_mode = finetune_mode
        self.checkpoint_path = checkpoint_path
        self.model = DinoV2(
            name="vitl",
            out_layers=self.out_layers,
            alt_start=8,
            qknorm_start=8,
            rope_start=8,
            cat_token=True,
        )
        self.embed_dim = 1024
        for parameter in self.model.parameters():
            parameter.requires_grad = finetune_mode == "full"
        self._load_checkpoint()

    def _load_checkpoint(self):
        if self.checkpoint_path is None:
            logger.warning(
                "No encoder checkpoint supplied; using random initialization."
            )
            return
        path = Path(self.checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Encoder checkpoint not found: {path}. Fetch it beforehand and pass a "
                "local path; nothing is downloaded at runtime."
            )
        if path.suffix == ".safetensors":
            state = load_file(str(path), device="cpu")
        else:
            state = torch.load(str(path), map_location="cpu", weights_only=True)
            for key in ("model_state_dict", "state_dict", "model", "net"):
                if key in state:
                    state = state[key]
                    break
        # DA3 bundles contain task heads; select a complete backbone, then load strictly.
        for prefix in ("model.da3.backbone.", "model.backbone."):
            if any(key.startswith(prefix) for key in state):
                state = {
                    key[len(prefix) :]: value
                    for key, value in state.items()
                    if key.startswith(prefix)
                }
                break
        self.model.load_state_dict(state, strict=True)
        logger.info(f"Loaded DA3-Large encoder from {path}")

    def forward(self, images, **kwargs):
        return self.model(images.permute(0, 1, 4, 2, 3), **kwargs)
