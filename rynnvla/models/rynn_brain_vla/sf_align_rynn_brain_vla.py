"""Spatial-Forcing style feature alignment for RynnBrain-VLA (option A, patch-token loss).

Faithful port of OpenHelix Spatial-Forcing (openpi-SF) onto the RynnVLA trunk, with
DA3MONO-LARGE replacing VGGT as the frozen 3D teacher:

- The loss is applied directly to the VLM trunk's *image patch token* hidden states at a
  mid layer (SF: layer 12 of 18, i.e. ~2/3 depth), NOT to extra query tokens. Gradients
  flow into the vision encoder and the first N trunk layers (SF's ablated mechanism:
  92.7 -> 96.9 on LIBERO).
- Projector identical in shape to SF's AlignProjector (projectors.py:6-58): optional
  LayerNorm on the VLM hidden, fc1(llm -> 2*teacher_dim), GELU, fc2(-> teacher_dim),
  xavier-uniform init, zero bias. SF's fc2 keeps 2*vggt_dim because VGGT features are
  2048-d; the invariant is `projector output dim == teacher feature dim` (DA3: 1024).
- Loss identical to SF's compute_align_loss_cosine: both sides L2-normalized per token,
  masked per-sample mean of (1 - cos), averaged over the batch.
- Teacher runs under no_grad + bf16 autocast on the *un-augmented* frames (SF feeds
  img_wo_aug), one view per batch element (N=1). DA3's any-view variant is deliberately
  NOT used: its cross-view attention makes a view's features depend on which other views
  share the forward (measured: cos 0.937 / 37% rel-L2 drift), which breaks target
  comparability across datasets with different camera counts. DA3MONO is batch-composition
  invariant (measured cos 0.999968).
- SF re-applies VGGT's positional embedding before pooling (use_vggt_pe) because VGGT's
  aggregator strips it from the token stream; DA3MONO features come straight out of
  DINOv2 blocks where positional information is already baked in, so there is no
  equivalent step here.
- SF pools the teacher grid to the VLA token resolution with bilinear interpolation and
  align_corners=True (vggt/heads/utils.py:155-176); we do the same per image, because the
  Qwen trunk's token grid is dynamic per image (image_grid_thw) instead of fixed.

The teacher is intentionally NOT registered as a submodule: it must stay out of the
state_dict (checkpoint bloat +1.3GB), out of the optimizer, and out of ZeRO partitioning.
"""
import json
import os
import sys
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ImageNet normalization: DA3's own InputProcessor (utils/io/input_processor.py:56).
_DA3_MEAN = (0.485, 0.456, 0.406)
_DA3_STD = (0.229, 0.224, 0.225)


class SFAlignProjector(nn.Module):
    """SF AlignProjector: LN? -> fc1(llm, 2*teacher) -> GELU -> fc2(-> teacher)."""

    def __init__(self, llm_dim: int, teacher_dim: int, use_vlm_norm: bool = True):
        super().__init__()
        self.vlm_norm = nn.LayerNorm(llm_dim) if use_vlm_norm else None
        self.fc1 = nn.Linear(llm_dim, 2 * teacher_dim, bias=True)
        self.act_fn1 = nn.GELU()
        self.fc2 = nn.Linear(2 * teacher_dim, teacher_dim, bias=True)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.vlm_norm is not None:
            hidden = self.vlm_norm(hidden)
        return self.fc2(self.act_fn1(self.fc1(hidden)))


class DA3MonoTeacher:
    """Frozen DA3MONO-LARGE feature extractor (not an nn.Module on purpose).

    Built straight from the checkpoint's own config.json via depth_anything_3's
    create_object. api.py is bypassed deliberately: its import chain pulls the
    Gaussian-Splatting export path (moviepy etc.).
    """

    def __init__(self, ckpt_dir: str, code_dir: Optional[str] = None, feat_layer: int = -1):
        try:
            from depth_anything_3.cfg import create_object
        except ImportError:
            if not code_dir:
                raise ImportError(
                    "depth_anything_3 is not importable and sf_teacher_code is not set; "
                    "point it at <Depth-Anything-3 repo>/src"
                )
            sys.path.insert(0, code_dir)
            from depth_anything_3.cfg import create_object
        from omegaconf import OmegaConf
        from safetensors.torch import load_file

        with open(os.path.join(ckpt_dir, "config.json")) as f:
            meta = json.load(f)
        net = create_object(OmegaConf.create(meta["config"]))

        state = load_file(os.path.join(ckpt_dir, "model.safetensors"))
        # Checkpoint keys carry the DepthAnything3 wrapper's `model.` prefix; without
        # stripping it, strict=False would silently leave the whole net at random init.
        state = {k[len("model."):] if k.startswith("model.") else k: v for k, v in state.items()}
        missing, unexpected = net.load_state_dict(state, strict=False)
        if unexpected:
            raise RuntimeError(f"DA3 teacher: unexpected keys, wrong checkpoint? {unexpected[:5]}")
        bad = [k for k in missing if not k.startswith("head.scratch.output_conv2_aux")]
        if bad:
            raise RuntimeError(f"DA3 teacher: missing non-head-aux keys {bad[:5]}")

        num_blocks = len(net.backbone.pretrained.blocks)
        self.feat_layer = feat_layer if feat_layer >= 0 else num_blocks - 1
        if not 0 <= self.feat_layer < num_blocks:
            raise ValueError(f"sf_teacher_layer {feat_layer} out of range for {num_blocks} blocks")

        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
        self.net = net
        self.feat_dim: Optional[int] = None  # filled on first forward

    def to(self, device):
        # Weights stay fp32: DA3's depth head disables autocast internally and expects
        # fp32 parameters; the DINOv2 backbone still runs bf16 via autocast in extract().
        self.net.to(device=device)
        return self

    @torch.no_grad()
    def extract(self, images: torch.Tensor) -> torch.Tensor:
        """images: (M, 3, H, W) in [0, 1], un-augmented. Returns (M, GH, GW, D) fp32.

        Each image is its own view set (N=1): DA3MONO is single-view, and per-view
        forwards keep targets independent of the sample's camera count.
        """
        mean = images.new_tensor(_DA3_MEAN).view(1, 3, 1, 1)
        std = images.new_tensor(_DA3_STD).view(1, 3, 1, 1)
        x = ((images - mean) / std).unsqueeze(1)  # (M, 1, 3, H, W)
        with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16):
            out = self.net(x, export_feat_layers=[self.feat_layer])
        feats = out.aux[f"feat_layer_{self.feat_layer}"]  # (M, 1, GH, GW, D)
        feats = feats[:, 0].float()
        if self.feat_dim is None:
            self.feat_dim = feats.shape[-1]
        return feats


def compute_sf_align_loss(
    mid_hidden: torch.Tensor,
    visual_pos_masks: torch.Tensor,
    image_grid_thw: torch.Tensor,
    teacher_feats: torch.Tensor,
    projector: SFAlignProjector,
    spatial_merge_size: int,
    teacher_grid_indices: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """SF cosine alignment between per-image VLM patch tokens and teacher features.

    mid_hidden:        (B, P, H) trunk hidden at the alignment layer.
    visual_pos_masks:  (B, P) bool, True at image-token positions.
    image_grid_thw:    (M, 3) batch-ordered grid rows, one per image.
    teacher_feats:     (M, GH, GW, D) fp32 teacher features, batch-ordered like the grid.
    Returns a scalar loss, or None when the layout cannot be matched (packing guard,
    video inputs) -- mirroring _compute_depth_loss's silent-skip contract.
    """
    if mid_hidden is None or visual_pos_masks is None or image_grid_thw is None:
        return None
    B = mid_hidden.size(0)
    if visual_pos_masks.size(0) != B:
        return None
    merge = spatial_merge_size**2
    tokens_per_image = (image_grid_thw.prod(dim=-1) // merge).tolist()
    if int(visual_pos_masks.sum().item()) != int(sum(tokens_per_image)):
        return None
    if teacher_grid_indices is None:
        if len(tokens_per_image) != teacher_feats.size(0):
            return None
        teacher_grid_indices = torch.arange(teacher_feats.size(0), device=image_grid_thw.device)
    teacher_grid_indices = teacher_grid_indices.reshape(-1).to(torch.long)
    if teacher_grid_indices.numel() != teacher_feats.size(0):
        return None
    if teacher_grid_indices.numel() and (
        teacher_grid_indices.min() < 0 or teacher_grid_indices.max() >= len(tokens_per_image)
    ):
        return None

    grid_spans = {}
    img_idx = 0
    for b in range(B):
        pos = visual_pos_masks[b].nonzero(as_tuple=True)[0]
        offset = 0
        while offset < pos.numel() and img_idx < len(tokens_per_image):
            n = int(tokens_per_image[img_idx])
            span = pos[offset : offset + n]
            if span.numel() != n:
                return None
            grid_spans[img_idx] = (b, span)
            offset += n
            img_idx += 1
    if img_idx != len(tokens_per_image):
        return None

    per_image_losses: List[torch.Tensor] = []
    for teacher_idx, grid_idx in enumerate(teacher_grid_indices.tolist()):
        b, span = grid_spans[grid_idx]
        n = int(tokens_per_image[grid_idx])
        t, gh, gw = (int(v) for v in image_grid_thw[grid_idx])
        gh, gw = gh // spatial_merge_size, gw // spatial_merge_size
        if t == 1 and gh * gw == n:
            student = projector(mid_hidden[b, span])
            target = teacher_feats[teacher_idx].permute(2, 0, 1).unsqueeze(0)
            target = F.interpolate(target, size=(gh, gw), mode="bilinear", align_corners=True)
            target = target[0].permute(1, 2, 0).reshape(n, -1)
            student = F.normalize(student.float(), dim=-1)
            target = F.normalize(target.to(student.dtype), dim=-1)
            per_image_losses.append(1.0 - (student * target).sum(dim=-1).mean())

    if not per_image_losses:
        return None
    return torch.stack(per_image_losses).mean()
