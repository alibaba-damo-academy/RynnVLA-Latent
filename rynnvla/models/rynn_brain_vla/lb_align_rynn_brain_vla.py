"""LingBot-VLA 2.0 style query-readout distillation (option B) for RynnBrain-VLA.

The counterpart of ``sf_align_rynn_brain_vla`` (option A). Same frozen teacher
(DA3MONO-LARGE), same 224x224 un-augmented frames, same 16x16x1024 target grid --
only the *student-side mechanism* differs, so a fix2+SF vs fix2+LB pair isolates the
mechanism:

- option A (SF): the loss lands directly on the trunk's image-token hidden states at
  2/3 depth through a 2-layer MLP. The trunk must itself carry 3D structure.
- option B (LingBot): learnable query tokens are inserted immediately before the
  state token, and a Perceiver resampler reads [primary-view image tokens + query
  hiddens] out of the final layer to predict the teacher grid. The trunk is shaped
  only indirectly.

Ported from lingbot-vla-v2 (robbyant), which we read at:
- align_heads/resampler.py:29-202  PerceiverAttention / FeedForward / TaskTokenResampler
- modeling_lingbot_vla.py:799-806  depth_align_embs (256, llm_dim) + TaskTokenDepthHead
- modeling_lingbot_vla_v2.py:583-586  256 -> num_task_tokens block-mean pooling
- modeling_lingbot_vla.py:1419-1427  readout context = view-0 patches + task hiddens
- modeling_lingbot_vla.py:1604-1606  loss = smooth_l1(pred, target.detach()).mean()
- module_utils.py:429-445  teacher runs on camera 0 only; grid flattened row-major
- configs/vla/robotwin/robotwin.yaml:78-101  num_backbone_tokens 256, num_task_tokens 8,
  num_layers 1, num_heads 4, dim_head 32, ff_mult 1, dim_out 1024, weight 0.004

Implementation differences from the upstream architecture:
1. Teacher is DA3MONO instead of MoGe-2 + LingBot-Depth (MoRGBD): those weights are not
   public, and sharing SF's teacher is what makes the two arms comparable. DA3MONO
   features measure mean 0.02 / std 1.04, i.e. the same scale MoRGBD must have for
   LingBot's LayerNorm-terminated student to fit them with an unnormalized smooth_l1.
2. Default objective is cosine at weight 0.5 (SF's), not smooth_l1 at 0.004: measured,
   smooth_l1 starts at 0.72 here, so 0.004 would contribute 0.3% of the total loss and
   the run could not distinguish "mechanism does not help" from "aux loss was a no-op".
   ``lb_align_loss="smooth_l1"`` restores the published setting.
3. Current-frame arm only. LingBot's future-depth query needs a second decode of frame
   t+chunk in the dataset; it is a follow-up, not part of this comparison.
"""
import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


def reshape_tensor(x, heads):
    bs, length, width = x.shape
    x = x.view(bs, length, heads, -1)
    x = x.transpose(1, 2)
    return x.reshape(bs, heads, length, -1)


class PerceiverAttention(nn.Module):
    """Verbatim port of lingbot-vla-v2 align_heads/resampler.py:29-75.

    Note the two quirks that are kept on purpose: the keys/values are taken from
    ``cat(context, latents)`` (so queries also attend to each other), and the scale is
    applied as ``1/sqrt(sqrt(dim_head))`` on both q and k before the matmul.
    """

    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        x = self.norm1(x)
        latents = self.norm2(latents)

        b, l, _ = latents.shape

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v

        out = out.permute(0, 2, 1, 3).reshape(b, l, -1)
        return self.to_out(out)


class TaskTokenResampler(nn.Module):
    """Verbatim port of lingbot-vla-v2 align_heads/resampler.py:163-202."""

    def __init__(
        self,
        dim_in=768,
        dim_mid=1024,
        dim_head=64,
        dim_out=1024,
        num_layers=8,
        num_queries=8,
        num_heads=16,
        ff_mult=4,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.proj_in1 = nn.Linear(dim_in, dim_mid)
        self.proj_in2 = nn.Linear(dim_in, dim_mid)
        self.proj_out = nn.Linear(dim_mid, dim_out)
        self.norm_out = nn.LayerNorm(dim_out)

        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleList([
                    PerceiverAttention(dim=dim_mid, dim_head=dim_head, heads=num_heads),
                    FeedForward(dim=dim_mid, mult=ff_mult),
                ])
            )

    def forward(self, x, queries):
        queries = self.proj_in1(queries)
        x = self.proj_in2(x)
        for attn, ff in self.layers:
            queries = attn(x, queries) + queries
            queries = ff(queries) + queries
        queries = self.proj_out(queries)
        return self.norm_out(queries)


class LBQueryAlignHead(nn.Module):
    """LingBot's dual role for one parameter block: prefix tokens + readout queries.

    ``align_embs`` (num_queries, llm_dim) is used twice, exactly as in LingBot: pooled
    into ``num_task_tokens`` prefix tokens (block-mean over contiguous groups), and fed
    raw as the resampler's latents.
    """

    def __init__(
        self,
        llm_dim: int,
        teacher_dim: int,
        num_queries: int = 256,
        num_task_tokens: int = 8,
        num_layers: int = 1,
        num_heads: int = 4,
        dim_head: int = 32,
        ff_mult: int = 1,
        query_init_std: float = 1.0,
    ):
        super().__init__()
        if num_queries % num_task_tokens != 0:
            raise ValueError(
                f"lb_num_queries ({num_queries}) must be divisible by "
                f"lb_num_task_tokens ({num_task_tokens})"
            )
        self.num_queries = num_queries
        self.num_task_tokens = num_task_tokens
        self.query_init_std = float(query_init_std)
        self.align_embs = nn.Parameter(torch.randn(num_queries, llm_dim) * self.query_init_std)
        self.resampler = TaskTokenResampler(
            dim_in=llm_dim,
            dim_mid=llm_dim,
            dim_head=dim_head,
            dim_out=teacher_dim,
            num_layers=num_layers,
            num_queries=num_queries,
            num_heads=num_heads,
            ff_mult=ff_mult,
        )

    def task_tokens(self) -> torch.Tensor:
        """(num_task_tokens, llm_dim) block-mean pool of the queries."""
        group = self.num_queries // self.num_task_tokens
        return self.align_embs.view(self.num_task_tokens, group, -1).mean(dim=1)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """context: (B, L, llm_dim) -> (B, num_queries, teacher_dim)."""
        queries = self.align_embs.unsqueeze(0).expand(context.size(0), -1, -1)
        return self.resampler(context, queries.to(context.dtype))


def compute_lb_align_loss(
    final_hidden: torch.Tensor,
    visual_pos_masks: torch.Tensor,
    image_grid_thw: torch.Tensor,
    teacher_feats: torch.Tensor,
    query_mask: torch.Tensor,
    primary_grid_indices: torch.Tensor,
    head: LBQueryAlignHead,
    spatial_merge_size: int,
    loss_type: str = "cosine",
) -> torch.Tensor:
    """Compute LingBot query-readout distillation for each sample's primary image."""
    required = (final_hidden, visual_pos_masks, image_grid_thw, teacher_feats, query_mask, primary_grid_indices)
    if any(value is None for value in required):
        raise ValueError(
            "LBQ loss requires final_hidden, visual_pos_masks, image_grid_thw, teacher_feats, "
            "query_mask, and primary_grid_indices"
        )

    B = final_hidden.size(0)
    if visual_pos_masks.shape != final_hidden.shape[:2]:
        raise ValueError(
            f"LBQ visual mask shape={tuple(visual_pos_masks.shape)} does not match "
            f"hidden prefix={tuple(final_hidden.shape[:2])}"
        )
    if query_mask.shape != final_hidden.shape[:2]:
        raise ValueError(
            f"LBQ query mask shape={tuple(query_mask.shape)} does not match "
            f"hidden prefix={tuple(final_hidden.shape[:2])}"
        )
    query_counts = query_mask.sum(dim=1)
    if not torch.all(query_counts == head.num_task_tokens):
        raise ValueError(
            f"LBQ query mask must select {head.num_task_tokens} tokens per sample; "
            f"counts={query_counts.tolist()}"
        )
    if primary_grid_indices.shape != (B,):
        raise ValueError(
            f"primary_grid_indices must have shape ({B},), got {tuple(primary_grid_indices.shape)}"
        )

    merge = spatial_merge_size**2
    tokens_per_image = (image_grid_thw.prod(dim=-1) // merge).tolist()
    visual_tokens = int(visual_pos_masks.sum().item())
    grid_tokens = int(sum(tokens_per_image))
    if visual_tokens != grid_tokens:
        raise ValueError(
            f"LBQ visual token mismatch: visual_pos_masks has {visual_tokens}, grids require {grid_tokens}"
        )

    targets = teacher_feats.reshape(teacher_feats.size(0), -1, teacher_feats.size(-1))
    if teacher_feats.size(0) != B:
        raise ValueError(
            f"LBQ teacher batch={teacher_feats.size(0)} must equal sample batch={B}"
        )
    if targets.size(1) != head.num_queries:
        raise ValueError(
            f"teacher grid has {targets.size(1)} tokens but lb_num_queries="
            f"{head.num_queries}; LingBot maps them 1:1 with no interpolation, so the "
            "teacher input resolution and the query count must agree"
        )

    grid_spans = {}
    grid_owners = {}
    img_idx = 0
    for b in range(B):
        pos = visual_pos_masks[b].nonzero(as_tuple=True)[0]
        offset = 0
        while offset < pos.numel():
            if img_idx >= len(tokens_per_image):
                raise ValueError(f"LBQ sample {b} has more visual tokens than available image grids")
            n = int(tokens_per_image[img_idx])
            span = pos[offset : offset + n]
            if span.numel() != n:
                raise ValueError(
                    f"LBQ sample {b} grid {img_idx} requires {n} tokens but only "
                    f"{span.numel()} remain"
                )
            grid_spans[img_idx] = span
            grid_owners[img_idx] = b
            offset += n
            img_idx += 1
    if img_idx != len(tokens_per_image):
        raise ValueError(
            f"LBQ consumed {img_idx} image grids for batch size {B}, but received {len(tokens_per_image)}"
        )

    contexts: List[torch.Tensor] = []
    for b, primary_idx in enumerate(primary_grid_indices.to(torch.long).tolist()):
        if primary_idx not in grid_spans:
            raise ValueError(f"LBQ primary grid index {primary_idx} for sample {b} is out of range")
        if grid_owners[primary_idx] != b:
            raise ValueError(
                f"LBQ primary grid index {primary_idx} belongs to sample {grid_owners[primary_idx]}, "
                f"not sample {b}"
            )
        span = grid_spans[primary_idx]
        t, gh, gw = (int(v) for v in image_grid_thw[primary_idx])
        expected_primary = gh * gw // merge
        if t != 1 or expected_primary != span.numel():
            raise ValueError(
                f"LBQ sample {b} primary grid={(t, gh, gw)} maps to {expected_primary} tokens, "
                f"but its span has {span.numel()}"
            )
        contexts.append(torch.cat([final_hidden[b, span], final_hidden[b, query_mask[b]]], dim=0))

    groups: Dict[int, List[int]] = {}
    for b, ctx in enumerate(contexts):
        groups.setdefault(int(ctx.size(0)), []).append(b)

    weighted: List[torch.Tensor] = []
    for idxs in groups.values():
        ctx = torch.stack([contexts[b] for b in idxs], dim=0)
        pred = head(ctx).float()
        tgt = targets[idxs].to(pred.dtype)
        if loss_type == "cosine":
            cos = (F.normalize(pred, dim=-1) * F.normalize(tgt, dim=-1)).sum(dim=-1)
            group_loss = (1.0 - cos).mean()
        elif loss_type == "smooth_l1":
            group_loss = F.smooth_l1_loss(pred, tgt)
        else:
            raise ValueError(f"lb_align_loss must be 'cosine' or 'smooth_l1', got {loss_type!r}")
        weighted.append(group_loss * len(idxs))

    return torch.stack(weighted).sum() / B