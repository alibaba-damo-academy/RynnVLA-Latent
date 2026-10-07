"""Auxiliary depth-prediction head for RynnBrain-VLA.

A lightweight query-based readout: a fixed set of learnable query tokens cross-attend
to the VLM's image-token hidden states and regress a G x G depth grid. It is a pure
readout (queries are separate parameters that attend to the trunk; they are NOT injected
into the main token sequence), so the action stream can never attend to it -- the
"block-to-action" isolation holds by construction. The head is used only at training and
dropped at inference (zero deployment cost).

Two heads share this module: current-frame depth (an anchor that prevents the future
head from collapsing) and future-frame depth (foresight / dynamics signal). Both are
predicted from the *current* observation's visual tokens and supervised against the
paired ground-truth depth maps (pooled to G x G) with a validity mask.
"""
import torch
import torch.nn as nn


class _CrossAttnBlock(nn.Module):
    def __init__(self, dim, num_heads, ff_mult=4):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Linear(dim * ff_mult, dim),
        )

    def forward(self, q, kv, key_padding_mask=None):
        qn = self.norm_q(q)
        kvn = self.norm_kv(kv)
        attn_out, _ = self.attn(qn, kvn, kvn, key_padding_mask=key_padding_mask, need_weights=False)
        q = q + attn_out
        q = q + self.ff(self.norm_ff(q))
        return q


class DepthReadoutHead(nn.Module):
    """Query-based depth readout. Predicts a flattened G*G depth grid per image."""

    def __init__(self, dim, grid_size=16, num_layers=2, num_heads=8):
        super().__init__()
        self.grid_size = grid_size
        self.num_queries = grid_size * grid_size
        self.query = nn.Parameter(torch.randn(self.num_queries, dim) * 0.02)
        self.blocks = nn.ModuleList([_CrossAttnBlock(dim, num_heads) for _ in range(num_layers)])
        self.norm_out = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, 1)

    def forward(self, hidden, key_padding_mask):
        """hidden: (B, N, dim) trunk states; key_padding_mask: (B, N) True = ignore.

        Returns predicted depth grid, flattened: (B, G*G).
        """
        b = hidden.size(0)
        q = self.query.unsqueeze(0).expand(b, -1, -1)
        for blk in self.blocks:
            q = blk(q, hidden, key_padding_mask=key_padding_mask)
        return self.out(self.norm_out(q)).squeeze(-1)
