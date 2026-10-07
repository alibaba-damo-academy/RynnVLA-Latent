import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from torch.nn.functional import pad


def patchify(videos: Tensor, size: int) -> Tensor:
    B, T, H, W, C = videos.shape
    h_pad = -H % size
    w_pad = -W % size
    videos = pad(videos, (0, 0, 0, w_pad, 0, h_pad, 0, 0), mode="constant", value=0)
    x = rearrange(
        videos, "b t (hn hp) (wn wp) c -> b t (hn wn) (hp wp c)", hp=size, wp=size
    )
    return x


def unpatchify(patches: Tensor, size: int, h_out: int, w_out: int) -> Tensor:
    h_pad = -h_out % size
    w_pad = -w_out % size
    h_padded = h_out + h_pad
    w_padded = w_out + w_pad
    hn = h_padded // size
    wn = w_padded // size
    x = rearrange(
        patches,
        "b t (hn wn) (hp wp c) -> b t (hn hp) (wn wp) c",
        hp=size,
        wp=size,
        hn=hn,
        wn=wn,
    )
    return x[:, :, :h_out, :w_out]


class PositionalEncoding(nn.Module):

    def __init__(self, model_dim: int, max_len: int = 5000) -> None:
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        exponent = torch.arange(0, model_dim, 2).float() * -(
            math.log(10000.0) / model_dim
        )
        div_term = torch.exp(exponent)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pos_enc", pe, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pos_enc[: x.shape[2]].to(x.device)


class SelfAttention(nn.Module):

    def __init__(
        self, model_dim: int, num_heads: int, dropout: float = 0.0, rope=None
    ) -> None:
        super(SelfAttention, self).__init__()
        self.heads = num_heads
        self.dropout = dropout
        self.to_q = nn.Linear(model_dim, model_dim, bias=False)
        self.to_k = nn.Linear(model_dim, model_dim, bias=False)
        self.to_v = nn.Linear(model_dim, model_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(model_dim, model_dim), nn.Dropout(dropout)
        )
        self.rope = rope

    def forward(self, x: Tensor, is_causal: bool = False, pos: Tensor = None) -> Tensor:
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), (q, k, v)
        )
        if self.rope is not None and pos is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=is_causal,
            dropout_p=self.dropout if self.training else 0.0,
        )
        del q, k, v
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


def modulate(norm_x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return norm_x * (1 + scale) + shift


class AdaLNSpatioBlock(nn.Module):

    def __init__(
        self, model_dim: int, num_heads: int, dropout: float = 0.0, rope=None
    ) -> None:
        super(AdaLNSpatioBlock, self).__init__()
        self.norm1 = nn.LayerNorm(model_dim, elementwise_affine=False, eps=1e-06)
        self.attn = SelfAttention(model_dim, num_heads, dropout=dropout, rope=rope)
        self.norm2 = nn.LayerNorm(model_dim, elementwise_affine=False, eps=1e-06)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 4, model_dim),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(model_dim, 6 * model_dim, bias=True)
        )

    def forward(self, x: Tensor, cond: Tensor, pos: Tensor = None) -> Tensor:
        B, T_minus_1, N, E = x.shape
        x_flat = rearrange(x, "b t n e -> (b t) n e")
        cond_expanded = cond.expand(-1, -1, N, -1)
        cond_flat = rearrange(cond_expanded, "b t n e -> (b t) n e")
        adaln_params = self.adaLN_modulation(cond_flat)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            adaln_params.chunk(6, dim=-1)
        )
        x_flat_norm = modulate(self.norm1(x_flat), shift_msa, scale_msa)
        x_flat = x_flat + gate_msa * self.attn(x_flat_norm, pos=pos)
        x_flat = x_flat + gate_mlp * self.mlp(
            modulate(self.norm2(x_flat), shift_mlp, scale_mlp)
        )
        x = rearrange(x_flat, "(b t) n e -> b t n e", b=B, t=T_minus_1)
        return x


class AdaLNSpatioTransformer(nn.Module):

    def __init__(
        self,
        in_dim: int,
        model_dim: int,
        out_dim: int,
        num_blocks: int,
        num_heads: int,
        dropout: float = 0.0,
        latent_dim: int = 32,
        use_rope: bool = False,
    ) -> None:
        super(AdaLNSpatioTransformer, self).__init__()
        self.patch_up = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, model_dim)
        )
        self.pos_enc = PositionalEncoding(model_dim)
        self.use_rope = use_rope
        if use_rope:
            from .rope import RotaryPositionEmbedding2D, PositionGetter

            self.rope = RotaryPositionEmbedding2D()
            self.position_getter = PositionGetter()
        else:
            self.rope = None
            self.position_getter = None
        self.action_proj = nn.Linear(latent_dim, model_dim)
        self.input_adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(model_dim, 3 * model_dim, bias=True)
        )
        self.input_norm = nn.LayerNorm(model_dim, elementwise_affine=False, eps=1e-06)
        self.blocks = nn.ModuleList(
            [
                AdaLNSpatioBlock(model_dim, num_heads, dropout, rope=self.rope)
                for _ in range(num_blocks)
            ]
        )
        self.out = nn.Linear(model_dim, out_dim)

    def forward(self, x: Tensor, latent_z: Tensor, grid_hw: tuple = None) -> Tensor:
        B, T_minus_1, N, E = x.shape
        x = self.patch_up(x)
        x = self.pos_enc(x)
        pos = None
        if self.use_rope and grid_hw is not None:
            nH, nW = grid_hw
            pos = self.position_getter(B * T_minus_1, nH, nW, x.device)
        z_proj = self.action_proj(latent_z).unsqueeze(2)
        cond_expanded = z_proj.expand(-1, -1, N, -1)
        adaln_params = self.input_adaLN_modulation(cond_expanded)
        shift, scale, gate = adaln_params.chunk(3, dim=-1)
        x = gate * modulate(self.input_norm(x), shift, scale)
        for block in self.blocks:
            x = block(x, z_proj, pos=pos)
        x = self.out(x)
        return x
