"""Independent action expert (lingbot-vla-v2 / pi0 style) for RynnBrain-VLA.

A narrow Qwen2-style transformer stack that is run in two-stream lockstep with the
VLM: at every layer the expert computes its own Q/K/V (from a narrow residual
stream) which are concatenated with the VLM's Q/K/V for a single joint attention,
then each stream applies its own output projection + MLP.

The expert MUST share the VLM's head layout so the per-layer Q/K/V concat is valid:
  num_attention_heads, num_key_value_heads, head_dim, num_hidden_layers == VLM.
Only the residual width (``hidden_size``) and the MLP ``intermediate_size`` are
expert-specific (narrow), which is what makes the denoising decode cheaper.

Time conditioning uses DiT-style AdaLN (FiLM) driven by a sinusoidal timestep
embedding, applied at every layer's input/post norms and (optionally) the final
norm. AdaLN projections are zero-initialized so the expert starts as a clean
(identity-modulation) residual network.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class AdaRMSNorm(nn.Module):
    """AdaRMSNorm: RMSNorm + FiLM (conditional normalization with scale and shift).

    DiT-style initialization: gamma/beta weights and biases are 0 at init, so FiLM is identity.
    """

    def __init__(self, hidden_size, cond_dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.gamma = nn.Linear(cond_dim, hidden_size)
        self.beta = nn.Linear(cond_dim, hidden_size)

        # DiT style init: gamma.weight=0, gamma.bias=0; beta.weight=0, beta.bias=0
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)
        # The zeros above are NOT enough on the training path: the model is built on the meta
        # device (models/__init__.py:_init_empty_params), where every nn.init.* here is a
        # silent no-op, and the real values are drawn later by _init_missing_weights ->
        # _init_weights(<owning module>), which dispatches on module TYPE and gives any
        # nn.Linear a normal_(std=initializer_range) draw. Marking the modules is the only
        # way the zero survives that path (same trap as view_role_emb / slot_seed_proj).
        # Note AdaRMSNorm.weight itself is safe without a marker: HF's _init_weights has a
        # name-based branch that fills any "*RMSNorm" module's weight with 1.0.
        self.gamma._is_zero_init = True
        self.beta._is_zero_init = True

    def forward(self, hidden_states, cond):
        """
        Args:
            hidden_states: (B, S, H)
            cond: (B, cond_dim) or (B, 1, cond_dim)
        """
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        hidden_states = self.weight * hidden_states
        # Handle both (B, cond_dim) and (B, 1, cond_dim)
        if cond.dim() == 2:
            cond = cond.unsqueeze(1)  # (B, 1, cond_dim)
        gamma = self.gamma(cond)  # (B, 1, H)
        beta = self.beta(cond)    # (B, 1, H)
        hidden_states = (1 + gamma.to(torch.float32)) * hidden_states + beta.to(torch.float32)
        return hidden_states.to(input_dtype)


def replace_lnorm_with_adanorm(module, hidden_size, cond_dim):
    """Recursively replace all ExpertRMSNorm with AdaRMSNorm (except q/k norms)."""
    for name, child in module.named_children():
        if isinstance(child, ExpertRMSNorm):
            # q_norm/k_norm are per-head (head_dim wide) Q/K scale norms, not residual
            # stream norms: they must stay plain RMSNorm (and are not hidden_size wide).
            if name not in ("q_layernorm", "k_layernorm", "q_norm", "k_norm"):
                setattr(module, name, AdaRMSNorm(hidden_size, cond_dim))
        else:
            replace_lnorm_with_adanorm(child, hidden_size, cond_dim)


class ExpertMLP(nn.Module):
    """SwiGLU MLP (Qwen2/LLaMA style)."""

    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def _film(x, shift, scale):
    """DiT FiLM modulation: x * (1 + scale) + shift. shift/scale are (B, H)."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class ActionExpertLayer(nn.Module):
    """One expert decoder layer with a split forward (compute_qkv / apply_output).

    Q/K/V are returned in (B, seq, heads, head_dim) layout (pre-RoPE) so the caller
    can concatenate them with the VLM's Q/K/V and apply a shared RoPE + attention.
    """

    def __init__(self, hidden_size, intermediate_size, num_heads, num_kv_heads,
                 head_dim, eps=1e-6, use_adaln=True, use_qk_norm=False):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.use_adaln = use_adaln

        self.input_layernorm = ExpertRMSNorm(hidden_size, eps)
        self.post_attention_layernorm = ExpertRMSNorm(hidden_size, eps)
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        # Per-head Q/K RMSNorm matching Qwen3-VL's q_norm/k_norm, so both streams of the
        # joint attention feed same-scale Q/K into one softmax. None (default) keeps the
        # legacy unnormalized forward for checkpoints trained without it.
        if use_qk_norm:
            self.q_norm = ExpertRMSNorm(head_dim, eps)
            self.k_norm = ExpertRMSNorm(head_dim, eps)
        else:
            self.q_norm = None
            self.k_norm = None
        self.mlp = ExpertMLP(hidden_size, intermediate_size)
        if use_adaln:
            # cond -> (shift_in, scale_in, shift_post, scale_post)
            self.adaln = nn.Linear(hidden_size, 4 * hidden_size)

    def compute_qkv(self, hidden_states, cond=None):
        # Per-layer AdaRMSNorm: pass cond directly to the norm
        if isinstance(self.input_layernorm, AdaRMSNorm) and cond is not None:
            normed = self.input_layernorm(hidden_states, cond)
        else:
            # Fallback: ExpertRMSNorm + FiLM
            normed = self.input_layernorm(hidden_states)
            if self.use_adaln and cond is not None:
                shift_in, scale_in, _, _ = self.adaln(cond).chunk(4, dim=-1)
                normed = _film(normed, shift_in, scale_in)
        b, s, _ = normed.shape
        q = self.q_proj(normed).view(b, s, self.num_heads, self.head_dim)
        k = self.k_proj(normed).view(b, s, self.num_kv_heads, self.head_dim)
        v = self.v_proj(normed).view(b, s, self.num_kv_heads, self.head_dim)
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        return q, k, v

    def apply_output(self, hidden_states, attn_out, cond=None):
        # attn_out: (B, S, num_heads, head_dim)
        b, s = attn_out.shape[:2]
        hidden_states = hidden_states + self.o_proj(attn_out.reshape(b, s, -1))
        residual = hidden_states
        # Per-layer AdaRMSNorm: pass cond directly to the norm
        if isinstance(self.post_attention_layernorm, AdaRMSNorm) and cond is not None:
            normed = self.post_attention_layernorm(hidden_states, cond)
        else:
            # Fallback: ExpertRMSNorm + FiLM
            normed = self.post_attention_layernorm(hidden_states)
            if self.use_adaln and cond is not None:
                _, _, shift_post, scale_post = self.adaln(cond).chunk(4, dim=-1)
                normed = _film(normed, shift_post, scale_post)
        return residual + self.mlp(normed)


def _create_sinusoidal_pos_embedding(time, dimension, min_period=0.004, max_period=4.0, device="cpu"):
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    import math
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=torch.float32, device=device)
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


class ActionExpert(nn.Module):
    """Narrow expert stack: time MLP + N split-forward layers + final (AdaLN) norm."""

    def __init__(self, hidden_size, intermediate_size, num_layers, num_heads,
                 num_kv_heads, head_dim, eps=1e-6, use_adaln=True, final_adaln=True,
                 use_per_layer_adanorm=False, num_foresight_tokens=0, use_qk_norm=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_adaln = use_adaln
        self.final_adaln = final_adaln
        self.num_foresight_tokens = num_foresight_tokens
        self.use_per_layer_adanorm = use_per_layer_adanorm
        self.use_qk_norm = use_qk_norm

        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.layers = nn.ModuleList([
            ActionExpertLayer(hidden_size, intermediate_size, num_heads, num_kv_heads,
                              head_dim, eps=eps, use_adaln=use_adaln,
                              use_qk_norm=use_qk_norm)
            for _ in range(num_layers)
        ])
        self.norm = ExpertRMSNorm(hidden_size, eps)
        if use_adaln and final_adaln:
            self.adaln_final = nn.Linear(hidden_size, 2 * hidden_size)

        # Foresight tokens: learnable tokens that predict future dynamics (InternVLA style)
        if num_foresight_tokens > 0:
            self.learnable_tokens = nn.Parameter(
                torch.zeros(num_foresight_tokens, hidden_size)
            )
            nn.init.trunc_normal_(self.learnable_tokens, std=0.02)
            self.learnable_tokens_in_proj = nn.Linear(hidden_size, hidden_size)

        # Per-layer AdaRMSNorm: replace ExpertRMSNorm with AdaRMSNorm in each layer
        if use_per_layer_adanorm:
            replace_lnorm_with_adanorm(self.layers, hidden_size, hidden_size)
            if final_adaln:
                self.norm = AdaRMSNorm(hidden_size, hidden_size, eps)

    def embed_time(self, times):
        time_emb = _create_sinusoidal_pos_embedding(
            times, self.hidden_size, device=times.device
        ).to(self.time_mlp[0].weight.dtype)
        return self.time_mlp(time_emb)  # (B, hidden)

    def embed_action_positions(self, num_actions, device, dtype):
        """Generate sinusoidal position embeddings for action tokens.

        Args:
            num_actions: Number of action tokens (A).
            device: Target device.
            dtype: Target dtype.

        Returns:
            pos_emb: (1, A, hidden_size) sinusoidal position embeddings.
        """
        # Create position indices: [0, 1, 2, ..., A-1]
        position = torch.arange(num_actions, dtype=torch.float32, device=device).unsqueeze(1)

        # Compute sinusoidal position encoding
        div_term = torch.exp(
            torch.arange(0, self.hidden_size, 2, dtype=torch.float32, device=device)
            * (-math.log(10000.0) / self.hidden_size)
        )
        pe = torch.zeros(num_actions, self.hidden_size, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe.unsqueeze(0).to(dtype)  # (1, A, hidden_size)

    def final(self, hidden_states, cond=None):
        # Per-layer AdaRMSNorm: pass cond directly to the norm
        if isinstance(self.norm, AdaRMSNorm) and cond is not None:
            hidden_states = self.norm(hidden_states, cond)
        else:
            # Fallback: ExpertRMSNorm + FiLM
            hidden_states = self.norm(hidden_states)
            if self.use_adaln and self.final_adaln and cond is not None:
                shift, scale = self.adaln_final(cond).chunk(2, dim=-1)
                hidden_states = _film(hidden_states, shift, scale)
        return hidden_states
