# Portions of this file are derived from HuggingFace Transformers
# (https://github.com/huggingface/transformers), Copyright The HuggingFace Inc. team,
# licensed under the Apache License, Version 2.0. The license text is in LICENSE; the
# attribution is recorded in NOTICE.
# Upstream reference: src/transformers/models/qwen2/modeling_qwen2.py and models/qwen3_vl/modeling_qwen3_vl.py

"""Standard-backbone action expert (peer-aligned) for RynnBrain-VLA.

WHY THIS FILE EXISTS
--------------------
``expert_rynn_brain_vla.py`` hand-writes the expert decoder layer. Every peer that
uses the same two-stream / single-joint-softmax design instead instantiates a
*stock* transformer and reaches into its projections:

  * openpi pi0 / pi0.5 -- ``openpi/models/gemma.py:get_config("gemma_300m")``
        a second Gemma ``Config`` (width 1024, mlp 4096, depth 18) run in the same
        stack as the VLM's Gemma.
  * lingbot-vla-v2 -- ``modeling_lingbot_vla_v2.py:138,324``
        ``Qwen2ForCausalLM._from_config(...)`` then
        ``models = [self.qwenvl.model.language_model, self.qwen_expert.model]``
        (they fork HF's Qwen2 layer to add a ``compute_kqv=True`` split forward).
  * InternVLA-A1.5 -- ``modeling_internvla_a1_5.py:409,145,194-220``
        ``Qwen3_5TextModel(config=...)`` then
        ``models = [qwen3_5.language_model, action_expert]`` and an *external*
        reach-in: ``layer.self_attn.q_norm(layer.self_attn.q_proj(h).view(...))``,
        ``torch.cat`` -> one softmax -> ``layer.self_attn.o_proj(...)``.

Our control flow already matches InternVLA's (external ``compute_qkv`` /
``apply_output``); we just rebuilt the layer by hand instead of taking a tested
one. Consequences measured on shipped checkpoints:

  * 117.6M (2B) / 150.9M (4B) dead ``adaln`` parameters -- allocated, never called
    (``use_per_layer_adanorm=True`` routes ``compute_qkv`` through the
    ``isinstance(..., AdaRMSNorm)`` branch).
  * no per-head Q/K norm by default, while our VLM (Qwen3-VL) *does* have it ->
    the two sides of the joint softmax are asymmetric. pi0 is symmetric (neither
    Gemma side has Q/K norm); InternVLA is symmetric (both Qwen3.5 sides do);
    lingbot-vla-v2 has the SAME asymmetry we do (Qwen3-VL VLM w/ q_norm+k_norm in
    ``qwen3vl_in_vla.py:185-186`` vs Qwen2 expert without).

WHY ``Qwen3VLTextModel`` AND NOT Qwen2 / Qwen3.5
------------------------------------------------
The backbone *family* carries almost no information here: RoPE is applied
externally by the caller on the concatenated sequence (lingbot does the same --
``apply_mrope(query_states, key_states, position_ids)`` at
``modeling_lingbot_vla_v2.py:352``), and no peer loads pretrained expert weights
(all three are random-init). The only substantive family difference is whether
the layer ships ``q_norm``/``k_norm``. So we take the VLM's own family -- which is
what every peer effectively does -- giving symmetric Q/K normalization by
construction and a head layout that is guaranteed to match the joint softmax.

WHAT THIS FILE DOES *NOT* CHANGE
--------------------------------
``embed_time``, ``embed_action_positions`` and ``final`` are inherited verbatim
from ``ActionExpert`` so that a custom-vs-standard comparison isolates the layer
stack only. The expert's public surface used by ``modeling_rynn_brain_vla`` --
``layers[i].compute_qkv`` / ``layers[i].apply_output`` / ``final`` /
``embed_time`` / ``embed_action_positions`` / ``num_foresight_tokens`` /
``learnable_tokens`` / ``learnable_tokens_in_proj`` -- is identical, so no
forward path needs to know which backbone is in use.
"""

import torch
import torch.nn as nn

from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

from .expert_rynn_brain_vla import ActionExpert, AdaRMSNorm, ExpertRMSNorm


def build_stock_layers(family, hidden_size, intermediate_size, num_layers,
                       num_heads, num_kv_heads, head_dim, eps):
    """Instantiate a stock decoder stack and return ``(layers, has_qk_norm)``.

    The head layout (``head_dim`` / heads / kv-heads / depth) is NOT a free choice: the
    joint softmax concatenates the expert's Q/K/V with the VLM's along the head axis, so
    all three must match the VLM exactly. Every peer derives its expert the same way --
    only width and MLP are expert-specific:

      * openpi pi0/pi0.5 ``models/gemma.py:69`` -- ``gemma_300m`` IS ``gemma_2b`` with
        width 2048->1024 and mlp 16384->4096; depth 18, heads 8, kv 1, head_dim 256 are
        identical to their VLM.
      * InternVLA-A1.5 ``modeling_internvla_a1_5.py:388-405`` -- 18 consecutive
        ``action_expert_config_hf.X = vlm_text_config.X`` assignments (head_dim, heads,
        kv, depth, rope, rms_eps, layer_types); only hidden 1024 / intermediate 3072 are
        its own.
      * lingbot-vla-v2 ``modeling_lingbot_vla_v2.py:329`` -- asserts
        ``action_num_layers == num_layers``.

    ``family`` therefore selects only the *layer type*, which carries exactly one
    substantive bit: whether the layer ships per-head ``q_norm``/``k_norm``.

      "qwen3_vl" -- our VLM's own family. q_norm/k_norm present -> the joint softmax is
                    symmetric. This is not what any peer literally instantiates.
      "qwen2"    -- lingbot-vla-v2's actual expert (``modeling_lingbot_vla_v2.py:138``,
                    ``Qwen2ForCausalLM._from_config``). NO q_norm/k_norm, and q/k/v carry
                    a bias -> reproduces lingbot's joint-softmax ASYMMETRY against our
                    Qwen3-VL VLM, which DOES normalize (``qwen3vl_in_vla.py:185-186``).
                    That asymmetry is the entire point of the faithful arm.

    ``vocab_size`` is a placeholder: ``embed_tokens`` is discarded by the caller, as in
    lingbot (``del self.qwen_expert.model.embed_tokens`` :155) and InternVLA
    (``self.action_expert.embed_tokens = None`` :410).
    """
    common = dict(
        vocab_size=8,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        rms_norm_eps=eps,
        attention_dropout=0.0,
    )
    if family == "qwen3_vl":
        cfg = Qwen3VLTextConfig(head_dim=head_dim, **common)
        stock = Qwen3VLTextModel(cfg)
    elif family == "qwen2":
        cfg = Qwen2Config(**common)
        # Qwen2Config derives head_dim = hidden // heads, which is wrong for a NARROW
        # expert sharing the VLM's head layout, so it is overridden after construction;
        # the decoder layer reads config.head_dim at build time.
        cfg.head_dim = head_dim
        stock = Qwen2Model(cfg)
    else:
        raise ValueError(
            f"Unknown stock expert family {family!r}; expected 'qwen3_vl' or 'qwen2'."
        )
    layers = stock.layers
    del stock
    has_qk_norm = hasattr(layers[0].self_attn, "q_norm")
    return layers, has_qk_norm


def replace_stock_norms_with_adanorm(module, hidden_size, cond_dim):
    """AdaRMSNorm swap for *stock* HF layers.

    ``expert_rynn_brain_vla.replace_lnorm_with_adanorm`` keys off our own
    ``ExpertRMSNorm`` class and therefore cannot see ``Qwen3VLTextRMSNorm``.
    This variant matches any residual-stream RMSNorm structurally (a module with
    a 1-D ``weight`` of exactly ``hidden_size``), and skips the per-head Q/K
    norms both by name and by width (they are ``head_dim`` wide, not
    ``hidden_size``), mirroring lingbot's ``replace_lnorm_with_adanorm``
    (``modeling_lingbot_vla.py:274``).
    """
    for name, child in module.named_children():
        w = getattr(child, "weight", None)
        is_rmsnorm = (
            type(child).__name__.endswith("RMSNorm")
            and isinstance(w, nn.Parameter)
            and w.dim() == 1
            and w.numel() == hidden_size
        )
        if is_rmsnorm and name not in ("q_layernorm", "k_layernorm", "q_norm", "k_norm"):
            setattr(module, name, AdaRMSNorm(hidden_size, cond_dim))
        else:
            replace_stock_norms_with_adanorm(child, hidden_size, cond_dim)


class StdActionExpertLayer(nn.Module):
    """``compute_qkv`` / ``apply_output`` adapter over a stock Qwen3-VL decoder layer.

    Byte-for-byte the same computation as InternVLA-A1.5
    (``modeling_internvla_a1_5.py:194-205`` for Q/K/V, ``:317-319`` + MLP for the
    output half), and the same as our hand-written ``ActionExpertLayer`` except
    that ``q_norm``/``k_norm`` always exist and the dead ``adaln`` linear does not.

    Q/K/V are returned in ``(B, seq, heads, head_dim)`` -- the layout the caller
    concatenates with the VLM's, pre-RoPE -- matching ``ActionExpertLayer``.
    """

    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    @staticmethod
    def _norm(mod, hidden_states, cond):
        if isinstance(mod, AdaRMSNorm) and cond is not None:
            return mod(hidden_states, cond)
        return mod(hidden_states)

    def compute_qkv(self, hidden_states, cond=None):
        sa = self.layer.self_attn
        normed = self._norm(self.layer.input_layernorm, hidden_states, cond)
        b, s, _ = normed.shape
        hd = sa.head_dim
        q = sa.q_proj(normed).view(b, s, -1, hd)
        k = sa.k_proj(normed).view(b, s, -1, hd)
        v = sa.v_proj(normed).view(b, s, -1, hd)
        # q_norm/k_norm exist on Qwen3-family layers and NOT on Qwen2/Gemma ones. Keeping
        # them optional is what makes a faithful lingbot arm possible: their Qwen2 expert
        # feeds RAW projections into a softmax shared with a Qwen3-VL prefix that IS
        # normalized (qwen3vl_in_vla.py:185-186). pi0 is symmetric-without (neither Gemma
        # side has them), InternVLA symmetric-with (both Qwen3.5 sides do).
        if getattr(sa, "q_norm", None) is not None:
            q = sa.q_norm(q)
            k = sa.k_norm(k)
        return q, k, v

    def apply_output(self, hidden_states, attn_out, cond=None):
        sa = self.layer.self_attn
        b, s = attn_out.shape[:2]
        hidden_states = hidden_states + sa.o_proj(attn_out.reshape(b, s, -1))
        residual = hidden_states
        normed = self._norm(self.layer.post_attention_layernorm, hidden_states, cond)
        return residual + self.layer.mlp(normed)


class StdActionExpert(ActionExpert):
    """``ActionExpert`` with the layer stack replaced by stock HF decoder layers.

    ``family`` selects which peer's expert is reproduced: "qwen3_vl" (our VLM's own
    family, symmetric joint softmax) or "qwen2" (lingbot-vla-v2's actual expert,
    asymmetric). See ``build_stock_layers``.

    Subclasses ``ActionExpert`` purely to inherit ``embed_time`` /
    ``embed_action_positions`` / ``final`` unchanged; ``ActionExpert.__init__`` is
    deliberately bypassed because it would build the hand-written stack.
    """

    def __init__(self, hidden_size, intermediate_size, num_layers, num_heads,
                 num_kv_heads, head_dim, eps=1e-6, use_adaln=True, final_adaln=True,
                 use_per_layer_adanorm=False, num_foresight_tokens=0,
                 foresight_visibility="first", time_concat=False, family="qwen3_vl"):
        nn.Module.__init__(self)
        # AdaLN must be all-or-nothing. Two failure modes are guarded here:
        #  * use_adaln and not per-layer  -> the hand-written stack would fall back to the
        #    FiLM `adaln` linear, which is dead code and deliberately not reproduced.
        #  * per-layer and not use_adaln  -> the norms would be AdaRMSNorm but the forward
        #    passes cond=None (time_conditioning != "adaln"), i.e. a TypeError at runtime.
        #    This is the configuration an InternVLA-style concat-only arm would hit, so it
        #    must fail loudly at construction rather than mid-training.
        if bool(use_adaln) != bool(use_per_layer_adanorm):
            raise ValueError(
                "AdaLN is all-or-nothing for expert_backbone='qwen3_vl': "
                f"time_conditioning=='adaln' -> {bool(use_adaln)} but "
                f"expert_per_layer_adanorm=={bool(use_per_layer_adanorm)}. "
                "Use time_conditioning='adaln' + expert_per_layer_adanorm=True (pi0.5 / "
                "lingbot), or a non-adaln time_conditioning + expert_per_layer_adanorm=False "
                "+ expert_time_concat=True (pi0 / InternVLA)."
            )
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_adaln = use_adaln
        self.final_adaln = final_adaln
        self.num_foresight_tokens = num_foresight_tokens
        self.foresight_visibility = foresight_visibility
        self.use_per_layer_adanorm = use_per_layer_adanorm
        # Whether the joint softmax is symmetric is decided by the layer FAMILY, not by a
        # flag: Qwen3-VL ships q_norm/k_norm (symmetric, the pi0 / InternVLA property),
        # Qwen2 does not (asymmetric, which is what lingbot-vla-v2 actually runs). Set
        # below from the instantiated stack rather than assumed.
        self.family = family
        self.time_concat = time_concat

        # Time MLP: identical module to ActionExpert so ``embed_time`` is inherited
        # verbatim. Allocated ONLY when AdaLN is on -- ``embed_time`` is called from
        # exactly one place (`cond = expert.embed_time(times) if time_conditioning ==
        # "adaln" else None`), so under a concat-only arm it would be 4 untrained
        # tensors with grad=None, i.e. the same dead-parameter bug this file exists to
        # remove. The concat path uses the RAW sinusoid instead, which is what all three
        # peers do (pi0.py:170 posemb_sincos -> concat; lingbot :1136 time_emb_ori;
        # InternVLA :564).
        if use_adaln:
            self.time_mlp = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
            )

        # Stock narrow Qwen3-VL text stack. vocab_size is set to a placeholder because
        # ``embed_tokens`` is dropped immediately (both peers do the same:
        # lingbot ``del self.qwen_expert.model.embed_tokens`` :155,
        # InternVLA ``self.action_expert.embed_tokens = None`` :410).
        # Keep ONLY the decoder layers: embed_tokens is unused (no vocab), rotary_emb is
        # unused (the caller applies a shared M-RoPE over [prefix|suffix], exactly as in
        # lingbot ``apply_mrope(...)`` :354), and ``stock.norm`` is replaced below so the
        # final-norm semantics stay identical to the hand-written expert.
        layers, has_qk_norm = build_stock_layers(
            family=family,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_layers=num_layers,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            eps=eps,
        )
        self.use_qk_norm = has_qk_norm

        self.layers = nn.ModuleList([StdActionExpertLayer(l) for l in layers])
        self.norm = ExpertRMSNorm(hidden_size, eps)

        # Foresight tokens: identical construction to ActionExpert (InternVLA
        # ``modeling_internvla_a1_5.py:568-573``).
        if num_foresight_tokens > 0:
            self.learnable_tokens = nn.Parameter(
                torch.zeros(num_foresight_tokens, hidden_size)
            )
            nn.init.trunc_normal_(self.learnable_tokens, std=0.02)
            self.learnable_tokens_in_proj = nn.Linear(hidden_size, hidden_size)

        # pi0-style concat time conditioning (``action_time_mlp_in/out``):
        #   pi0            openpi/models/pi0.py:172-178   concat only
        #   lingbot-vla-v2 modeling_lingbot_vla_v2.py:475 concat AND per-layer AdaRMSNorm
        #                  (deployed with adanorm_time: true in both shipped yamls)
        #   InternVLA-A1.5 modeling_internvla_a1_5.py:564 concat only
        #   pi0.5 / us     AdaRMSNorm only
        if time_concat:
            self.action_time_mlp_in = nn.Linear(2 * hidden_size, hidden_size)
            self.action_time_mlp_out = nn.Linear(hidden_size, hidden_size)

        if use_per_layer_adanorm:
            replace_stock_norms_with_adanorm(self.layers, hidden_size, hidden_size)
            if final_adaln:
                self.norm = AdaRMSNorm(hidden_size, hidden_size, eps)

    def maybe_time_concat(self, action_hidden, times):
        """pi0's ``embed_suffix`` time fusion: MLP([action_emb ; time_emb]) + swish.

        No-op unless ``time_concat`` is on, so the AdaLN-only (pi0.5) arms are bit-identical.
        """
        if not self.time_concat:
            return action_hidden
        time_emb = self.embed_time_raw(times).to(action_hidden.dtype)   # (B, H)
        time_emb = time_emb[:, None, :].expand(-1, action_hidden.size(1), -1)
        x = torch.cat([action_hidden, time_emb], dim=-1)
        x = nn.functional.silu(self.action_time_mlp_in(x))
        return self.action_time_mlp_out(x)

    def embed_time_raw(self, times):
        """Sinusoidal timestep embedding *before* the time MLP (pi0 concatenates the raw one)."""
        from .expert_rynn_brain_vla import _create_sinusoidal_pos_embedding
        return _create_sinusoidal_pos_embedding(
            times, self.hidden_size, device=times.device
        ).to(self.action_time_mlp_in.weight.dtype)
