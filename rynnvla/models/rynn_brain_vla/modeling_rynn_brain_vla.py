# Portions of this file are derived from HuggingFace Transformers
# (https://github.com/huggingface/transformers), Copyright The HuggingFace Inc. team,
# licensed under the Apache License, Version 2.0. The license text is in LICENSE; the
# attribution is recorded in NOTICE.
# Upstream reference: src/transformers/models/qwen3_vl/modeling_qwen3_vl.py

import math
from dataclasses import dataclass
from typing import Any, Optional, Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.processing_utils import Unpack
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLVisionModel,
    Qwen3VLTextModel,
    Qwen3VLModel,
    Qwen3VLModelOutputWithPast,
)
from transformers.utils import can_return_tuple
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.utils.generic import TransformersKwargs
from transformers.models.qwen2.modeling_qwen2 import (
    apply_rotary_pos_emb,
    eager_attention_forward,
    repeat_kv,
)

from ...constants import NUM_VIEW_SLOTS
from .configuration_rynn_brain_vla import RynnBrainVLAConfig
from .expert_rynn_brain_vla import ActionExpert, AdaRMSNorm
from .depth_head_rynn_brain_vla import DepthReadoutHead
from .sf_align_rynn_brain_vla import DA3MonoTeacher, SFAlignProjector, compute_sf_align_loss
from .lb_align_rynn_brain_vla import LBQueryAlignHead, compute_lb_align_loss


@dataclass
class RynnBrainVLAModelOutputWithPast(Qwen3VLModelOutputWithPast):
    actions: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None


def _global_last_position(
    position_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if attention_mask is not None:
        position_ids = position_ids.masked_fill(~attention_mask.to(torch.bool).unsqueeze(0), 0)
    last = position_ids.amax(dim=(0, 2)).view(1, position_ids.size(1), 1)
    return last.expand(position_ids.size(0), -1, -1)


class RynnBrainVLACache:
    """KV cache that stores the prefix once and concatenates with new KV on every later call.

    The first `update` for each layer captures the prefix key/value tensors. Later calls return
    `cat([prefix, new], dim=-2)` without ever mutating the stored prefix, so gradients through
    the freshly computed key/value tensors are preserved (required by the RTC decode path).
    """

    is_compileable = False
    is_sliding: list[bool] = []

    def __init__(self, config=None, max_cache_len=None):
        self.prefix_keys: list[Optional[torch.Tensor]] = []
        self.prefix_values: list[Optional[torch.Tensor]] = []
        self.prefix_last_position: Optional[torch.Tensor] = None
        self.prefix_valid_mask: Optional[torch.Tensor] = None
        self._frozen_keys: Optional[list[Optional[torch.Tensor]]] = None
        self._frozen_values: Optional[list[Optional[torch.Tensor]]] = None
        self._frozen_prefix_valid_mask: Optional[torch.Tensor] = None

    def _ensure_layer(self, layer_idx: int) -> None:
        while len(self.prefix_keys) <= layer_idx:
            self.prefix_keys.append(None)
            self.prefix_values.append(None)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._ensure_layer(layer_idx)

        prefix_k = self.prefix_keys[layer_idx]
        if prefix_k is None:
            self.prefix_keys[layer_idx] = key_states.detach()
            self.prefix_values[layer_idx] = value_states.detach()
            return key_states, value_states

        keys = torch.cat([prefix_k, key_states], dim=-2)
        values = torch.cat([self.prefix_values[layer_idx], value_states], dim=-2)
        return keys, values

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self.prefix_keys) or self.prefix_keys[layer_idx] is None:
            return 0
        return self.prefix_keys[layer_idx].shape[-2]

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int) -> tuple[int, int]:
        prefix_len = self.get_seq_length(layer_idx)
        kv_length = prefix_len + cache_position.shape[0]
        return kv_length, 0

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        return -1

    def freeze(self):
        """Snapshot prefix KV and replace inference tensors with regular clones."""
        if self._frozen_keys is not None:
            return
        self._frozen_keys = self.prefix_keys
        self._frozen_values = self.prefix_values
        self._frozen_prefix_valid_mask = self.prefix_valid_mask
        self.prefix_keys = [k.clone() if k is not None else None for k in self._frozen_keys]
        self.prefix_values = [v.clone() if v is not None else None for v in self._frozen_values]
        if self.prefix_valid_mask is not None:
            self.prefix_valid_mask = self.prefix_valid_mask.clone()

    def unfreeze(self):
        """Restore prefix state from the snapshot taken by freeze()."""
        if self._frozen_keys is None:
            return
        self.prefix_keys = self._frozen_keys
        self.prefix_values = self._frozen_values
        self.prefix_valid_mask = self._frozen_prefix_valid_mask
        self._frozen_keys = None
        self._frozen_values = None
        self._frozen_prefix_valid_mask = None


def _attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    num_action_tokens: int,
    past_key_values: Optional[RynnBrainVLACache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        if past_key_values.get_seq_length(self.layer_idx) == 0:
            # prefilling
            prefix_query_states = query_states
            prefix_key_states = key_states
            prefix_value_states = value_states
            prefix_attention_mask = attention_mask
            action_query_states = None
        else:
            # decoding
            prefix_query_states = None
            action_query_states = query_states
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
    else:
        # training
        assert query_states.size(2) == key_states.size(2)

        num_vlm_tokens = hidden_states.size(1) - num_action_tokens
        prefix_query_states = query_states[:, :, :num_vlm_tokens]
        prefix_key_states = key_states[:, :, :num_vlm_tokens]
        prefix_value_states = value_states[:, :, :num_vlm_tokens]

        if attention_mask is None:
            prefix_attention_mask = None
        elif attention_mask.ndim == 4:
            prefix_attention_mask = attention_mask[:, :, :num_vlm_tokens, :num_vlm_tokens]
        else:
            prefix_attention_mask = attention_mask[:, :num_vlm_tokens]

        if num_action_tokens == 0:
            action_query_states = None
        else:
            action_query_states = query_states[:, :, num_vlm_tokens:]

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    kwargs.pop("position_ids", None)
    prefix_kwargs = dict(kwargs)
    prefix_kwargs.pop("cu_seq_lens_action", None)
    prefix_kwargs.pop("max_length_action", None)

    attn_output = None
    if prefix_query_states is not None:
        attn_output, _ = attention_interface(
            self,
            prefix_query_states,
            prefix_key_states,
            prefix_value_states,
            prefix_attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=None,
            is_causal=True,
            **prefix_kwargs,
        )

    if action_query_states is not None:
        # Knowledge insulation: detach the prefix (VLM) KV that action queries attend
        # to so the action loss does not backprop into the VLM. Only needed on the
        # training path (no cache); the decode cache already stores the prefix detached.
        ki_key_states = key_states
        ki_value_states = value_states
        if (
            past_key_values is None
            and getattr(self.config, "knowledge_insulation", False)
            and 0 < num_action_tokens < key_states.size(2)
        ):
            prefix_len = key_states.size(2) - num_action_tokens
            ki_key_states = torch.cat(
                [key_states[:, :, :prefix_len].detach(), key_states[:, :, prefix_len:]], dim=2
            )
            ki_value_states = torch.cat(
                [value_states[:, :, :prefix_len].detach(), value_states[:, :, prefix_len:]], dim=2
            )

        action_key_states = ki_key_states
        action_value_states = ki_value_states
        action_kwargs = dict(kwargs)
        cu_seq_lens_action = action_kwargs.pop("cu_seq_lens_action", None)
        max_length_action = action_kwargs.pop("max_length_action", None)
        if cu_seq_lens_action is not None and max_length_action is not None:
            cu_seq_lens_prefix = action_kwargs.get("cu_seq_lens_q")
            if cu_seq_lens_prefix is not None:
                prefix_lengths = cu_seq_lens_prefix[1:] - cu_seq_lens_prefix[:-1]
                action_lengths = cu_seq_lens_action[1:] - cu_seq_lens_action[:-1]
                full_lengths = prefix_lengths + action_lengths
                action_kwargs["cu_seq_lens_q"] = cu_seq_lens_action
                action_kwargs["max_length_q"] = max_length_action
                action_kwargs["cu_seq_lens_k"] = torch.cat(
                    [full_lengths.new_zeros(1), torch.cumsum(full_lengths, dim=0)]
                ).to(cu_seq_lens_prefix.dtype)
                action_kwargs["max_length_k"] = int(full_lengths.max().item())

                if full_lengths.numel() > 1:
                    prefix_total = int(cu_seq_lens_prefix[-1].item())
                    key_chunks = []
                    value_chunks = []
                    for i in range(full_lengths.numel()):
                        prefix_start = int(cu_seq_lens_prefix[i].item())
                        prefix_end = int(cu_seq_lens_prefix[i + 1].item())
                        action_start = prefix_total + int(cu_seq_lens_action[i].item())
                        action_end = prefix_total + int(cu_seq_lens_action[i + 1].item())
                        key_chunks.append(ki_key_states[:, :, prefix_start:prefix_end])
                        key_chunks.append(ki_key_states[:, :, action_start:action_end])
                        value_chunks.append(ki_value_states[:, :, prefix_start:prefix_end])
                        value_chunks.append(ki_value_states[:, :, action_start:action_end])
                    action_key_states = torch.cat(key_chunks, dim=2)
                    action_value_states = torch.cat(value_chunks, dim=2)

        if attention_mask is None:
            action_attention_mask = None
        elif attention_mask.ndim == 4:
            if past_key_values is not None:
                # Decoding: queries are already action-only; the 4D mask is
                # expected to be (B, 1, num_action_tokens, kv_len) as-is.
                action_attention_mask = attention_mask
            else:
                # 4D additive mask (B, 1, q_total, kv_total): slice action query rows
                # and zero out action-to-action region to allow bidirectional attention
                action_attention_mask = attention_mask[:, :, num_vlm_tokens:, :]
                action_attention_mask = action_attention_mask.clone()
                action_attention_mask[:, :, :, num_vlm_tokens:] = 0
        else:
            action_attention_mask = attention_mask

        action_attn_output, _ = attention_interface(
            self,
            action_query_states,
            action_key_states,
            action_value_states,
            action_attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=None,
            is_causal=False,
            **action_kwargs,
        )

        if attn_output is not None:
            attn_output = torch.cat([attn_output, action_attn_output], dim=1)
        else:
            attn_output = action_attn_output

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, _


def _create_sinusoidal_pos_embedding(
    time: torch.tensor,
    dimension: int,
    min_period: float = 0.004,
    max_period: float = 4.0,
    device: str = "cpu",
):
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=torch.float32, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb


class RynnBrainVLAModel(Qwen3VLModel):
    config: RynnBrainVLAConfig
    model_type = "rynn_brain_vla"
    _default_key_mapping = {
        "model.visual": "visual",
        "model.language_model": "language_model",
    }

    def __init__(self, config: RynnBrainVLAConfig):
        super(Qwen3VLModel, self).__init__(config)
        self.visual = Qwen3VLVisionModel._from_config(config.vision_config)
        self.language_model = Qwen3VLTextModel._from_config(config.text_config)
        self.rope_deltas = None  # cache rope_deltas here

        self.action_head_type = getattr(config, "action_head_type", "shared")
        self.time_conditioning = getattr(config, "time_conditioning", "adaln")

        _vlm_hidden = config.text_config.hidden_size
        # Action-stream residual width: VLM width for the shared head, or the narrow
        # expert width for the independent-expert head.
        if self.action_head_type == "expert":
            _act_hidden = config.expert_hidden_size
        else:
            _act_hidden = _vlm_hidden

        # State token lives in the VLM prefix stream, so state_proj is always VLM-width.
        self.state_proj = nn.Linear(config.action_dim, _vlm_hidden)
        # Action in/out projections live at the action-stream width.
        self.action_in_proj = nn.Linear(config.action_dim, _act_hidden)
        # action_out_proj stays unconditional even though the Stage-2 recipe never reaches it
        # (use_latent_head_readout routes _readout_action through latent_readout_proj instead):
        # it is only action_dim x expert_hidden parameters, and dropping it would add it to the
        # Stage-1-only key set and break the invariant that Stage-2 differs from Stage-1 solely
        # by the latent_in_proj -> latent_readout_proj swap. It is also what
        # expert_action_step_multiview reads through on a Stage-1 checkpoint.
        self.action_out_proj = nn.Linear(_act_hidden, config.action_dim)

        # B2 camera identity (prefix side). One row per constants.VIEW_ROLES role, added to
        # that camera's image tokens in _embed_prefix. Constructed only when enabled so
        # checkpoints trained without it keep exactly the same state_dict keys. Zero-init
        # (re-zeroed after post_init below) => enabling it is a bit-identical no-op at step 0.
        # Arm V4: per-camera context tokens in the expert suffix. Reuses
        # slot_seed_norm / slot_seed_proj (built below under action_head_type=="expert"),
        # which is exactly what makes stage-1's camera knowledge transferable: the same
        # matrix does the same job (pooled camera view -> one expert token) in both stages.
        self.use_view_cond_slots = getattr(config, "use_view_cond_slots", False)
        self.view_role_emb = (
            nn.Embedding(NUM_VIEW_SLOTS, _vlm_hidden)
            if getattr(config, "use_view_role_embedding", False)
            else None
        )
        if self.view_role_emb is not None:
            # Marker consumed by _init_weights. Zeroing in __init__ is NOT enough: the
            # training path builds on the meta device (models/__init__.py:_init_empty_params),
            # where every nn.init.* here is a silent no-op, and the real values are drawn
            # later by _init_missing_weights -> _init_weights(<owning module>), which
            # dispatches on module TYPE and gives any nn.Embedding a normal_ draw. Marking
            # the module is the only way the zero survives that path. (Same trap already
            # documented for action_expert.learnable_tokens below; slot_seed_proj used to
            # fall into it too -- its "starts at zero" comment was violated in every
            # checkpoint shipped before 2026-08-21, e.g. xl_lingbot has
            # slot_seed_proj.weight absmax 1.0e-1. Now marked as well.)
            self.view_role_emb._is_zero_init = True

        # Hierarchical VLA: latent action head for pretrain (robot-agnostic, from LAM).
        # When use_latent_actions=True, the model predicts latent_action_dim actions
        # instead of robot-specific action_dim actions.
        self.use_latent_actions = getattr(config, "use_latent_actions", False)
        self.latent_action_dim = getattr(config, "latent_action_dim", 256)
        # fix-v7 (2026-08-23): keep the stage-1-pretrained latent_action_head as the
        # DIRECT-path readout trunk (hidden -> hidden -> SiLU -> 256) and finish with a
        # fresh thin latent_readout_proj (256 -> action_dim). Per-token FM target is
        # action_dim (81) < 256, so the readout-bottleneck bound that killed the grouped
        # 5-step decoders (405 > 256) does not apply here. Default False => the module
        # graph and every forward stay byte-identical to before this change.
        self.use_latent_head_readout = getattr(config, "use_latent_head_readout", False)
        bypass_latent_output = getattr(config, "bypass_latent_output_projection", False)
        if bypass_latent_output and (self.use_latent_actions or not self.use_latent_head_readout):
            raise ValueError(
                "bypass_latent_output_projection requires a Stage-2 latent-head readout "
                "(use_latent_head_readout=True, use_latent_actions=False)."
            )
        if self.use_latent_actions and self.use_latent_head_readout:
            raise ValueError(
                "use_latent_head_readout reuses latent_action_head as the direct action "
                "readout trunk and is mutually exclusive with use_latent_actions."
            )
        if self.use_latent_actions or self.use_latent_head_readout:
            latent_head_layers = [
                nn.Linear(_act_hidden, _act_hidden),
                nn.SiLU(),
            ]
            if not bypass_latent_output:
                latent_head_layers.append(nn.Linear(_act_hidden, self.latent_action_dim))
            # Preserve latent_action_head.0 keys so the first layer transfers
            # exactly. When the output layer is bypassed there are no
            # latent_action_head.2 keys: the Stage-1 output layer is deliberately
            # excluded from the model.
            self.latent_action_head = nn.Sequential(*latent_head_layers)
            self.latent_in_proj = (
                nn.Linear(self.latent_action_dim, _act_hidden)
                if self.use_latent_actions
                else None
            )
        else:
            self.latent_action_head = None
            self.latent_in_proj = None
        # fix-v7 final projection; None unless use_latent_head_readout.
        self.latent_readout_proj = (
            nn.Linear(_act_hidden if bypass_latent_output else self.latent_action_dim, config.action_dim)
            if self.use_latent_head_readout
            else None
        )
        # Shared-head time conditioning (VLM width). get_action_features -- the only caller of
        # all four -- is reached solely on the shared-backbone head; the expert head uses the
        # action expert's own embed_time and AdaRMSNorm. Both formal recipes use
        # action_head_type="expert", so building these unconditionally left ~9 * vlm_hidden^2
        # dead parameters holding optimizer and EMA slots. Build only what the selected head's
        # time-conditioning branch actually calls.
        _shared_head = self.action_head_type != "expert"
        _shared_adaln = _shared_head and self.time_conditioning == "adaln"
        # Legacy concat-then-MLP time conditioning (shared path only, VLM width).
        self.action_time_proj = (
            nn.Sequential(
                nn.Linear(_vlm_hidden * 2, _vlm_hidden),
                nn.SiLU(),
                nn.Linear(_vlm_hidden, _vlm_hidden),
            )
            if _shared_head and not _shared_adaln
            else None
        )
        self.time_mlp = (
            nn.Sequential(
                nn.Linear(_vlm_hidden, _vlm_hidden),
                nn.SiLU(),
                nn.Linear(_vlm_hidden, _vlm_hidden),
            )
            if _shared_adaln
            else None
        )
        self.adaln_in = nn.Linear(_vlm_hidden, 2 * _vlm_hidden) if _shared_adaln else None
        self.adaln_final = nn.Linear(_vlm_hidden, 2 * _vlm_hidden) if _shared_adaln else None
        # Flag for zero-init in _init_weights (identity modulation at start).
        if self.adaln_in is not None:
            self.adaln_in._is_adaln = True
        if self.adaln_final is not None:
            self.adaln_final._is_adaln = True

        # Independent action expert (two-stream lockstep with the VLM). The head layout
        # is forced to match the VLM so the per-layer Q/K/V concat is valid; only the
        # residual width and MLP size are expert-specific.
        self.action_expert = None
        if self.action_head_type == "expert":
            tc = config.text_config
            expert_num_heads = getattr(config, "expert_num_attention_heads", None) or tc.num_attention_heads
            if expert_num_heads != tc.num_attention_heads:
                raise ValueError(
                    "expert_num_attention_heads must equal the VLM text num_attention_heads "
                    f"({tc.num_attention_heads}) for the joint Q/K/V concat; got {expert_num_heads}."
                )
            # NOTE: use_per_layer_adanorm / num_foresight_tokens used to be hard-coded
            # here (True / 50). They are now config fields whose DEFAULTS are those same
            # values, so every existing checkpoint and script is bit-identical.
            _backbone = getattr(config, "expert_backbone", "custom")
            _expert_kwargs = dict(
                hidden_size=config.expert_hidden_size,
                intermediate_size=config.expert_intermediate_size,
                num_layers=tc.num_hidden_layers,
                num_heads=tc.num_attention_heads,
                num_kv_heads=tc.num_key_value_heads,
                head_dim=tc.head_dim,
                eps=tc.rms_norm_eps,
                use_adaln=(self.time_conditioning == "adaln"),
                final_adaln=(self.time_conditioning == "adaln"),
                use_per_layer_adanorm=getattr(config, "expert_per_layer_adanorm", True),
                num_foresight_tokens=getattr(config, "expert_num_foresight_tokens", 50),
            )
            if _backbone == "custom":
                self.action_expert = ActionExpert(
                    **_expert_kwargs,
                    # Per-head Q/K RMSNorm on the expert stream, mirroring the VLM's
                    # q_norm/k_norm in the joint softmax (see configuration_rynn_brain_vla).
                    use_qk_norm=getattr(config, "expert_qk_norm", False),
                )
            elif _backbone in ("qwen3_vl", "qwen2"):
                # Stock narrow decoder stack -- what every peer does (lingbot-vla-v2
                # -> Qwen2ForCausalLM, InternVLA-A1.5 -> Qwen3_5TextModel, pi0 -> a second
                # Gemma config). The head layout is always derived from text_config, which
                # is the peers' own rule (see std_expert_rynn_brain_vla.build_stock_layers);
                # the family chooses only whether the layer ships q_norm/k_norm.
                #   "qwen3_vl" -- our VLM's family. Symmetric joint softmax; the dead
                #                 `adaln` linears of the hand-written stack do not exist.
                #                 Not literally what any peer instantiates.
                #   "qwen2"    -- lingbot-vla-v2's ACTUAL expert
                #                 (modeling_lingbot_vla_v2.py:138). No q_norm/k_norm and
                #                 biased q/k/v, so the joint softmax is asymmetric exactly
                #                 as in their stack. Use this for a faithful lingbot arm.
                from .std_expert_rynn_brain_vla import StdActionExpert
                self.action_expert = StdActionExpert(
                    **_expert_kwargs,
                    foresight_visibility=getattr(config, "expert_foresight_visibility", "first"),
                    time_concat=getattr(config, "expert_time_concat", False),
                    family=_backbone,
                )
            else:
                raise ValueError(
                    f"Unknown expert_backbone={_backbone!r}; expected 'custom', "
                    "'qwen3_vl' or 'qwen2'."
                )
            # Attention-mask offset for foresight visibility. "first" (default, this repo's
            # original behaviour) hides foresight[1:] from the action tokens; "all" hides
            # none, which is what InternVLA actually does (modeling_internvla_a1_5.py:944).
            _vis = getattr(config, "expert_foresight_visibility", "first")
            if _vis not in ("first", "all"):
                raise ValueError(f"expert_foresight_visibility must be 'first' or 'all', got {_vis!r}")
            self.action_expert.foresight_vis_offset = (
                1 if _vis == "first" else max(1, self.action_expert.num_foresight_tokens)
            )
            if getattr(config, "expert_time_concat", False):
                if _backbone == "custom":
                    # action_time_mlp_in/out live on StdActionExpert only; the
                    # hand-written stack has no concat path.
                    raise ValueError(
                        "expert_time_concat requires a stock expert_backbone "
                        "('qwen3_vl' or 'qwen2'), not 'custom'."
                    )
                # (2026-08-28) The multiview/latent family used to be refused here. It is now
                # implemented: _build_action_stream_multiview takes an explicit `time_fuse`
                # and expert_action_step_multiview fuses its own noisy action block. The
                # refusal was blocking stage-1 from pretraining the pipeline xl_lingbot was
                # validated on, which is a bigger problem than the missing feature was.
            # Flag expert AdaLN projections for zero-init in _init_weights.
            # Skip entirely when use_per_layer_adanorm=True: in that mode the
            # per-layer and final AdaRMSNorm modules do the modulation themselves
            # and the per-layer FiLM ``adaln`` / ``adaln_final`` linears are dead
            # code (never called) — flagging them for zero-init would waste both
            # the init work and the optimizer slot. AdaRMSNorm's gamma/beta
            # zero-init is enforced by the re-zero walk after post_init below.
            if not self.action_expert.use_per_layer_adanorm:
                for _layer in self.action_expert.layers:
                    if getattr(_layer, "adaln", None) is not None:
                        _layer.adaln._is_adaln = True
                if getattr(self.action_expert, "adaln_final", None) is not None:
                    self.action_expert.adaln_final._is_adaln = True

            # Multi-view slot binding: slot k's identity comes from ITS camera's pooled
            # visual tokens ("content seed"), not a fixed role vocabulary. Nothing about
            # the number of views or their roles is baked into the weights, so the same
            # model serves any N (2, 6, 10, ...) cameras. Zero-init (re-zeroed after
            # post_init below): slots start symmetric and differentiate via gradient from
            # the per-view supervision.
            self.num_view_slots = getattr(config, "num_view_slots", 1)
            self.slot_seed_norm = nn.LayerNorm(config.text_config.hidden_size)
            self.slot_seed_proj = nn.Linear(config.text_config.hidden_size, config.expert_hidden_size)
            # Only additive slots use meta-safe zero initialization; standalone zero tokens amplify normalization gradients.
            if not getattr(config, "use_view_cond_slots", False):
                self.slot_seed_proj._is_zero_init = True

        # Auxiliary depth-prediction heads (pure readout off the VLM trunk; training-only,
        # dropped at inference). Current-frame depth is an anti-collapse anchor; the
        # optional future-frame head is the foresight / dynamics signal.
        self.use_depth_aux = getattr(config, "use_depth_aux", False)
        self.depth_loss_weight = getattr(config, "depth_loss_weight", 1.0)
        self.predict_future_depth = getattr(config, "predict_future_depth", True)
        self.depth_grid_size = getattr(config, "depth_grid_size", 16)
        self.depth_head_current = None
        self.depth_head_future = None
        if self.use_depth_aux:
            self.depth_head_current = DepthReadoutHead(
                dim=config.text_config.hidden_size,
                grid_size=self.depth_grid_size,
                num_layers=config.depth_aux_num_layers,
                num_heads=config.depth_aux_num_heads,
            )
            if self.predict_future_depth:
                self.depth_head_future = DepthReadoutHead(
                    dim=config.text_config.hidden_size,
                    grid_size=self.depth_grid_size,
                    num_layers=config.depth_aux_num_layers,
                    num_heads=config.depth_aux_num_heads,
                )

        # Spatial-Forcing style feature alignment (training-only): cosine-align the trunk's
        # image-token hidden states at a mid layer with a frozen 3D teacher (DA3MONO).
        # Defaults mirror openpi-SF: layer 12/18 = 2/3 depth, align_loss_coeff = 0.5.
        self.use_sf_align = getattr(config, "use_sf_align", False)
        self.sf_align_weight = getattr(config, "sf_align_weight", 0.5)
        num_trunk_layers = config.text_config.num_hidden_layers
        self.sf_align_layer = getattr(config, "sf_align_layer", None) or round(num_trunk_layers * 2 / 3)
        self.sf_teacher_path = getattr(config, "sf_teacher_path", None)
        self.sf_teacher_code = getattr(config, "sf_teacher_code", None)
        self.sf_teacher_dim = getattr(config, "sf_teacher_dim", 1024)
        self.sf_teacher_layer = getattr(config, "sf_teacher_layer", -1)
        self.sf_align_proj = None
        # Plain list keeps the frozen teacher out of state_dict, optimizer and ZeRO.
        self._sf_teacher_holder = []
        if self.use_sf_align:
            if not (1 <= self.sf_align_layer <= num_trunk_layers):
                raise ValueError(
                    f"sf_align_layer must be in [1, {num_trunk_layers}], got {self.sf_align_layer}"
                )
            self.sf_align_proj = SFAlignProjector(
                llm_dim=config.text_config.hidden_size,
                teacher_dim=self.sf_teacher_dim,
                use_vlm_norm=getattr(config, "sf_use_vlm_norm", True),
            )

        # LingBot-VLA 2.0 style query readout: the alternative student mechanism
        # for the *same* DA3MONO teacher. Unlike SF this is not training-only -- the
        # pooled query tokens live in the prefix at inference too.
        self.use_lb_align = getattr(config, "use_lb_align", False)
        self.lb_align_weight = getattr(config, "lb_align_weight", 0.5)
        self.lb_align_loss = getattr(config, "lb_align_loss", "cosine")
        self.lb_num_task_tokens = getattr(config, "lb_num_task_tokens", 8)
        self.lb_align_layer = getattr(config, "lb_align_layer", None) or num_trunk_layers
        self.lb_align = None
        if self.use_lb_align:
            if not (1 <= self.lb_align_layer <= num_trunk_layers):
                raise ValueError(
                    f"lb_align_layer must be in [1, {num_trunk_layers}], got {self.lb_align_layer}"
                )
            self.lb_align = LBQueryAlignHead(
                llm_dim=config.text_config.hidden_size,
                teacher_dim=self.sf_teacher_dim,
                num_queries=getattr(config, "lb_num_queries", 256),
                num_task_tokens=self.lb_num_task_tokens,
                num_layers=getattr(config, "lb_resampler_layers", 1),
                num_heads=getattr(config, "lb_resampler_heads", 4),
                dim_head=getattr(config, "lb_resampler_dim_head", 32),
                ff_mult=getattr(config, "lb_resampler_ff_mult", 1),
                query_init_std=getattr(config, "lb_query_init_std", 1.0),
            )

        # End-effector / embodiment type embedding, added to the action-head time
        # conditioning (cond width == action-stream width _act_hidden).
        self.use_ee_embedding = getattr(config, "use_ee_embedding", False)
        self.ee_embedding = None
        if self.use_ee_embedding:
            self.ee_embedding = nn.Embedding(config.num_ee_types, _act_hidden)

        # In-context history conditioning: encode past (state, action) pairs into tokens
        # that are scattered into <|history_pad|> prompt slots (VLM-width prefix tokens
        # the action stream attends to). Inputs are available at inference.
        self.use_history_cond = getattr(config, "use_history_cond", False)
        self.history_token_id = getattr(config, "history_token_id", -1)
        self.history_proj = None
        if self.use_history_cond:
            self.history_proj = nn.Sequential(
                nn.Linear(config.action_dim * 2, _vlm_hidden),
                nn.SiLU(),
                nn.Linear(_vlm_hidden, _vlm_hidden),
            )

        for layer in self.language_model.layers:
            layer.self_attn.forward = _attention_forward.__get__(layer.self_attn)

        # Initialize weights and apply final processing
        self.post_init()

        # Re-zero AdaRMSNorm gamma/beta: post_init() -> apply(_init_weights) walks
        # every module and runs the parent ``Qwen3VLModel._init_weights`` on
        # AdaRMSNorm's ``gamma`` / ``beta`` Linear children, which overwrites the
        # DiT-style zero-init with ``normal_(std=...)``. Re-establish the identity-
        # modulation starting point (scale=0, shift=0) so the expert begins as a
        # clean residual network.
        if self.action_expert is not None:
            for m in self.action_expert.modules():
                if isinstance(m, AdaRMSNorm):
                    nn.init.zeros_(m.gamma.weight)
                    nn.init.zeros_(m.gamma.bias)
                    nn.init.zeros_(m.beta.weight)
                    nn.init.zeros_(m.beta.bias)
            # Slot seed projection starts at zero IN THE ADDITIVE FORM: all view slots begin
            # identical and per-view supervision differentiates them. Redundant with the
            # _is_zero_init marker set at construction (which is what actually covers the
            # meta-device training path); kept so a plain local construction is zero even if
            # the marker path is ever refactored away. Gated on the marker so Arm V4
            # (use_view_cond_slots) keeps its normal_(0.02) draw here too -- see the long
            # note at construction for the measured reason.
            if getattr(self.slot_seed_proj, "_is_zero_init", False):
                nn.init.zeros_(self.slot_seed_proj.weight)
                nn.init.zeros_(self.slot_seed_proj.bias)
        # Same for the prefix role table: post_init would give it normal_(std=...), which
        # would perturb the pretrained VLM's image tokens from step 0. Starting at zero makes
        # "flag on" identical to "flag off" until gradients move it.
        if self.view_role_emb is not None:
            nn.init.zeros_(self.view_role_emb.weight)

    def _init_weights(self, module):
        super()._init_weights(module)
        # Zero-init AdaLN projections so time modulation starts as the identity
        # (scale=0, shift=0), leaving the pretrained backbone / action_out_proj intact.
        # Applies to both the shared head's adaln_in/adaln_final and the expert's
        # per-layer / final adaln (all flagged with _is_adaln).
        if getattr(module, "_is_adaln", False):
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)
        # Modules that must start at exactly zero so that enabling their feature is a
        # no-op at step 0 (an A/B then differs only by what training does with them).
        if getattr(module, "_is_zero_init", False):
            for _p in module.parameters(recurse=False):
                nn.init.zeros_(_p)
        # ``action_expert.learnable_tokens`` is a BARE nn.Parameter hanging off the
        # expert module, and BOTH initializer paths dispatch purely on the owning
        # module's TYPE:
        #   - HF   ``PreTrainedModel.initialize_weights`` -> ``_init_weights(module)``
        #   - repo ``models/__init__.py:_init_missing_weights`` -> for the missing key
        #     "action_expert.learnable_tokens" it takes rsplit(".", 1)[0] ==
        #     "action_expert" and calls ``_init_weights(<ActionExpert>)``
        # ``Qwen3VLPreTrainedModel._init_weights`` only handles Linear/Embedding/RMSNorm,
        # so an ActionExpert / StdActionExpert instance matches nothing and this
        # parameter was NEVER initialized: the ``nn.init.trunc_normal_(std=0.02)`` in
        # the expert __init__ is a no-op because the model is built on the meta device
        # (``models/__init__.py:_init_empty_params()`` / HF ``init_empty_weights()``).
        # It is the ONLY parameter in the model that no module-type dispatch reaches
        # (verified by scanning every named_parameter against its owner's type).
        # Measured consequences before this fix:
        #   - training path: ``torch.empty_like(..., device="cuda")`` on a fresh
        #     cudaMalloc returns ZEROS, so all 50 foresight tokens started identical
        #     and exactly 0 (verified: _init_missing_weights left 51200/51200 zeros
        #     bit-unchanged; the 4B E2 run's step-2000 learnable_tokens has std 8.7e-4,
        #     the same magnitude as its known zero-init Linear bias 8.0e-4, versus
        #     2.0e-2 for the correctly-initialized lt_in_proj.weight next to it).
        #     This is the mechanical cause of the effective-rank~1 foresight collapse
        #     measured on the E1/E2/E3 checkpoints.
        #   - CPU, or any allocator handing back a recycled dirty block: genuine
        #     uninitialized memory -- measured up to 8064/51200 non-finite entries with
        #     absmax 3.2e38, which NaNs the loss on the very first forward.
        # Arms with expert_num_foresight_tokens=0 (xp / xl) do not create this
        # parameter at all and are unaffected either way.
        lt = getattr(module, "learnable_tokens", None)
        if isinstance(lt, nn.Parameter) and not lt.is_meta:
            # Draw in fp32, then cast. ``trunc_normal_`` is NOT bf16-safe: it does
            # ``uniform_(2l-1, 2u-1).erfinv_()``, and in bfloat16 (8-bit mantissa)
            # ~0.2% of the uniform draws round to exactly +/-1, where erfinv is +/-inf;
            # the trailing ``clamp_(a, b)`` then pins them to the BOUNDS a=-2 / b=2
            # instead of discarding them. Measured directly on the bf16 parameter:
            # std 8.9e-2 with absmax exactly 2.0, i.e. 100x the requested std=0.02.
            with torch.no_grad():
                buf = torch.empty(lt.shape, dtype=torch.float32, device=lt.device)
                nn.init.trunc_normal_(buf, std=0.02)
                lt.copy_(buf)
        if (
            isinstance(module, LBQueryAlignHead)
            and not module.align_embs.is_meta
            and not getattr(module.align_embs, "_is_hf_initialized", False)
        ):
            with torch.no_grad():
                buf = torch.empty(
                    module.align_embs.shape,
                    dtype=torch.float32,
                    device=module.align_embs.device,
                )
                nn.init.normal_(buf, mean=0.0, std=module.query_init_std)
                module.align_embs.copy_(buf)

    @staticmethod
    def _film(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """DiT-style FiLM modulation: x * (1 + scale) + shift.

        ``x`` is (B, chunk, H); ``shift`` / ``scale`` are (B, H), broadcast over chunk.
        """
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def _compute_depth_loss(
        self,
        prefix_hidden,
        visual_pos_masks,
        depth_target,
        depth_mask,
        future_depth_target,
        future_depth_mask,
    ):
        """Auxiliary depth loss (masked smooth_l1) over the current + future depth grids.

        Pure readout off the VLM trunk visual tokens; returns None when disabled, when
        no depth target is provided, or when batch dims don't line up (packing guard).
        """
        if self.depth_head_current is None or depth_target is None or visual_pos_masks is None:
            return None
        b = prefix_hidden.size(0)
        if visual_pos_masks.size(0) != b or depth_target.size(0) != b:
            return None

        key_padding_mask = ~visual_pos_masks.bool()  # True = ignore (non-visual token)
        # Rows with zero visual tokens would make attention produce NaN; relax them to
        # attend everything (their depth target should be masked out anyway).
        valid_rows = (~key_padding_mask).any(dim=1)
        if not bool(valid_rows.all()):
            key_padding_mask = key_padding_mask & valid_rows.unsqueeze(1)

        def _masked_smooth_l1(pred_flat, tgt, msk):
            tgt = tgt.reshape(b, -1).to(pred_flat.dtype)
            if msk is None:
                msk = torch.ones_like(tgt)
            else:
                msk = msk.reshape(b, -1).to(pred_flat.dtype)
            diff = F.smooth_l1_loss(pred_flat, tgt, reduction="none") * msk
            return diff.sum() / msk.sum().clamp(min=1.0)

        pred_cur = self.depth_head_current(prefix_hidden, key_padding_mask)
        loss = _masked_smooth_l1(pred_cur, depth_target, depth_mask)
        if self.depth_head_future is not None and future_depth_target is not None:
            pred_fut = self.depth_head_future(prefix_hidden, key_padding_mask)
            loss = loss + _masked_smooth_l1(pred_fut, future_depth_target, future_depth_mask)
        return loss

    def _compute_sf_align_loss(
        self,
        mid_hidden,
        visual_pos_masks,
        image_grid_thw,
        teacher_images,
        teacher_image_grid_indices,
    ):
        """Spatial-Forcing alignment loss; None when disabled or inputs don't line up."""
        if not self.use_sf_align or self.sf_align_proj is None:
            return None
        if mid_hidden is None or teacher_images is None or image_grid_thw is None:
            return None
        if not self._sf_teacher_holder:
            if not self.sf_teacher_path:
                raise ValueError("use_sf_align=True requires sf_teacher_path (DA3MONO checkpoint dir)")
            teacher = DA3MonoTeacher(
                self.sf_teacher_path, code_dir=self.sf_teacher_code, feat_layer=self.sf_teacher_layer
            ).to(mid_hidden.device)
            if teacher.feat_dim is not None and teacher.feat_dim != self.sf_teacher_dim:
                raise ValueError(
                    f"sf_teacher_dim={self.sf_teacher_dim} != teacher feature dim {teacher.feat_dim}"
                )
            self._sf_teacher_holder.append(teacher)
        teacher = self._sf_teacher_holder[0]
        teacher_feats = teacher.extract(teacher_images.to(device=mid_hidden.device, dtype=torch.float32))
        if teacher_feats.shape[-1] != self.sf_teacher_dim:
            raise ValueError(
                f"sf_teacher_dim={self.sf_teacher_dim} != teacher feature dim {teacher_feats.shape[-1]}"
            )
        return compute_sf_align_loss(
            mid_hidden=mid_hidden,
            visual_pos_masks=visual_pos_masks,
            image_grid_thw=image_grid_thw,
            teacher_feats=teacher_feats,
            projector=self.sf_align_proj,
            spatial_merge_size=self.config.vision_config.spatial_merge_size,
            teacher_grid_indices=teacher_image_grid_indices,
        )

    def _append_align_queries(
        self,
        inputs_embeds,
        position_ids,
        visual_pos_masks,
        attention_mask,
        input_ids,
    ):
        Q = self.lb_num_task_tokens
        B, seq_len = inputs_embeds.shape[:2]
        if input_ids is None or input_ids.shape != (B, seq_len):
            shape = None if input_ids is None else tuple(input_ids.shape)
            raise ValueError(
                f"use_lb_align=True requires input_ids aligned with the prefix; "
                f"input_ids={shape}, prefix={(B, seq_len)}"
            )

        valid = (
            attention_mask.to(device=input_ids.device, dtype=torch.bool)
            if attention_mask is not None
            else torch.ones_like(input_ids, dtype=torch.bool)
        )
        state_mask = input_ids.eq(self.config.state_token_id) & valid
        state_counts = state_mask.sum(dim=1)
        if not torch.all(state_counts == 1):
            raise ValueError(
                "use_lb_align=True requires exactly one valid state token per sample; "
                f"counts={state_counts.tolist()}"
            )

        task = self.lb_align.task_tokens().to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        query_mask = torch.zeros(B, seq_len + Q, dtype=torch.bool, device=inputs_embeds.device)
        embed_rows = []
        position_rows = [] if position_ids is not None else None
        visual_rows = [] if visual_pos_masks is not None else None
        attention_rows = [] if attention_mask is not None else None

        for b, state_idx in enumerate(state_mask.to(torch.int64).argmax(dim=1).tolist()):
            embed_rows.append(torch.cat([inputs_embeds[b, :state_idx], task, inputs_embeds[b, state_idx:]], dim=0))
            query_mask[b, state_idx : state_idx + Q] = True

            if position_ids is not None:
                state_position = position_ids[:, b, state_idx].amax()
                query_positions = state_position + torch.arange(
                    Q, device=position_ids.device, dtype=position_ids.dtype
                )
                position_rows.append(torch.cat([
                    position_ids[:, b, :state_idx],
                    query_positions.view(1, Q).expand(3, -1),
                    position_ids[:, b, state_idx:] + Q,
                ], dim=-1))
            if visual_pos_masks is not None:
                visual_rows.append(torch.cat([
                    visual_pos_masks[b, :state_idx],
                    torch.zeros(Q, dtype=torch.bool, device=visual_pos_masks.device),
                    visual_pos_masks[b, state_idx:],
                ]))
            if attention_mask is not None:
                attention_rows.append(torch.cat([
                    attention_mask[b, :state_idx],
                    torch.ones(Q, dtype=attention_mask.dtype, device=attention_mask.device),
                    attention_mask[b, state_idx:],
                ]))

        inputs_embeds = torch.stack(embed_rows, dim=0)
        if position_rows is not None:
            position_ids = torch.stack(position_rows, dim=1)
        if visual_rows is not None:
            visual_pos_masks = torch.stack(visual_rows, dim=0)
        if attention_rows is not None:
            attention_mask = torch.stack(attention_rows, dim=0)
        return inputs_embeds, position_ids, visual_pos_masks, attention_mask, query_mask

    def _compute_lb_align_loss(
        self,
        readout_hidden,
        visual_pos_masks,
        image_grid_thw,
        teacher_images,
        query_mask,
        teacher_image_grid_indices,
        primary_teacher_indices,
    ):
        """Compute the required LBQ distillation objective when the arm is enabled."""
        if not self.use_lb_align or self.lb_align is None:
            return None
        inputs = {
            "readout_hidden": readout_hidden,
            "visual_pos_masks": visual_pos_masks,
            "image_grid_thw": image_grid_thw,
            "teacher_images": teacher_images,
            "query_mask": query_mask,
            "teacher_image_grid_indices": teacher_image_grid_indices,
            "primary_teacher_indices": primary_teacher_indices,
        }
        missing = [name for name, value in inputs.items() if value is None]
        if missing:
            shapes = {name: None if value is None else tuple(value.shape) for name, value in inputs.items()}
            raise ValueError(
                f"use_lb_align=True requires every LBQ input; missing={missing}, shapes={shapes}"
            )
        primary_teacher_indices = primary_teacher_indices.to(device=teacher_images.device, dtype=torch.long)
        if primary_teacher_indices.shape != (readout_hidden.size(0),):
            raise ValueError(
                f"primary_teacher_indices must have shape ({readout_hidden.size(0)},), "
                f"got {tuple(primary_teacher_indices.shape)}"
            )
        if primary_teacher_indices.numel() and (
            primary_teacher_indices.min() < 0 or primary_teacher_indices.max() >= teacher_images.size(0)
        ):
            raise ValueError(
                f"primary_teacher_indices={primary_teacher_indices.tolist()} are out of range for "
                f"teacher_images batch={teacher_images.size(0)}"
            )
        teacher_image_grid_indices = teacher_image_grid_indices.to(
            device=teacher_images.device, dtype=torch.long
        )
        if teacher_image_grid_indices.shape != (teacher_images.size(0),):
            raise ValueError(
                f"teacher_image_grid_indices must have shape ({teacher_images.size(0)},), "
                f"got {tuple(teacher_image_grid_indices.shape)}"
            )
        primary_grid_indices = teacher_image_grid_indices[primary_teacher_indices]

        if not self._sf_teacher_holder:
            if not self.sf_teacher_path:
                raise ValueError("use_lb_align=True requires sf_teacher_path (DA3MONO checkpoint dir)")
            teacher = DA3MonoTeacher(
                self.sf_teacher_path, code_dir=self.sf_teacher_code, feat_layer=self.sf_teacher_layer
            ).to(readout_hidden.device)
            self._sf_teacher_holder.append(teacher)
        teacher = self._sf_teacher_holder[0]
        primary_images = teacher_images[primary_teacher_indices]
        teacher_feats = teacher.extract(primary_images.to(device=readout_hidden.device, dtype=torch.float32))
        if teacher_feats.shape[-1] != self.sf_teacher_dim:
            raise ValueError(
                f"sf_teacher_dim={self.sf_teacher_dim} != teacher feature dim {teacher_feats.shape[-1]}"
            )
        return compute_lb_align_loss(
            final_hidden=readout_hidden,
            visual_pos_masks=visual_pos_masks,
            image_grid_thw=image_grid_thw,
            teacher_feats=teacher_feats,
            query_mask=query_mask,
            primary_grid_indices=primary_grid_indices,
            head=self.lb_align,
            spatial_merge_size=self.config.vision_config.spatial_merge_size,
            loss_type=self.lb_align_loss,
        )

    def sample_noise(self, shape):
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=self.device,
        )
        return noise

    def sample_time(self, batch_size):
        beta_dist = torch.distributions.Beta(concentration1=1.5, concentration0=1.0)
        time_beta = beta_dist.sample((batch_size,)).to(device=self.device, dtype=torch.float32)
        time = time_beta * 0.999 + 0.001
        return time

    def preprocess_actions(
        self,
        actions: Optional[torch.FloatTensor] = None,
        times: Optional[torch.FloatTensor] = None,
    ):
        targets = None
        if self.training:
            assert actions is not None

            if times is None:
                times = self.sample_time(actions.size(0))

            # Broadcast per-sample time over any action rank: 3D robot actions
            # (B, chunk, dim) or 4D multi-view latent targets (B, K, chunk, dim).
            expanded_times = times.view(times.size(0), *([1] * (actions.dim() - 1)))
            noises = self.sample_noise(actions.shape)
            clean_actions = actions
            actions = expanded_times * noises + (1 - expanded_times) * clean_actions
            targets = noises - clean_actions

        return actions, targets, times

    def get_state_features(
        self,
        states: Optional[torch.FloatTensor] = None,
    ):
        assert states.ndim == 3
        assert states.size(-1) == self.config.action_dim
        return self.state_proj(states.type(self.dtype))

    def get_action_features(
        self,
        actions: Optional[torch.FloatTensor] = None,
        times: Optional[torch.FloatTensor] = None,
        ee_bias: Optional[torch.FloatTensor] = None,
    ):
        assert actions.ndim == 3
        assert actions.size(-1) == self.config.action_dim

        action_embeds = self.action_in_proj(actions.type(self.dtype))
        time_embeds = _create_sinusoidal_pos_embedding(
            times,
            action_embeds.size(-1),
            device=action_embeds.device,
        ).type(dtype=self.dtype)

        if self.time_conditioning == "adaln":
            # DiT-style AdaLN/FiLM: modulate the action tokens at the input.
            cond = self.time_mlp(time_embeds)  # (B, H)
            if ee_bias is not None:
                cond = cond + ee_bias
            shift_in, scale_in = self.adaln_in(cond).chunk(2, dim=-1)
            action_embeds = self._film(action_embeds, shift_in, scale_in)
            return action_embeds, cond

        # Legacy concat-then-MLP time conditioning.
        time_embeds = time_embeds.unsqueeze(1).expand_as(action_embeds)
        action_embeds = torch.cat([action_embeds, time_embeds], dim=-1)
        action_embeds = self.action_time_proj(action_embeds)
        return action_embeds, None
    
    def get_state_token_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        features: Optional[torch.FloatTensor] = None,
    ):
        if input_ids is None:
            special_token_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.state_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_token_mask = special_token_mask.all(-1)
        else:
            special_token_mask = input_ids == self.config.state_token_id

        n_special_tokens = special_token_mask.sum()
        special_token_mask = special_token_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if features is not None and inputs_embeds[special_token_mask].numel() != features.numel():
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_special_tokens}, features {features.shape[0]}"
            )

        return special_token_mask

    def get_history_token_mask(self, input_ids, inputs_embeds):
        if input_ids is None:
            mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.history_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            mask = mask.all(-1)
        else:
            mask = input_ids == self.history_token_id
        return mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)

    def get_history_features(self, history_states, history_actions):
        """Encode past (state, action) pairs into VLM-width history tokens.

        history_states / history_actions: (B, K, action_dim). Returns (B, K, hidden).
        """
        x = torch.cat([history_states, history_actions], dim=-1).type(self.dtype)
        return self.history_proj(x)

    def _embed_prefix(
        self,
        input_ids,
        inputs_embeds,
        position_ids,
        attention_mask,
        past_key_values,
        pixel_values,
        pixel_values_videos,
        image_grid_thw,
        video_grid_thw,
        states,
        actions,
        history_states=None,
        history_actions=None,
        camera_slot_ids=None,
    ):
        """Build the VLM prefix embeddings (image/video/state scatter) and position ids.

        Shared by both the shared-backbone head and the independent action expert.
        """
        if inputs_embeds is None and input_ids is None:
            assert actions is not None
            inputs_embeds = torch.zeros(
                (actions.size(0), 0, self.config.text_config.hidden_size),
                dtype=self.dtype,
                device=self.device,
            )
        elif inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_mask = None
        video_mask = None

        def _get_visual_features(fn, *args):
            try:
                outputs = fn(*args, return_dict=True)
            except TypeError as exc:
                if "return_dict" not in str(exc):
                    raise
                outputs = fn(*args)
            if hasattr(outputs, "pooler_output"):
                return outputs.pooler_output, outputs.deepstack_features
            pooler_output = outputs[0]
            deepstack_features = outputs[1] if len(outputs) > 1 else None
            return pooler_output, deepstack_features

        if pixel_values is not None:
            image_embeds, deepstack_image_embeds = _get_visual_features(
                self.get_image_features, pixel_values, image_grid_thw
            )
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds, deepstack_video_embeds = _get_visual_features(
                self.get_video_features, pixel_values_videos, video_grid_thw
            )
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask is not None and video_mask is not None:
            # aggregate visual_pos_masks and deepstack_visual_embeds
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
            visual_pos_masks = image_mask | video_mask
            deepstack_visual_embeds = []
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
                embed_joint[image_mask_joint, :] = img_embed
                embed_joint[video_mask_joint, :] = vid_embed
                deepstack_visual_embeds.append(embed_joint)
        elif image_mask is not None:
            image_mask = image_mask[..., 0]
            visual_pos_masks = image_mask
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            video_mask = video_mask[..., 0]
            visual_pos_masks = video_mask
            deepstack_visual_embeds = deepstack_video_embeds

        state_mask = self.get_state_token_mask(input_ids, inputs_embeds=inputs_embeds)
        if state_mask.any():
            state_embeds = self.get_state_features(states)
            state_embeds = state_embeds.view(-1, state_embeds.size(-1))
            inputs_embeds = inputs_embeds.masked_scatter(state_mask, state_embeds)

        if self.use_history_cond and history_states is not None and history_actions is not None:
            history_mask = self.get_history_token_mask(input_ids, inputs_embeds=inputs_embeds)
            if history_mask.any():
                history_embeds = self.get_history_features(history_states, history_actions)
                history_embeds = history_embeds.reshape(-1, history_embeds.size(-1))
                inputs_embeds = inputs_embeds.masked_scatter(history_mask, history_embeds)

        # B2 camera identity, applied after every scatter so it lands on the final image
        # token embeddings and is seen by all downstream paths (direct expert, shared head
        # and multi-view latent alike). No-op unless use_view_role_embedding.
        inputs_embeds = self._add_view_role_embedding(
            inputs_embeds, visual_pos_masks, image_grid_thw, camera_slot_ids
        )

        if position_ids is None:
            position_ids = self.compute_3d_position_ids(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )

        return inputs_embeds, position_ids, visual_pos_masks, deepstack_visual_embeds

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[RynnBrainVLACache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        actions: Optional[torch.FloatTensor] = None,
        action_mask: Optional[torch.BoolTensor] = None,
        states: Optional[torch.FloatTensor] = None,
        times: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        depth_target: Optional[torch.FloatTensor] = None,
        depth_mask: Optional[torch.BoolTensor] = None,
        future_depth_target: Optional[torch.FloatTensor] = None,
        future_depth_mask: Optional[torch.BoolTensor] = None,
        history_states: Optional[torch.FloatTensor] = None,
        history_actions: Optional[torch.FloatTensor] = None,
        ee_type_id: Optional[torch.LongTensor] = None,
        latent_targets: Optional[torch.FloatTensor] = None,
        latent_noise: Optional[torch.FloatTensor] = None,
        slot_mask: Optional[torch.BoolTensor] = None,
        camera_slot_ids: Optional[torch.LongTensor] = None,
        teacher_images: Optional[torch.FloatTensor] = None,
        teacher_image_grid_indices: Optional[torch.LongTensor] = None,
        primary_teacher_indices: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, RynnBrainVLAModelOutputWithPast]:
        inputs_embeds, position_ids, visual_pos_masks, deepstack_visual_embeds = self._embed_prefix(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            states=states,
            actions=actions,
            history_states=history_states,
            history_actions=history_actions,
            camera_slot_ids=camera_slot_ids,
        )

        # Multi-view slot-based latent path: expert emits one latent-action chunk per
        # view-role slot. Takes precedence over the single-stream / amortized paths.
        if (
            self.action_head_type == "expert"
            and self.use_latent_actions
            and (latent_targets is not None or latent_noise is not None)
            and past_key_values is None
        ):
            if "cu_seq_lens_q" in kwargs:
                raise ValueError(
                    "action_head_type='expert' does not support sequence packing yet; "
                    "run with --sequence_packing False."
                )
            if latent_targets is not None and latent_noise is not None:
                raise ValueError("Pass either latent_targets (supervision) or latent_noise (canvas), not both.")
            if latent_noise is not None:
                # Generation canvas: the input is already x_t (typically pure noise at
                # times=1); no noising, no flow targets, in any train/eval mode.
                x_t, lat_targets = latent_noise, None
            else:
                if not self.training:
                    raise ValueError(
                        "latent_targets carries clean supervision latents and requires training "
                        "mode (they get noised and turned into flow-matching targets). On an "
                        "eval-mode model they would silently be used as a raw generation canvas "
                        "instead — pass latent_noise=... if a canvas is what you meant."
                    )
                x_t, lat_targets, times = self.preprocess_actions(latent_targets, times)
            lb_query_mask = None
            if self.use_lb_align:
                (
                    inputs_embeds,
                    position_ids,
                    visual_pos_masks,
                    attention_mask,
                    lb_query_mask,
                ) = self._append_align_queries(
                    inputs_embeds, position_ids, visual_pos_masks, attention_mask, input_ids
                )
            mv_ee_bias = None
            if self.use_ee_embedding and ee_type_id is not None:
                mv_ee_bias = self.ee_embedding(ee_type_id)
            mv_slot_seeds = self._pool_view_seeds(
                inputs_embeds, visual_pos_masks, image_grid_thw, camera_slot_ids,
                num_slots=x_t.size(1),
            )
            mv_view_cond = self._build_view_cond_tokens(
                inputs_embeds, visual_pos_masks, image_grid_thw, camera_slot_ids
            )
            return self._forward_expert_multiview(
                prefix_embeds=inputs_embeds,
                prefix_position_ids=position_ids,
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_visual_embeds,
                attention_mask=attention_mask,
                actions=x_t,
                times=times,
                targets=lat_targets,
                slot_mask=slot_mask,
                rope_deltas=rope_deltas,
                depth_target=depth_target,
                depth_mask=depth_mask,
                future_depth_target=future_depth_target,
                future_depth_mask=future_depth_mask,
                ee_bias=mv_ee_bias,
                slot_seeds=mv_slot_seeds,
                view_cond=mv_view_cond,
                image_grid_thw=image_grid_thw,
                teacher_images=teacher_images,
                teacher_image_grid_indices=teacher_image_grid_indices,
                primary_teacher_indices=primary_teacher_indices,
                lb_query_mask=lb_query_mask,
            )

        # VLM-forward amortization (training, expert path): run the VLM prefix once and
        # the narrow expert stream over N=expert_train_repeat independent noise/time draws.
        _train_repeat = getattr(self.config, "expert_train_repeat", 1)
        if (
            self.action_head_type == "expert"
            and self.training
            and _train_repeat > 1
            and past_key_values is None
            and actions is not None
        ):
            if "cu_seq_lens_q" in kwargs:
                raise ValueError(
                    "action_head_type='expert' does not support sequence packing yet; "
                    "run with --sequence_packing False."
                )
            if self.use_sf_align:
                raise ValueError(
                    "use_sf_align is not supported with expert_train_repeat > 1; "
                    "set expert_train_repeat=1 for SF alignment runs."
                )
            N = _train_repeat
            clean_rep = actions.repeat_interleave(N, dim=0)
            actions_noisy, targets, times = self.preprocess_actions(clean_rep, None)
            am = action_mask.repeat_interleave(N, dim=0) if action_mask is not None else None
            if am is not None:
                actions_noisy[~am] = 0.0
            amortized_ee_bias = None
            if self.use_ee_embedding and ee_type_id is not None:
                amortized_ee_bias = self.ee_embedding(ee_type_id).repeat_interleave(N, dim=0)
            if getattr(self, "use_view_cond_slots", False):
                raise ValueError(
                    "use_view_cond_slots is wired for _forward_expert / _expert_decode "
                    "(expert_train_repeat=1, which is what the xl_lingbot setting uses) and "
                    "for the latent multiview path. The amortized repeat>1 path would "
                    "silently drop the camera tokens at train time while inference still "
                    "builds them -- a train/eval mismatch. Run with expert_train_repeat=1."
                )
            return self._forward_expert_amortized(
                prefix_embeds=inputs_embeds,
                prefix_position_ids=position_ids,
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_visual_embeds,
                attention_mask=attention_mask,
                actions=actions_noisy,
                times=times,
                targets=targets,
                action_mask=am,
                rope_deltas=rope_deltas,
                train_repeat=N,
                depth_target=depth_target,
                depth_mask=depth_mask,
                future_depth_target=future_depth_target,
                future_depth_mask=future_depth_mask,
                ee_bias=amortized_ee_bias,
            )

        actions, targets, times = self.preprocess_actions(
            actions, times
        )

        if actions is not None and action_mask is not None:
            actions[~action_mask] = 0.0

        ee_bias = None
        if self.use_ee_embedding and ee_type_id is not None:
            ee_bias = self.ee_embedding(ee_type_id)

        if self.action_head_type == "expert":
            if "cu_seq_lens_q" in kwargs:
                raise ValueError(
                    "action_head_type='expert' does not support sequence packing yet; "
                    "run with --sequence_packing False."
                )
            lb_query_mask = None
            if self.use_lb_align and not (past_key_values is not None and actions is not None):
                (
                    inputs_embeds,
                    position_ids,
                    visual_pos_masks,
                    attention_mask,
                    lb_query_mask,
                ) = self._append_align_queries(
                    inputs_embeds, position_ids, visual_pos_masks, attention_mask, input_ids
                )
            return self._forward_expert(
                prefix_embeds=inputs_embeds,
                prefix_position_ids=position_ids,
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_visual_embeds,
                attention_mask=attention_mask,
                actions=actions,
                times=times,
                targets=targets,
                action_mask=action_mask,
                rope_deltas=rope_deltas,
                past_key_values=past_key_values,
                cache_position=cache_position,
                depth_target=depth_target,
                depth_mask=depth_mask,
                future_depth_target=future_depth_target,
                future_depth_mask=future_depth_mask,
                ee_bias=ee_bias,
                image_grid_thw=image_grid_thw,
                camera_slot_ids=camera_slot_ids,
                teacher_images=teacher_images,
                teacher_image_grid_indices=teacher_image_grid_indices,
                primary_teacher_indices=primary_teacher_indices,
                lb_query_mask=lb_query_mask,
            )

        num_action_tokens = 0
        adaln_cond = None
        if actions is not None:
            action_embeds, adaln_cond = self.get_action_features(actions, times, ee_bias=ee_bias)
            action_position_ids = torch.arange(
                start=1,
                end=actions.size(1) + 1,
                step=1,
                dtype=position_ids.dtype,
                device=position_ids.device,
            ).unsqueeze(0).unsqueeze(0).repeat(3, actions.size(0), 1)

            if "cu_seq_lens_q" in kwargs:
                kwargs["cu_seq_lens_action"] = torch.arange(
                    start=0,
                    end=actions.size(0) * actions.size(1) + 1,
                    step=actions.size(1),
                    dtype=torch.int32,
                    device=self.device,
                )
                kwargs["max_length_action"] = actions.size(1)
                action_embeds = action_embeds.flatten(0, 1).unsqueeze(0)
                action_position_ids += position_ids[:, 0, kwargs["cu_seq_lens_q"][1:] - 1].unsqueeze(-1)
                action_position_ids = action_position_ids.flatten(1, 2).unsqueeze(1)
            else:
                action_position_ids += position_ids[:, :, -1:]

            num_action_tokens = action_embeds.size(1)
            inputs_embeds = torch.cat([inputs_embeds, action_embeds], dim=1)

            if attention_mask is not None:
                attention_mask = F.pad(attention_mask, (0, num_action_tokens), value=1)

            if position_ids.size(-1) != inputs_embeds.size(1):
                position_ids = torch.cat([position_ids, action_position_ids], dim=-1)

            if visual_pos_masks is not None:
                visual_pos_masks = F.pad(visual_pos_masks, (0, inputs_embeds.size(1) - visual_pos_masks.size(1)), value=False)

        if self.use_lb_align:
            raise ValueError(
                "use_lb_align is only wired for the multi-view latent expert path; this "
                "run reached the shared-backbone action head."
            )
        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            num_action_tokens=num_action_tokens,
            **kwargs,
        )

        loss = pred_actions = None
        if actions is not None:
            action_hidden_states = outputs.last_hidden_state[:, -num_action_tokens:]
            # Normalize layout to (B, chunk, H); the action tail is row-major over
            # (B, chunk) in both packed and unpacked modes.
            action_hidden_states = action_hidden_states.reshape(
                actions.size(0), actions.size(1), -1
            )
            if self.time_conditioning == "adaln" and adaln_cond is not None:
                # DiT final-layer conditioning before the output projection.
                shift_out, scale_out = self.adaln_final(adaln_cond).chunk(2, dim=-1)
                action_hidden_states = self._film(action_hidden_states, shift_out, scale_out)

            # Hierarchical VLA: return latent actions for an external decoder; direct mode predicts action velocity.
            latent_mode = self.use_latent_actions and self.latent_action_head is not None
            if latent_mode:
                pred_actions = self.latent_action_head(action_hidden_states)
            else:
                pred_actions = self._readout_action(action_hidden_states)
                pred_actions = pred_actions.view(actions.shape)

            if targets is not None and not latent_mode:
                if action_mask is not None:
                    pred_masked = pred_actions.type(targets.dtype)[action_mask]
                    targets_masked = targets[action_mask]
                    loss = F.mse_loss(pred_masked, targets_masked)
                else:
                    loss = F.mse_loss(pred_actions.type(targets.dtype), targets)

        if self.use_depth_aux and depth_target is not None:
            depth_loss = self._compute_depth_loss(
                outputs.last_hidden_state, visual_pos_masks,
                depth_target, depth_mask, future_depth_target, future_depth_mask,
            )
            if depth_loss is not None:
                loss = depth_loss * self.depth_loss_weight if loss is None else loss + self.depth_loss_weight * depth_loss

        return RynnBrainVLAModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            rope_deltas=rope_deltas,
            actions=pred_actions,
            loss=loss,
        )

    def _image_token_spans(self, visual_pos_masks, image_grid_thw):
        """Locate every image's prefix token span: list of ``(b, img_idx, j, positions)``.

        ``img_idx`` indexes the batch-flat image list (the order camera_slot_ids and
        image_grid_thw use), ``j`` is the image's position within its own sample. Returns
        None when the visual layout cannot be matched to the grid (e.g. video inputs), so
        callers no-op rather than mis-attribute tokens.
        """
        if visual_pos_masks is None or image_grid_thw is None:
            return None
        merge = self.config.vision_config.spatial_merge_size ** 2
        tokens_per_image = (image_grid_thw.prod(dim=-1) // merge).tolist()
        if visual_pos_masks.sum().item() != sum(tokens_per_image):
            return None
        spans = []
        img_idx = 0
        for b in range(visual_pos_masks.size(0)):
            pos = visual_pos_masks[b].nonzero(as_tuple=True)[0]
            offset = 0
            j = 0
            while offset < pos.numel() and img_idx < len(tokens_per_image):
                n = int(tokens_per_image[img_idx])
                spans.append((b, img_idx, j, pos[offset:offset + n]))
                offset += n
                img_idx += 1
                j += 1
        return spans

    def _add_view_role_embedding(self, inputs_embeds, visual_pos_masks, image_grid_thw, camera_slot_ids):
        """B2: add camera-role identity to each camera's image tokens in the VLM prefix.

        Roles are the dataset's constants.VIEW_ROLES ids carried by ``camera_slot_ids``;
        anything outside [0, NUM_VIEW_SLOTS) (notably the -1 tag on a visual_instruction
        image, or a camera the dataset did not map) is left untouched. No-op unless
        ``use_view_role_embedding``.
        """
        if self.view_role_emb is None or camera_slot_ids is None:
            return inputs_embeds
        spans = self._image_token_spans(visual_pos_masks, image_grid_thw)
        if not spans:
            return inputs_embeds
        n_roles = self.view_role_emb.num_embeddings
        role_ids = torch.full(
            inputs_embeds.shape[:2], -1, dtype=torch.long, device=inputs_embeds.device
        )
        for b, img_idx, _j, span in spans:
            role = int(camera_slot_ids[img_idx])
            if 0 <= role < n_roles:
                role_ids[b, span] = role
        valid = role_ids >= 0
        if not bool(valid.any()):
            return inputs_embeds
        emb = self.view_role_emb(role_ids.clamp(min=0)).to(inputs_embeds.dtype)
        return inputs_embeds + emb * valid.unsqueeze(-1).to(inputs_embeds.dtype)

    def _pool_view_seeds(self, prefix_embeds, visual_pos_masks, image_grid_thw, camera_slot_ids,
                         num_slots, return_valid=False):
        """Pool each camera image's visual tokens into a per-slot seed: (B, K, H_text).

        Image j of a sample goes to slot ``camera_slot_ids[j]`` when provided, else to slot
        j (sample order). Returns None when the visual layout cannot be matched to the grid
        (e.g. video inputs), in which case slots stay symmetric.

        ``return_valid=True`` additionally returns a (B, K) bool marking which slots an
        actual camera landed in. Needed by the fixed-role axis (Arm V4 / plan 1), where a
        dataset with 2 cameras leaves K-2 of the ``constants.NUM_VIEW_SLOTS`` role slots
        empty -- those must be key-masked rather than attended as all-zero tokens. Do NOT
        infer this from ``seeds == 0``: a pooled feature is not guaranteed nonzero, and the
        failure mode is silent.
        """
        spans = self._image_token_spans(visual_pos_masks, image_grid_thw)
        if spans is None:
            return (None, None) if return_valid else None
        seeds = prefix_embeds.new_zeros(prefix_embeds.size(0), num_slots, prefix_embeds.size(-1))
        valid = torch.zeros(prefix_embeds.size(0), num_slots, dtype=torch.bool,
                            device=prefix_embeds.device)
        for b, img_idx, j, span in spans:
            slot = int(camera_slot_ids[img_idx]) if camera_slot_ids is not None else j
            if 0 <= slot < num_slots:
                seeds[b, slot] = prefix_embeds[b, span].mean(dim=0)
                valid[b, slot] = True
        return (seeds, valid) if return_valid else seeds

    def _build_view_cond_tokens(self, prefix_embeds, visual_pos_masks, image_grid_thw,
                                camera_slot_ids):
        """Arm V4: one expert-width context token per CAMERA ROLE, or (None, None).

        Layout produced by the callers is ``[foresight | V camera tokens | action tokens]``
        -- structurally the same trick as the foresight tokens (prepend, then slice off
        before readout), but the content is ``slot_seed_proj(norm(pool(camera_k)))`` instead
        of a learnable constant, and slot k is pinned to role k for every dataset and every
        stage. That pinning is the whole point: it is what lets a stage-1-pretrained
        slot_seed_proj mean the same thing here.

        Why a separate token per camera and not a sum folded into ``cond``: slot_seed_proj is
        LINEAR, so sum_k proj(s_k) == proj(sum_k s_k) + (K-1)b -- permutation invariant, i.e.
        zero camera identity. Identity has to be carried by POSITION.
        """
        if not self.use_view_cond_slots or self.action_expert is None:
            return None, None
        if camera_slot_ids is None:
            # Without role ids _pool_view_seeds falls back to slot = sample order, which for
            # LIBERO's sorted (front, wrist) pair means roles 0,1 instead of 3,1: the arm
            # would train and eval on DIFFERENT role axes and only show up as lost accuracy.
            # The dataset (base.py, via camera_slot_map) and the inference servers
            # (via SERVER_CAMERA_SLOT_MAP) always supply them, so reaching here means a
            # caller that does not, and it must not be silent.
            raise ValueError(
                "use_view_cond_slots=True requires camera_slot_ids (constants.VIEW_ROLES "
                "ids from the dataset's / server's camera_slot_map). Got None, which would "
                "silently fall back to sample-order slots and skew train against eval."
            )
        seeds, valid = self._pool_view_seeds(
            prefix_embeds, visual_pos_masks, image_grid_thw, camera_slot_ids,
            num_slots=NUM_VIEW_SLOTS, return_valid=True,
        )
        if seeds is None:
            return None, None
        tokens = self.slot_seed_proj(self.slot_seed_norm(seeds.type(self.dtype)))
        return tokens, valid

    def _build_action_stream_multiview(self, actions, times, slot_mask, ee_bias, position_dtype,
                                       slot_seeds=None, view_cond=(None, None), *, time_fuse):
        """Build the K-slot action stream shared by the joint forward and the cached step.

        Returns ``(action_hidden, cond, act_pos, action_valid, num_lt, num_vc)`` where
        ``act_pos`` holds RoPE offsets relative to the last prefix position and
        ``action_valid`` marks which action keys may be attended (inactive slots are
        key-masked). ``num_vc`` is the width of the Arm V4 camera-context block, which the
        callers need for the "context cannot read actions" row rule and the readout slice.

        ``time_fuse`` says whether ``actions`` is the stream being DENOISED at ``times``
        (true for the two latent-denoise callers) or a block of already-integrated CLEAN
        latents carried as context (false for expert_action_step_multiview, where ``times``
        refers to the robot-action step, not to these latents). It is keyword-only and has
        NO default on purpose: a future caller must answer the question rather than silently
        skip the pi0 time fusion, which is the exact silent-skew the old construction-time
        guard existed to prevent.
        """
        expert = self.action_expert
        B, K, chunk, ldim = actions.shape
        device = actions.device

        # Action-stream input: (B, K*chunk, ldim) -> narrow proj -> (B, K*chunk, H).
        xt_flat = actions.reshape(B, K * chunk, ldim)
        action_hidden = self.latent_in_proj(xt_flat.type(self.dtype))
        cond = expert.embed_time(times) if self.time_conditioning == "adaln" else None
        if cond is not None and ee_bias is not None:
            cond = cond + ee_bias

        # pi0-style concat time conditioning, the multiview twin of _forward_expert:2266.
        # Placed here so it sees ONLY the noisy latent embeddings -- before the slot-seed
        # bias, before the per-chunk positions, and before the camera/foresight blocks are
        # concatenated on. That mirrors the direct path, where the fusion is applied to
        # action_in_proj(x_t) alone and explicitly "before the foresight prepend".
        # (With use_view_cond_slots=True -- what stage-1 runs -- the additive slot bias below
        # is off anyway, so the only ordering that is actually exercised is this one.)
        # This is what lets stage-1 pretrain the SAME pipeline xl_lingbot was validated on:
        # without it the 541 transferable expert tensors are trained under AdaLN-only time
        # conditioning and then fine-tuned with an extra fusion MLP in front of the stack.
        if time_fuse and getattr(expert, "time_concat", False):
            action_hidden = expert.maybe_time_concat(action_hidden, times)

        # Slot binding: add slot k's camera-content seed to its chunk block (per-sample,
        # replaces the old fixed 6-role embedding lookup). None -> slots stay symmetric.
        #
        # Arm V4 SUPERSEDES this additive form (hence the view_cond guard): both feed the
        # same slot_seed_proj, but V4 emits the result as its own context token instead of
        # a per-block bias. Running both would give slot_seed_proj two different jobs in
        # stage 1 and only one in fine-tuning, which is exactly the transfer mismatch V4
        # exists to remove -- the direct path has no per-slot action block to add a bias to,
        # so the context-token form is the only form the two stages can share.
        if slot_seeds is not None and view_cond[0] is None:
            seed_h = self.slot_seed_proj(self.slot_seed_norm(slot_seeds.type(self.dtype)))  # (B, K, H)
            action_hidden = action_hidden + seed_h.repeat_interleave(chunk, dim=1)

        # Per-chunk sinusoidal position (same 0..chunk-1 for every slot; slots are
        # parallel views of the same time window).
        pos = expert.embed_action_positions(chunk, device, self.dtype)  # (1, chunk, H)
        action_hidden = action_hidden + pos.repeat(1, K, 1)             # (1, K*chunk, H)

        # Arm V4 camera-role context block, between foresight and the slot blocks so the
        # layout is [foresight | V cameras | slot0 chunk | slot1 chunk | ...] -- the same
        # ordering the direct path's _forward_expert uses, which is what makes a
        # stage-1-pretrained slot_seed_proj mean the same thing in both.
        vc_tokens, vc_valid = view_cond
        num_vc = 0
        if vc_tokens is not None:
            num_vc = vc_tokens.size(1)
            action_hidden = torch.cat([vc_tokens.type(action_hidden.dtype), action_hidden], dim=1)

        num_lt = expert.num_foresight_tokens
        if num_lt > 0:
            lt_emb = expert.learnable_tokens_in_proj(expert.learnable_tokens)
            lt_emb = lt_emb.unsqueeze(0).expand(B, -1, -1)
            action_hidden = torch.cat([lt_emb, action_hidden], dim=1)
        A = action_hidden.size(1)

        # Action RoPE positions: foresight -> 1..num_lt; cameras -> 1..num_vc; each slot's
        # chunk -> 1..chunk (parallel views share temporal positions). Every suffix block
        # restarts at 1 -- that is this path's existing convention (the K slot blocks
        # already collide with each other and with foresight by construction; blocks are
        # told apart by content, not by RoPE phase). All offset by last prefix position.
        fore_pos = torch.arange(1, num_lt + 1, device=device) if num_lt > 0 else torch.zeros(0, device=device)
        vc_pos = torch.arange(1, num_vc + 1, device=device) if num_vc > 0 else torch.zeros(0, device=device)
        slot_pos = torch.arange(1, chunk + 1, device=device).repeat(K)
        act_pos = torch.cat([fore_pos.to(torch.long), vc_pos.to(torch.long), slot_pos]).to(position_dtype)

        # Per-key validity: inactive-slot masking (foresight always valid; camera roles that
        # no camera landed in are key-masked, so an empty role is never read as a zero token).
        if slot_mask is not None:
            slot_valid = slot_mask.to(torch.bool).repeat_interleave(chunk, dim=1)  # (B, K*chunk)
        else:
            slot_valid = torch.ones(B, K * chunk, dtype=torch.bool, device=device)
        valid_parts = []
        if num_lt > 0:
            valid_parts.append(torch.ones(B, num_lt, dtype=torch.bool, device=device))
        if num_vc > 0:
            valid_parts.append(vc_valid.to(slot_valid.device))
        if valid_parts:
            action_valid = torch.cat(valid_parts + [slot_valid], dim=1)
        else:
            action_valid = slot_valid
        assert action_valid.size(1) == A
        return action_hidden, cond, act_pos, action_valid, num_lt, num_vc

    def prefill_prefix_multiview(
        self,
        prefix_embeds,
        prefix_position_ids,
        visual_pos_masks,
        deepstack_visual_embeds,
        attention_mask,
    ):
        """Run the VLM prefix stream once and cache its per-layer post-RoPE K/V.

        Valid because the joint mask in ``_forward_expert_multiview`` never lets prefix
        tokens attend to action tokens: the prefix stream is independent of ``x_t`` and
        ``t``, so one prefill serves every denoise step instead of re-running the vision
        tower and the LLM per step.
        """
        B, P, _ = prefix_embeds.shape
        device, dtype = prefix_embeds.device, prefix_embeds.dtype
        cos, sin = self.language_model.rotary_emb(prefix_embeds, prefix_position_ids)

        idx = torch.arange(P, device=device)
        allow = (idx[:, None] >= idx[None, :]).unsqueeze(0).expand(B, P, P)
        prefix_valid = attention_mask.to(torch.bool) if attention_mask is not None \
            else torch.ones(B, P, dtype=torch.bool, device=device)
        allow = allow & prefix_valid[:, None, :]
        attn_bias = torch.zeros(B, 1, P, P, dtype=dtype, device=device)
        attn_bias.masked_fill_(~allow.unsqueeze(1), torch.finfo(dtype).min)

        head_dim = self.language_model.layers[0].self_attn.head_dim
        vlm_hidden = prefix_embeds
        sf_mid_hidden = None
        lb_hidden = None
        kv_cache = []
        for layer_idx in range(self.config.text_config.num_hidden_layers):
            vlm_layer = self.language_model.layers[layer_idx]
            vlm_attn = vlm_layer.self_attn

            v_normed = vlm_layer.input_layernorm(vlm_hidden)
            qv = vlm_attn.q_norm(vlm_attn.q_proj(v_normed).view(B, P, -1, head_dim)).transpose(1, 2)
            kv = vlm_attn.k_norm(vlm_attn.k_proj(v_normed).view(B, P, -1, head_dim)).transpose(1, 2)
            vv = vlm_attn.v_proj(v_normed).view(B, P, -1, head_dim).transpose(1, 2)
            qv, kv = apply_rotary_pos_emb(qv, kv, cos, sin)
            n_rep = qv.size(1) // kv.size(1)
            attn_out = F.scaled_dot_product_attention(
                qv, repeat_kv(kv, n_rep), repeat_kv(vv, n_rep),
                attn_mask=attn_bias.to(qv.dtype), scale=vlm_attn.scaling,
            ).transpose(1, 2)

            vlm_hidden = vlm_hidden + vlm_attn.o_proj(attn_out.reshape(B, P, -1))
            vlm_hidden = vlm_hidden + vlm_layer.mlp(vlm_layer.post_attention_layernorm(vlm_hidden))
            if deepstack_visual_embeds is not None and layer_idx < len(deepstack_visual_embeds):
                vlm_hidden = self.language_model._deepstack_process(
                    vlm_hidden, visual_pos_masks, deepstack_visual_embeds[layer_idx]
                )
            if self.use_sf_align and layer_idx + 1 == self.sf_align_layer:
                sf_mid_hidden = vlm_hidden
            if self.use_lb_align and layer_idx + 1 == self.lb_align_layer:
                lb_hidden = (
                    self.language_model.norm(vlm_hidden)
                    if layer_idx + 1 == self.config.text_config.num_hidden_layers
                    else vlm_hidden
                )
            kv_cache.append((kv, vv))

        return {
            "kv_cache": kv_cache,
            "prefix_len": P,
            "last_position": prefix_position_ids[:, :, -1:],
            "prefix_valid": prefix_valid,
            "hidden": vlm_hidden,
            "sf_mid_hidden": sf_mid_hidden,
            "lb_hidden": lb_hidden,
            "visual_pos_masks": visual_pos_masks,
        }

    def expert_step_multiview(self, prefix_state, actions, times, slot_mask=None, ee_bias=None):
        """One denoise step of the K-slot expert against a cached prefix.

        Numerically equivalent to ``_forward_expert_multiview`` (same mask, RoPE and
        projections) but skips the prefix stream entirely.
        """
        expert = self.action_expert
        B, K, chunk, ldim = actions.shape
        P = prefix_state["prefix_len"]
        device = actions.device
        dtype = prefix_state["hidden"].dtype
        last_position = prefix_state["last_position"]

        action_hidden, cond, act_pos, action_valid, num_lt, num_vc = self._build_action_stream_multiview(
            actions, times, slot_mask, ee_bias, last_position.dtype,
            slot_seeds=prefix_state.get("slot_seeds"),
            view_cond=prefix_state.get("view_cond", (None, None)),
            time_fuse=True,   # `actions` IS the latent stream being denoised at `times`
        )
        A = action_hidden.size(1)

        action_position_ids = act_pos.view(1, 1, A).repeat(3, B, 1) + last_position
        cos, sin = self.language_model.rotary_emb(action_hidden, action_position_ids)

        # Action queries see every valid prefix and action key, minus the hidden foresight
        # tokens -- the action rows of the joint mask.
        allow = torch.ones(B, A, P + A, dtype=torch.bool, device=device)
        if num_lt > 0:
            allow[:, num_lt:, P + self.action_expert.foresight_vis_offset:P + num_lt] = False
        if num_vc > 0:
            # V4 camera tokens are context: read by the slots, never reading them back
            # (pi0's state-block rule, as in expert_action_step_multiview).
            allow[:, num_lt:num_lt + num_vc, P + num_lt + num_vc:] = False
        key_valid = torch.cat([prefix_state["prefix_valid"], action_valid], dim=1)
        allow = allow & key_valid[:, None, :]
        attn_bias = torch.zeros(B, 1, A, P + A, dtype=dtype, device=device)
        attn_bias.masked_fill_(~allow.unsqueeze(1), torch.finfo(dtype).min)

        for layer_idx in range(self.config.text_config.num_hidden_layers):
            exp_layer = expert.layers[layer_idx]
            vlm_attn = self.language_model.layers[layer_idx].self_attn
            k_prefix, v_prefix = prefix_state["kv_cache"][layer_idx]

            qa, ka, va = exp_layer.compute_qkv(action_hidden, cond)
            qa, ka, va = qa.transpose(1, 2), ka.transpose(1, 2), va.transpose(1, 2)
            qa, ka = apply_rotary_pos_emb(qa, ka, cos, sin)
            k = torch.cat([k_prefix, ka], dim=2)
            v = torch.cat([v_prefix, va], dim=2)
            n_rep = qa.size(1) // k.size(1)
            attn_out = F.scaled_dot_product_attention(
                qa, repeat_kv(k, n_rep), repeat_kv(v, n_rep),
                attn_mask=attn_bias.to(qa.dtype), scale=vlm_attn.scaling,
            ).transpose(1, 2)
            action_hidden = exp_layer.apply_output(action_hidden, attn_out, cond)

        action_hidden = expert.final(action_hidden, cond)
        _drop = num_lt + num_vc
        if _drop > 0:
            action_hidden = action_hidden[:, _drop:]
        return self.latent_action_head(action_hidden).reshape(B, K, chunk, ldim)

    def _forward_expert_multiview(
        self,
        prefix_embeds,
        prefix_position_ids,
        visual_pos_masks,
        deepstack_visual_embeds,
        attention_mask,
        actions,
        times,
        targets,
        slot_mask,
        rope_deltas,
        depth_target=None,
        depth_mask=None,
        future_depth_target=None,
        future_depth_mask=None,
        ee_bias=None,
        slot_seeds=None,
        view_cond=(None, None),
        image_grid_thw=None,
        teacher_images=None,
        teacher_image_grid_indices=None,
        primary_teacher_indices=None,
        lb_query_mask=None,
    ):
        """K-slot latent forward: the expert emits one latent-action chunk per view slot.

        The action stream is ``[foresight(num_lt) | slot0: chunk | slot1: chunk | ...]``.
        Each slot block carries the content seed pooled from its camera's visual tokens
        (so slots bind to views without a fixed role vocabulary) plus a per-chunk
        sinusoidal position embedding. Inactive slots (``slot_mask``
        False) are key-masked (not attended by anyone) and loss-masked, so they receive no
        gradient. Output is ``(B, K, chunk, latent_dim)``.

        Args:
            actions: noised latents ``x_t`` of shape (B, K, chunk, latent_dim).
            targets: flow-matching velocity targets, same shape.
            slot_mask: (B, K) bool, which slots are active for each sample.
        """
        expert = self.action_expert
        B, K, chunk, ldim = actions.shape
        P = prefix_embeds.size(1)
        device = prefix_embeds.device
        dtype = prefix_embeds.dtype

        action_hidden, cond, act_pos, action_valid, num_lt, num_vc = self._build_action_stream_multiview(
            actions, times, slot_mask, ee_bias, prefix_position_ids.dtype, slot_seeds=slot_seeds,
            view_cond=view_cond,
            time_fuse=True,   # training forward: `actions` is the noised latent target
        )
        A = action_hidden.size(1)  # num_lt + num_vc + K*chunk
        L = P + A

        action_position_ids = act_pos.view(1, 1, A).repeat(3, B, 1) + prefix_position_ids[:, :, -1:]
        full_position_ids = torch.cat([prefix_position_ids, action_position_ids], dim=-1)
        cos, sin = self.language_model.rotary_emb(prefix_embeds, full_position_ids)

        # Joint attention mask (B, L, L): prefix causal; action sees all prefix + all action;
        # foresight visibility (only foresight[0] visible); inactive-slot key-masking.
        idx = torch.arange(P, device=device)
        allow = torch.zeros(L, L, dtype=torch.bool, device=device)
        allow[:P, :P] = idx[:, None] >= idx[None, :]
        allow[P:, :] = True
        if num_lt > 0:
            allow[P + num_lt:, P + self.action_expert.foresight_vis_offset:P + num_lt] = False
        if num_vc > 0:
            # V4 camera tokens are context (cannot read the slot/action tokens). Their KEY
            # side (empty roles) is handled by action_valid -> key_valid below.
            _v0, _v1 = P + num_lt, P + num_lt + num_vc
            allow[_v0:_v1, _v1:] = False
        allow = allow.unsqueeze(0).expand(B, L, L).clone()

        prefix_valid = attention_mask.to(torch.bool) if attention_mask is not None \
            else torch.ones(B, P, dtype=torch.bool, device=device)
        key_valid = torch.cat([prefix_valid, action_valid], dim=1)  # (B, L)
        allow = allow & key_valid[:, None, :]
        attn_bias = torch.zeros(B, 1, L, L, dtype=dtype, device=device)
        attn_bias.masked_fill_(~allow.unsqueeze(1), torch.finfo(dtype).min)

        head_dim = self.language_model.layers[0].self_attn.head_dim
        vlm_hidden = prefix_embeds
        sf_mid_hidden = None
        lb_hidden = None
        for layer_idx in range(self.config.text_config.num_hidden_layers):
            vlm_layer = self.language_model.layers[layer_idx]
            exp_layer = expert.layers[layer_idx]
            vlm_attn = vlm_layer.self_attn

            v_normed = vlm_layer.input_layernorm(vlm_hidden)
            qv = vlm_attn.q_norm(vlm_attn.q_proj(v_normed).view(B, P, -1, head_dim))
            kv = vlm_attn.k_norm(vlm_attn.k_proj(v_normed).view(B, P, -1, head_dim))
            vv = vlm_attn.v_proj(v_normed).view(B, P, -1, head_dim)

            qa, ka, va = exp_layer.compute_qkv(action_hidden, cond)

            q = torch.cat([qv, qa], dim=1).transpose(1, 2)
            k = torch.cat([kv, ka], dim=1).transpose(1, 2)
            v = torch.cat([vv, va], dim=1).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            n_rep = q.size(1) // k.size(1)
            k = repeat_kv(k, n_rep)
            v = repeat_kv(v, n_rep)
            attn_out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_bias.to(q.dtype), scale=vlm_attn.scaling
            ).transpose(1, 2)
            av = attn_out[:, :P]
            aa = attn_out[:, P:]

            vlm_hidden = vlm_hidden + vlm_attn.o_proj(av.reshape(B, P, -1))
            residual = vlm_hidden
            vlm_hidden = residual + vlm_layer.mlp(vlm_layer.post_attention_layernorm(vlm_hidden))
            if deepstack_visual_embeds is not None and layer_idx < len(deepstack_visual_embeds):
                vlm_hidden = self.language_model._deepstack_process(
                    vlm_hidden, visual_pos_masks, deepstack_visual_embeds[layer_idx]
                )
            if self.use_sf_align and layer_idx + 1 == self.sf_align_layer:
                sf_mid_hidden = vlm_hidden
            if self.use_lb_align and layer_idx + 1 == self.lb_align_layer:
                lb_hidden = (
                    self.language_model.norm(vlm_hidden)
                    if layer_idx + 1 == self.config.text_config.num_hidden_layers
                    else vlm_hidden
                )
            action_hidden = exp_layer.apply_output(action_hidden, aa, cond)

        action_hidden = expert.final(action_hidden, cond)
        _drop = num_lt + num_vc
        if _drop > 0:
            action_hidden = action_hidden[:, _drop:]  # (B, K*chunk, H)

        # Per-slot latent output -> (B, K, chunk, latent_dim).
        pred = self.latent_action_head(action_hidden)
        pred_actions = pred.reshape(B, K, chunk, ldim)

        loss = None
        if targets is not None:
            # Uniform flow-matching MSE (matches pi0 and the single-stream reference;
            # no (1-t) time weighting, which under-trains the high-t region where
            # inference-time integration starts).
            mse = F.mse_loss(pred_actions.type(targets.dtype), targets, reduction='none')
            per_slot = mse.mean(dim=(2, 3))  # (B, K)
            if slot_mask is not None:
                m = slot_mask.to(per_slot.dtype)
                per_sample = (per_slot * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
            else:
                per_sample = per_slot.mean(dim=1)
            loss = per_sample.mean()

        if self.use_depth_aux and depth_target is not None:
            depth_loss = self._compute_depth_loss(
                vlm_hidden, visual_pos_masks,
                depth_target, depth_mask, future_depth_target, future_depth_mask,
            )
            if depth_loss is not None:
                loss = depth_loss * self.depth_loss_weight if loss is None else loss + self.depth_loss_weight * depth_loss

        if self.training and self.use_sf_align:
            sf_loss = self._compute_sf_align_loss(
                sf_mid_hidden,
                visual_pos_masks,
                image_grid_thw,
                teacher_images,
                teacher_image_grid_indices,
            )
            if sf_loss is not None:
                loss = sf_loss * self.sf_align_weight if loss is None else loss + self.sf_align_weight * sf_loss

        if self.training and self.use_lb_align:
            lb_loss = self._compute_lb_align_loss(
                lb_hidden,
                visual_pos_masks,
                image_grid_thw,
                teacher_images,
                lb_query_mask,
                teacher_image_grid_indices,
                primary_teacher_indices,
            )
            loss = lb_loss * self.lb_align_weight if loss is None else loss + self.lb_align_weight * lb_loss

        return RynnBrainVLAModelOutputWithPast(
            last_hidden_state=vlm_hidden,
            past_key_values=None,
            rope_deltas=rope_deltas,
            actions=pred_actions,
            loss=loss,
        )

    def prefill_prefix_from_inputs(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        states=None,
        ee_type_id=None,
        history_states=None,
        history_actions=None,
        **unused,
    ):
        """Embed the prefix from raw processor inputs and prefill its per-layer K/V.

        Returns ``(prefix_state, ee_bias)``. Keys unrelated to the prefix (loss targets,
        slot ids, ...) are ignored so callers can forward a whole batch.
        """
        camera_slot_ids = unused.get("camera_slot_ids")
        inputs_embeds, position_ids, visual_pos_masks, deepstack_visual_embeds = self._embed_prefix(
            input_ids=input_ids,
            inputs_embeds=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=None,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            states=states,
            actions=None,
            history_states=history_states,
            history_actions=history_actions,
            camera_slot_ids=camera_slot_ids,
        )
        ee_bias = self.ee_embedding(ee_type_id) \
            if (self.use_ee_embedding and ee_type_id is not None) else None
        lb_query_mask = None
        if self.use_lb_align:
            (
                inputs_embeds,
                position_ids,
                visual_pos_masks,
                attention_mask,
                lb_query_mask,
            ) = self._append_align_queries(
                inputs_embeds, position_ids, visual_pos_masks, attention_mask, input_ids
            )
        prefix_state = self.prefill_prefix_multiview(
            prefix_embeds=inputs_embeds,
            prefix_position_ids=position_ids,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            attention_mask=attention_mask,
        )
        prefix_state["lb_query_mask"] = lb_query_mask
        # (2026-08-28) camera_slot_ids now carries constants.VIEW_ROLES role ids, not packing
        # positions, so max(id)+1 no longer means "how many slots": an episode whose only
        # camera is `side` (role 4) would ask for 5 slots, and one holding only `head` would
        # ask for 1 and drop every other role. The slot axis is the fixed role table, so its
        # width is num_view_slots; an explicit num_views still wins for callers that know
        # better.
        num_views = unused.get("num_views")
        if num_views is None:
            num_views = self.num_view_slots
        prefix_state["slot_seeds"] = self._pool_view_seeds(
            inputs_embeds, visual_pos_masks, image_grid_thw, camera_slot_ids,
            num_slots=num_views,
        )
        # Arm V4: a pure function of the prefix, so build once here and reuse for every
        # denoise step (same reason the direct path stashes it at prefill).
        prefix_state["view_cond"] = self._build_view_cond_tokens(
            inputs_embeds, visual_pos_masks, image_grid_thw, camera_slot_ids
        )
        return prefix_state, ee_bias

    def integrate_latents_cached(self, prefix_state, num_steps=10, chunk=None, slot_mask=None, ee_bias=None):
        """Integrate the K-slot latent chunk from noise using an already-prefilled prefix."""
        B = prefix_state["prefix_valid"].size(0)
        seeds = prefix_state.get("slot_seeds")
        K = seeds.size(1) if seeds is not None else self.num_view_slots
        chunk = chunk if chunk is not None else getattr(
            self.config, "latent_action_chunk_size", self.config.action_chunk_size
        )
        x = self.sample_noise((B, K, chunk, self.latent_action_dim))
        dt = -1.0 / num_steps
        times = torch.ones(B, dtype=torch.float32, device=x.device)
        for _ in range(num_steps):
            velocity = self.expert_step_multiview(
                prefix_state, x, times, slot_mask=slot_mask, ee_bias=ee_bias
            )
            x = x + dt * velocity
            times = times + dt
        return x

    @torch.no_grad()
    def generate_latents_from_inputs(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        states=None,
        slot_mask=None,
        ee_type_id=None,
        camera_slot_ids=None,
        num_steps: int = 10,
        chunk: int = None,
    ):
        """Deployment entry: raw processor inputs -> integrated K-slot latent chunk.

        Embeds the prefix (vision tower + scatter) once for the whole denoise loop.
        """
        prefix_state, ee_bias = self.prefill_prefix_from_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            states=states,
            ee_type_id=ee_type_id,
            camera_slot_ids=camera_slot_ids,
        )
        return self.integrate_latents_cached(
            prefix_state, num_steps=num_steps, chunk=chunk, slot_mask=slot_mask, ee_bias=ee_bias
        )

    def _readout_action(self, hidden):
        """Direct-path action readout at the action-stream width.

        Default: the pi0-style thin linear (action_out_proj) -- unchanged behavior.
        fix-v7 (use_latent_head_readout): route through the stage-1-pretrained
        latent_action_head (hidden -> hidden -> SiLU -> latent_dim) plus the fresh
        latent_readout_proj (latent_dim -> action_dim). Same output shape as
        action_out_proj, so every call site is a drop-in swap.
        """
        if self.use_latent_head_readout:
            return self.latent_readout_proj(self.latent_action_head(hidden))
        return self.action_out_proj(hidden)

    def expert_action_step_multiview(self, prefix_state, latent_slots, x_t, times, slot_mask=None, ee_bias=None):
        """One robot-action denoising step against a cached prefix.

        Suffix layout is ``[foresight | K-slot latent context | n_act noisy actions]``.
        The context block is built by the SAME ``_build_action_stream_multiview`` used
        for latent denoising (slot seeds, per-chunk positions), but carries the CLEAN
        integrated latents; the action block uses the direct path's conventions
        (``action_in_proj`` + sinusoidal step positions + RoPE 1..n_act).

        Attention follows lingbot-vla pi0's suffix rules (modeling_lingbot_vla.py:1656-1662):
        the context block cannot attend the action tokens (pi0's state block), action
        tokens attend the full prefix + context + each other bidirectionally, and one
        time condition modulates the whole suffix (pi0 ``ada_cond``, L1722). The
        foresight[1:] hiding matches the latent step.

        Args:
            latent_slots: (B, K, chunk, latent_dim) clean latent estimate.
            x_t:          (B, n_act, action_dim) noisy robot actions.
            times:        (B,) flow time for the ACTION flow.
        Returns velocity (B, n_act, action_dim).
        """
        expert = self.action_expert
        B = x_t.size(0)
        n_act = x_t.size(1)
        P = prefix_state["prefix_len"]
        device = x_t.device
        dtype = prefix_state["hidden"].dtype
        last_position = prefix_state["last_position"]

        # num_vc is unused here on purpose: the V4 camera tokens land INSIDE ctx_hidden, so
        # the existing "allow[:, :C, P+C:] = False" already applies the context rule to them
        # and the readout is hidden[:, -n_act:] -- nothing downstream needs the width.
        ctx_hidden, cond, ctx_pos, ctx_valid, num_lt, _num_vc = self._build_action_stream_multiview(
            latent_slots, times, slot_mask, ee_bias, last_position.dtype,
            slot_seeds=prefix_state.get("slot_seeds"),
            view_cond=prefix_state.get("view_cond", (None, None)),
            # FALSE: latent_slots here are the CLEAN integrated latents used as context, and
            # `times` is the ROBOT-ACTION denoise step. Fusing that time into a clean context
            # block would be a semantic error, not just a redundant op.
            time_fuse=False,
        )
        C = ctx_hidden.size(1)  # num_lt + num_vc + K*chunk

        act_hidden = self.action_in_proj(x_t.type(self.dtype))
        # THIS is the block `times` belongs to, so it gets the fusion the context block above
        # deliberately skipped. Same order as the direct path: fuse, then add positions.
        if getattr(expert, "time_concat", False):
            act_hidden = expert.maybe_time_concat(act_hidden, times)
        act_hidden = act_hidden + expert.embed_action_positions(n_act, device, act_hidden.dtype)
        act_pos = torch.arange(1, n_act + 1, device=device).to(last_position.dtype)

        suffix_hidden = torch.cat([ctx_hidden, act_hidden], dim=1)
        A = C + n_act
        position_ids = torch.cat([ctx_pos, act_pos]).view(1, 1, A).repeat(3, B, 1) + last_position
        cos, sin = self.language_model.rotary_emb(suffix_hidden, position_ids)

        # Rows: suffix queries; cols: prefix + suffix keys.
        allow = torch.ones(B, A, P + A, dtype=torch.bool, device=device)
        if num_lt > 0:
            allow[:, num_lt:, P + self.action_expert.foresight_vis_offset:P + num_lt] = False  # hide foresight[1:], as in the latent step
        allow[:, :C, P + C:] = False  # context block cannot see action tokens (pi0 state rule)
        act_valid = torch.ones(B, n_act, dtype=torch.bool, device=device)
        key_valid = torch.cat([prefix_state["prefix_valid"], ctx_valid, act_valid], dim=1)
        allow = allow & key_valid[:, None, :]
        attn_bias = torch.zeros(B, 1, A, P + A, dtype=dtype, device=device)
        attn_bias.masked_fill_(~allow.unsqueeze(1), torch.finfo(dtype).min)

        hidden = suffix_hidden
        for layer_idx in range(self.config.text_config.num_hidden_layers):
            exp_layer = expert.layers[layer_idx]
            vlm_attn = self.language_model.layers[layer_idx].self_attn
            k_prefix, v_prefix = prefix_state["kv_cache"][layer_idx]

            qa, ka, va = exp_layer.compute_qkv(hidden, cond)
            qa, ka, va = qa.transpose(1, 2), ka.transpose(1, 2), va.transpose(1, 2)
            qa, ka = apply_rotary_pos_emb(qa, ka, cos, sin)
            k = torch.cat([k_prefix, ka], dim=2)
            v = torch.cat([v_prefix, va], dim=2)
            n_rep = qa.size(1) // k.size(1)
            attn_out = F.scaled_dot_product_attention(
                qa, repeat_kv(k, n_rep), repeat_kv(v, n_rep),
                attn_mask=attn_bias.to(qa.dtype), scale=vlm_attn.scaling,
            ).transpose(1, 2)
            hidden = exp_layer.apply_output(hidden, attn_out, cond)

        hidden = expert.final(hidden, cond)
        return self._readout_action(hidden[:, -n_act:])

    def integrate_actions_cached(self, prefix_state, latent_slots, num_steps=10, slot_mask=None, ee_bias=None):
        """Euler-integrate the robot-action chunk from noise (fix4), given clean latents."""
        B = latent_slots.size(0)
        x = self.sample_noise((B, self.config.action_chunk_size, self.config.action_dim))
        dt = -1.0 / num_steps
        times = torch.ones(B, dtype=torch.float32, device=x.device)
        for _ in range(num_steps):
            velocity = self.expert_action_step_multiview(
                prefix_state, latent_slots, x, times, slot_mask=slot_mask, ee_bias=ee_bias
            )
            x = x + dt * velocity
            times = times + dt
        return x

    def _forward_expert(
        self,
        prefix_embeds,
        prefix_position_ids,
        visual_pos_masks,
        deepstack_visual_embeds,
        attention_mask,
        actions,
        times,
        targets,
        action_mask,
        rope_deltas,
        past_key_values=None,
        cache_position=None,
        depth_target=None,
        depth_mask=None,
        future_depth_target=None,
        future_depth_mask=None,
        ee_bias=None,
        image_grid_thw=None,
        camera_slot_ids=None,
        teacher_images=None,
        teacher_image_grid_indices=None,
        primary_teacher_indices=None,
        lb_query_mask=None,
    ):
        """Two-stream lockstep forward with the independent action expert.

        Dispatches to prefill / decode when a KV cache is provided (inference), else
        runs the joint training lockstep. In training there is no cache: the VLM and
        expert compute their own Q/K/V, concatenated for one joint attention (prefix
        causal, action bidirectional), then each stream applies its own output/MLP.
        The action stream has a narrow residual width.
        """
        if past_key_values is not None and actions is None:
            # Arm V4: the camera tokens are a pure function of the PREFIX, so build them once
            # here and stash them for the decode steps -- by then prefix_embeds is gone (only
            # per-layer K/V survives in the cache). Same lifetime as HF's self.rope_deltas.
            self._view_cond_cache = self._build_view_cond_tokens(
                prefix_embeds, visual_pos_masks, image_grid_thw, camera_slot_ids
            )
            return self._expert_prefill(
                prefix_embeds, prefix_position_ids, visual_pos_masks,
                deepstack_visual_embeds, attention_mask, past_key_values, rope_deltas,
            )
        if past_key_values is not None and actions is not None:
            return self._expert_decode(
                actions, times, prefix_position_ids, past_key_values, rope_deltas, ee_bias=ee_bias,
                view_cond=getattr(self, "_view_cond_cache", (None, None)),
            )

        expert = self.action_expert
        B = prefix_embeds.size(0)
        P = prefix_embeds.size(1)
        A = actions.size(1)
        device = prefix_embeds.device

        # Action-stream input embeddings (narrow width) + AdaLN time conditioning.
        action_hidden = self.action_in_proj(actions.type(self.dtype))
        cond = expert.embed_time(times) if self.time_conditioning == "adaln" else None
        if cond is not None and ee_bias is not None:
            cond = cond + ee_bias
        # pi0-style concat time conditioning: MLP([action_emb ; time_emb]) (openpi
        # pi0.py:172-178, lingbot :475, InternVLA :564). Applied to the ACTION tokens
        # only -- before the foresight prepend -- matching InternVLA's suffix layout
        # "[state] [learnable] [action_time]" (:918). No-op unless expert_time_concat.
        if getattr(expert, "time_concat", False):
            action_hidden = expert.maybe_time_concat(action_hidden, times)

        # Foresight tokens: prepend learnable tokens to action stream (InternVLA style)
        if expert.num_foresight_tokens > 0:
            lt_emb = expert.learnable_tokens_in_proj(expert.learnable_tokens)  # (num_lt, hidden)
            lt_emb = lt_emb.unsqueeze(0).expand(B, -1, -1)  # (B, num_lt, hidden)
            action_hidden = torch.cat([lt_emb, action_hidden], dim=1)  # (B, num_lt + A, hidden)
            A = expert.num_foresight_tokens + A  # Update A to include foresight tokens

        # Sinusoidal position encoding: add to action_hidden for better temporal structure
        action_pos_emb = expert.embed_action_positions(A, device, self.dtype)  # (1, A, hidden)
        action_hidden = action_hidden + action_pos_emb  # (B, A, hidden)

        # Action position ids: 1..A offset by the last prefix position (3D M-RoPE).
        act_pos = torch.arange(1, A + 1, dtype=prefix_position_ids.dtype, device=device)

        # Arm V4: per-camera context tokens, spliced between foresight and actions so the
        # layout is [foresight | V cameras | actions] -- the same slot ordering fix4's
        # expert_action_step_multiview and _build_action_stream_multiview use.
        #
        # Deliberately AFTER embed_action_positions / act_pos, for two reasons:
        #   1. the foresight and action tokens then carry exactly the content and RoPE
        #      phases they carry with the flag off, so V4 adds keys rather than also
        #      re-indexing the stream;
        #   2. the camera token is then plain slot_seed_proj(norm(pool(camera_k))) with no
        #      sinusoidal offset -- byte-identical to what the latent path builds. That
        #      sameness is the whole transfer argument: a stage-1-pretrained slot_seed_proj
        #      is fed the same input and read in the same position here.
        # Camera RoPE positions restart at 1 (this path's per-block convention, shared with
        # the latent stream where every slot block also restarts at 1).
        vc_tokens, vc_valid = self._build_view_cond_tokens(
            prefix_embeds, visual_pos_masks, image_grid_thw, camera_slot_ids
        )
        num_vc = 0
        if vc_tokens is not None:
            num_vc = vc_tokens.size(1)
            _lt = expert.num_foresight_tokens
            action_hidden = torch.cat(
                [action_hidden[:, :_lt], vc_tokens.type(action_hidden.dtype), action_hidden[:, _lt:]],
                dim=1,
            )
            vc_pos = torch.arange(1, num_vc + 1, dtype=act_pos.dtype, device=device)
            act_pos = torch.cat([act_pos[:_lt], vc_pos, act_pos[_lt:]])
            A = A + num_vc

        # Total sequence length (computed AFTER the foresight/camera expansion so the mask matches).
        L = P + A

        prefix_last_position = _global_last_position(prefix_position_ids, attention_mask)
        action_position_ids = act_pos.view(1, 1, A).repeat(3, B, 1)
        action_position_ids = action_position_ids + prefix_last_position
        full_position_ids = torch.cat([prefix_position_ids, action_position_ids], dim=-1)

        # Shared RoPE over the full [prefix, action] sequence.
        cos, sin = self.language_model.rotary_emb(prefix_embeds, full_position_ids)

        # Joint additive attention mask (B, 1, L, L): prefix rows causal over prefix;
        # action rows attend all prefix + all action (bidirectional); prefix cannot see
        # action; padded prefix key columns are masked for all queries.
        idx = torch.arange(P, device=device)
        allow = torch.zeros(L, L, dtype=torch.bool, device=device)
        allow[:P, :P] = idx[:, None] >= idx[None, :]
        allow[P:, :] = True  # Action/foresight can see all prefix + all action

        # Foresight tokens: only the first foresight token can be attended by action tokens
        if expert.num_foresight_tokens > 0:
            num_lt = expert.num_foresight_tokens
            # Action tokens (P+num_lt to L-1) cannot attend to foresight tokens (P+1 to P+num_lt-1)
            allow[P + num_lt:, P + self.action_expert.foresight_vis_offset:P + num_lt] = False

        # Arm V4 camera tokens sit at [P+num_lt, P+num_lt+num_vc). They are CONTEXT: action
        # tokens read them, they do not read the action tokens (pi0's state-block rule, which
        # fix4's expert_action_step_multiview already follows for its latent context block).
        if num_vc > 0:
            _lt = expert.num_foresight_tokens
            _v0, _v1 = P + _lt, P + _lt + num_vc
            allow[_v0:_v1, _v1:] = False

        allow = allow.unsqueeze(0).expand(B, L, L).clone()
        # Roles with no camera in THIS sample (LIBERO fills only left_wrist + front_third of
        # constants.VIEW_ROLES) are key-masked, so they are never attended as all-zero tokens.
        if num_vc > 0:
            allow[:, :, _v0:_v1] &= vc_valid[:, None, :]
        if attention_mask is not None:
            key_valid = torch.cat(
                [attention_mask.to(torch.bool), torch.ones(B, A, dtype=torch.bool, device=device)],
                dim=1,
            )
            allow = allow & key_valid[:, None, :]
        attn_bias = torch.zeros(B, 1, L, L, dtype=prefix_embeds.dtype, device=device)
        attn_bias.masked_fill_(~allow.unsqueeze(1), torch.finfo(prefix_embeds.dtype).min)

        head_dim = self.language_model.layers[0].self_attn.head_dim
        vlm_hidden = prefix_embeds
        sf_mid_hidden = None
        lb_hidden = None

        for layer_idx in range(self.config.text_config.num_hidden_layers):
            vlm_layer = self.language_model.layers[layer_idx]
            exp_layer = expert.layers[layer_idx]
            vlm_attn = vlm_layer.self_attn

            # VLM stream Q/K/V (with QK-norm), shape (B, P, heads, head_dim).
            v_normed = vlm_layer.input_layernorm(vlm_hidden)
            qv = vlm_attn.q_norm(vlm_attn.q_proj(v_normed).view(B, P, -1, head_dim))
            kv = vlm_attn.k_norm(vlm_attn.k_proj(v_normed).view(B, P, -1, head_dim))
            vv = vlm_attn.v_proj(v_normed).view(B, P, -1, head_dim)

            # Expert stream Q/K/V (narrow input, no QK-norm), shape (B, A, heads, head_dim).
            qa, ka, va = exp_layer.compute_qkv(action_hidden, cond)

            # Concat along sequence -> (B, L, H, head_dim) -> (B, H, L, head_dim).
            q = torch.cat([qv, qa], dim=1).transpose(1, 2)
            k = torch.cat([kv, ka], dim=1).transpose(1, 2)
            v = torch.cat([vv, va], dim=1).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

            n_rep = q.size(1) // k.size(1)
            k = repeat_kv(k, n_rep)
            v = repeat_kv(v, n_rep)

            attn_out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_bias.to(q.dtype), scale=vlm_attn.scaling
            )
            attn_out = attn_out.transpose(1, 2)  # (B, L, H, head_dim)
            av = attn_out[:, :P]
            aa = attn_out[:, P:]

            # VLM stream output: o_proj + residual + post-norm + MLP (+ deepstack).
            vlm_hidden = vlm_hidden + vlm_attn.o_proj(av.reshape(B, P, -1))
            residual = vlm_hidden
            vlm_hidden = residual + vlm_layer.mlp(vlm_layer.post_attention_layernorm(vlm_hidden))
            if deepstack_visual_embeds is not None and layer_idx < len(deepstack_visual_embeds):
                vlm_hidden = self.language_model._deepstack_process(
                    vlm_hidden, visual_pos_masks, deepstack_visual_embeds[layer_idx]
                )
            if self.use_sf_align and layer_idx + 1 == self.sf_align_layer:
                sf_mid_hidden = vlm_hidden
            if self.use_lb_align and layer_idx + 1 == self.lb_align_layer:
                lb_hidden = (
                    self.language_model.norm(vlm_hidden)
                    if layer_idx + 1 == self.config.text_config.num_hidden_layers
                    else vlm_hidden
                )

            # Expert stream output.
            action_hidden = exp_layer.apply_output(action_hidden, aa, cond)

        action_hidden = expert.final(action_hidden, cond)
        # Remove foresight AND Arm-V4 camera tokens before projecting to actions.
        _drop = expert.num_foresight_tokens + num_vc
        if _drop > 0:
            action_hidden = action_hidden[:, _drop:]

        # Hierarchical VLA: return latent actions for an external decoder; direct mode predicts action velocity.
        latent_mode = self.use_latent_actions and self.latent_action_head is not None
        if latent_mode:
            pred_actions = self.latent_action_head(action_hidden)
        else:
            pred_actions = self._readout_action(action_hidden)
            pred_actions = pred_actions.view(actions.shape)

        loss = None
        if targets is not None and not latent_mode:
            # Uniform flow-matching MSE (matches pi0 and the single-stream reference;
            # no (1-t) time weighting, which under-trains the high-t region where
            # inference-time integration starts).
            if action_mask is not None:
                # Per-token MSE loss, then apply mask
                mse_loss = F.mse_loss(pred_actions.type(targets.dtype), targets, reduction='none')
                masked_loss = mse_loss * action_mask.to(mse_loss.dtype)
                per_sample_loss = masked_loss.sum(dim=(1, 2)) / action_mask.sum(dim=(1, 2)).clamp(min=1)
                loss = per_sample_loss.mean()
            else:
                mse_loss = F.mse_loss(pred_actions.type(targets.dtype), targets, reduction='none')
                per_sample_loss = mse_loss.mean(dim=(1, 2))  # (B,)
                loss = per_sample_loss.mean()

        if self.use_depth_aux and depth_target is not None:
            depth_loss = self._compute_depth_loss(
                vlm_hidden, visual_pos_masks,
                depth_target, depth_mask, future_depth_target, future_depth_mask,
            )
            if depth_loss is not None:
                loss = depth_loss * self.depth_loss_weight if loss is None else loss + self.depth_loss_weight * depth_loss

        if self.training and self.use_sf_align:
            sf_loss = self._compute_sf_align_loss(
                sf_mid_hidden,
                visual_pos_masks,
                image_grid_thw,
                teacher_images,
                teacher_image_grid_indices,
            )
            if sf_loss is not None:
                loss = sf_loss * self.sf_align_weight if loss is None else loss + self.sf_align_weight * sf_loss

        if self.training and self.use_lb_align:
            lb_loss = self._compute_lb_align_loss(
                lb_hidden,
                visual_pos_masks,
                image_grid_thw,
                teacher_images,
                lb_query_mask,
                teacher_image_grid_indices,
                primary_teacher_indices,
            )
            loss = lb_loss * self.lb_align_weight if loss is None else loss + self.lb_align_weight * lb_loss

        return RynnBrainVLAModelOutputWithPast(
            last_hidden_state=vlm_hidden,
            past_key_values=None,
            rope_deltas=rope_deltas,
            actions=pred_actions,
            loss=loss,
        )

    def _forward_expert_amortized(
        self,
        prefix_embeds,
        prefix_position_ids,
        visual_pos_masks,
        deepstack_visual_embeds,
        attention_mask,
        actions,
        times,
        targets,
        action_mask,
        rope_deltas,
        train_repeat,
        depth_target=None,
        depth_mask=None,
        future_depth_target=None,
        future_depth_mask=None,
        ee_bias=None,
    ):
        """Training-only amortized two-stream forward.

        The VLM prefix stream is run ONCE (batch B), caching each layer's post-RoPE
        K/V; the narrow expert stream is run for ``N=train_repeat`` noise copies
        (batch N*B) against the cached prefix K/V. This reuses the expensive prefix
        forward across N flow-matching samples. Numerically equivalent (up to fp
        reordering) to calling ``_forward_expert`` N times with the same prefix.

        Inputs: prefix_* are batch B; actions/times/targets/action_mask/ee_bias are
        batch N*B (expanded by the caller).
        """
        if self.use_lb_align:
            raise ValueError(
                "use_lb_align is not supported with expert_train_repeat > 1; "
                "set expert_train_repeat=1 for query-alignment runs."
            )
        expert = self.action_expert
        B = prefix_embeds.size(0)
        N = int(train_repeat)
        NB = actions.size(0)
        assert NB == N * B, f"expected N*B={N*B} action rows, got {NB}"
        P = prefix_embeds.size(1)
        A = actions.size(1)
        device = prefix_embeds.device
        dtype = prefix_embeds.dtype
        neg_inf = torch.finfo(dtype).min

        # ---- action stream input (batch N*B) + AdaLN time cond ----
        action_hidden = self.action_in_proj(actions.type(self.dtype))
        cond = expert.embed_time(times) if self.time_conditioning == "adaln" else None
        if cond is not None and ee_bias is not None:
            cond = cond + ee_bias
        # pi0-style concat time conditioning: MLP([action_emb ; time_emb]) (openpi
        # pi0.py:172-178, lingbot :475, InternVLA :564). Applied to the ACTION tokens
        # only -- before the foresight prepend -- matching InternVLA's suffix layout
        # "[state] [learnable] [action_time]" (:918). No-op unless expert_time_concat.
        if getattr(expert, "time_concat", False):
            action_hidden = expert.maybe_time_concat(action_hidden, times)

        if expert.num_foresight_tokens > 0:
            lt_emb = expert.learnable_tokens_in_proj(expert.learnable_tokens)
            lt_emb = lt_emb.unsqueeze(0).expand(NB, -1, -1)
            action_hidden = torch.cat([lt_emb, action_hidden], dim=1)
            A = expert.num_foresight_tokens + A
        L = P + A

        action_pos_emb = expert.embed_action_positions(A, device, self.dtype)
        action_hidden = action_hidden + action_pos_emb

        # Action position ids (batch B) continue from one valid global prefix maximum.
        action_position_ids = torch.arange(
            1, A + 1, dtype=prefix_position_ids.dtype, device=device
        ).view(1, 1, A).repeat(3, B, 1)
        action_position_ids = action_position_ids + _global_last_position(
            prefix_position_ids, attention_mask
        )

        cos_p, sin_p = self.language_model.rotary_emb(prefix_embeds, prefix_position_ids)
        cos_a, sin_a = self.language_model.rotary_emb(prefix_embeds, action_position_ids)
        cos_a = cos_a.repeat_interleave(N, dim=0)
        sin_a = sin_a.repeat_interleave(N, dim=0)

        head_dim = self.language_model.layers[0].self_attn.head_dim
        scaling = self.language_model.layers[0].self_attn.scaling
        num_layers = self.config.text_config.num_hidden_layers

        # ---- Phase 1: VLM prefix stream (batch B), cache per-layer post-RoPE K/V ----
        idx = torch.arange(P, device=device)
        allow_p = (idx[:, None] >= idx[None, :]).unsqueeze(0).expand(B, P, P).clone()
        if attention_mask is not None:
            key_valid = attention_mask.to(torch.bool)
            allow_p = allow_p & key_valid[:, None, :]
        bias_p = torch.zeros(B, 1, P, P, dtype=dtype, device=device)
        bias_p.masked_fill_(~allow_p.unsqueeze(1), neg_inf)

        prefix_k_cache = []
        prefix_v_cache = []
        vlm_hidden = prefix_embeds
        for layer_idx in range(num_layers):
            vlm_layer = self.language_model.layers[layer_idx]
            vlm_attn = vlm_layer.self_attn
            v_normed = vlm_layer.input_layernorm(vlm_hidden)
            qv = vlm_attn.q_norm(vlm_attn.q_proj(v_normed).view(B, P, -1, head_dim)).transpose(1, 2)
            kv = vlm_attn.k_norm(vlm_attn.k_proj(v_normed).view(B, P, -1, head_dim)).transpose(1, 2)
            vv = vlm_attn.v_proj(v_normed).view(B, P, -1, head_dim).transpose(1, 2)
            qv, kv = apply_rotary_pos_emb(qv, kv, cos_p, sin_p)
            # Cache post-RoPE K/V (kv-heads, pre-repeat) for the expert stream to attend.
            prefix_k_cache.append(kv)
            prefix_v_cache.append(vv)
            n_rep = qv.size(1) // kv.size(1)
            av = F.scaled_dot_product_attention(
                qv, repeat_kv(kv, n_rep), repeat_kv(vv, n_rep),
                attn_mask=bias_p.to(qv.dtype), scale=scaling,
            ).transpose(1, 2)
            vlm_hidden = vlm_hidden + vlm_attn.o_proj(av.reshape(B, P, -1))
            residual = vlm_hidden
            vlm_hidden = residual + vlm_layer.mlp(vlm_layer.post_attention_layernorm(vlm_hidden))
            if deepstack_visual_embeds is not None and layer_idx < len(deepstack_visual_embeds):
                vlm_hidden = self.language_model._deepstack_process(
                    vlm_hidden, visual_pos_masks, deepstack_visual_embeds[layer_idx]
                )

        # ---- Phase 2: expert stream (batch N*B) attending cached prefix K/V ----
        allow_a = torch.ones(B, A, L, dtype=torch.bool, device=device)
        if attention_mask is not None:
            allow_a[:, :, :P] = attention_mask.to(torch.bool)[:, None, :]
        if expert.num_foresight_tokens > 0:
            num_lt = expert.num_foresight_tokens
            allow_a[:, num_lt:, P + self.action_expert.foresight_vis_offset:P + num_lt] = False
        bias_a = torch.zeros(B, 1, A, L, dtype=dtype, device=device)
        bias_a.masked_fill_(~allow_a.unsqueeze(1), neg_inf)
        bias_a = bias_a.repeat_interleave(N, dim=0)

        for layer_idx in range(num_layers):
            exp_layer = expert.layers[layer_idx]
            qa, ka, va = exp_layer.compute_qkv(action_hidden, cond)
            qa = qa.transpose(1, 2)
            ka = ka.transpose(1, 2)
            va = va.transpose(1, 2)
            qa, ka = apply_rotary_pos_emb(qa, ka, cos_a, sin_a)
            kp = prefix_k_cache[layer_idx].repeat_interleave(N, dim=0)
            vp = prefix_v_cache[layer_idx].repeat_interleave(N, dim=0)
            k = torch.cat([kp, ka], dim=2)
            v = torch.cat([vp, va], dim=2)
            n_rep = qa.size(1) // k.size(1)
            aa = F.scaled_dot_product_attention(
                qa, repeat_kv(k, n_rep), repeat_kv(v, n_rep),
                attn_mask=bias_a.to(qa.dtype), scale=scaling,
            ).transpose(1, 2)
            action_hidden = exp_layer.apply_output(action_hidden, aa, cond)

        action_hidden = expert.final(action_hidden, cond)
        if expert.num_foresight_tokens > 0:
            action_hidden = action_hidden[:, expert.num_foresight_tokens:]

        latent_mode = self.use_latent_actions and self.latent_action_head is not None
        if latent_mode:
            pred_actions = self.latent_action_head(action_hidden)
        else:
            pred_actions = self._readout_action(action_hidden)
            pred_actions = pred_actions.view(targets.shape)

        loss = None
        if targets is not None and not latent_mode:
            # Uniform flow-matching MSE (matches pi0 / reference; no (1-t) time weighting).
            mse_loss = F.mse_loss(pred_actions.type(targets.dtype), targets, reduction='none')
            if action_mask is not None:
                masked_loss = mse_loss * action_mask.to(mse_loss.dtype)
                per_sample_loss = masked_loss.sum(dim=(1, 2)) / action_mask.sum(dim=(1, 2)).clamp(min=1)
            else:
                per_sample_loss = mse_loss.mean(dim=(1, 2))
            loss = per_sample_loss.mean()

        # Depth aux is a prefix-side loss, computed once from the single VLM stream (batch B).
        # The action loss above is averaged over N*B samples, so the depth loss (averaged
        # over B) would have N times larger effective weight. Divide by N to keep the
        # intended relative strength set by ``depth_loss_weight``.
        if self.use_depth_aux and depth_target is not None:
            depth_loss = self._compute_depth_loss(
                vlm_hidden, visual_pos_masks,
                depth_target, depth_mask, future_depth_target, future_depth_mask,
            )
            if depth_loss is not None:
                depth_term = self.depth_loss_weight * depth_loss / N
                loss = depth_term if loss is None else loss + depth_term

        return RynnBrainVLAModelOutputWithPast(
            last_hidden_state=vlm_hidden,
            past_key_values=None,
            rope_deltas=rope_deltas,
            actions=pred_actions,
            loss=loss,
        )

    def _expert_prefill(
        self,
        prefix_embeds,
        prefix_position_ids,
        visual_pos_masks,
        deepstack_visual_embeds,
        attention_mask,
        past_key_values,
        rope_deltas,
    ):
        """Prefill for the expert path: run the VLM stream over the prefix and cache
        each layer's post-RoPE K/V (so decode's action queries can attend to them)."""
        B, P = prefix_embeds.shape[:2]
        device = prefix_embeds.device
        cos, sin = self.language_model.rotary_emb(prefix_embeds, prefix_position_ids)
        head_dim = self.language_model.layers[0].self_attn.head_dim

        # Causal (+ padding) prefix mask.
        prefix_valid = (
            attention_mask.to(device=device, dtype=torch.bool)
            if attention_mask is not None
            else torch.ones(B, P, dtype=torch.bool, device=device)
        )
        idx = torch.arange(P, device=device)
        allow = (idx[:, None] >= idx[None, :]).unsqueeze(0).expand(B, P, P).clone()
        allow = allow & prefix_valid[:, None, :]
        attn_bias = torch.zeros(B, 1, P, P, dtype=prefix_embeds.dtype, device=device)
        attn_bias.masked_fill_(~allow.unsqueeze(1), torch.finfo(prefix_embeds.dtype).min)

        vlm_hidden = prefix_embeds
        for layer_idx in range(self.config.text_config.num_hidden_layers):
            vlm_layer = self.language_model.layers[layer_idx]
            attn = vlm_layer.self_attn
            normed = vlm_layer.input_layernorm(vlm_hidden)
            q = attn.q_norm(attn.q_proj(normed).view(B, P, -1, head_dim)).transpose(1, 2)
            k = attn.k_norm(attn.k_proj(normed).view(B, P, -1, head_dim)).transpose(1, 2)
            v = attn.v_proj(normed).view(B, P, -1, head_dim).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            # Cache post-RoPE prefix K/V for this layer.
            past_key_values.update(k, v, layer_idx)
            n_rep = q.size(1) // k.size(1)
            attn_out = F.scaled_dot_product_attention(
                q, repeat_kv(k, n_rep), repeat_kv(v, n_rep),
                attn_mask=attn_bias.to(q.dtype), scale=attn.scaling,
            ).transpose(1, 2)
            vlm_hidden = vlm_hidden + attn.o_proj(attn_out.reshape(B, P, -1))
            residual = vlm_hidden
            vlm_hidden = residual + vlm_layer.mlp(vlm_layer.post_attention_layernorm(vlm_hidden))
            if deepstack_visual_embeds is not None and layer_idx < len(deepstack_visual_embeds):
                vlm_hidden = self.language_model._deepstack_process(
                    vlm_hidden, visual_pos_masks, deepstack_visual_embeds[layer_idx]
                )

        past_key_values.prefix_valid_mask = prefix_valid.detach()
        past_key_values.prefix_last_position = _global_last_position(
            prefix_position_ids, prefix_valid
        ).detach()
        return RynnBrainVLAModelOutputWithPast(
            last_hidden_state=vlm_hidden,
            past_key_values=past_key_values,
            rope_deltas=rope_deltas,
            actions=None,
            loss=None,
        )

    def _expert_decode(
        self,
        actions,
        times,
        action_position_ids,
        past_key_values,
        rope_deltas,
        ee_bias=None,
        view_cond=(None, None),
    ):
        """Decode for the expert path: run only the expert stream, with each layer's
        action queries attending to the cached prefix K/V + fresh action K/V."""
        expert = self.action_expert
        B, A = actions.shape[:2]
        device = actions.device
        action_hidden = self.action_in_proj(actions.type(self.dtype))
        cond = expert.embed_time(times) if self.time_conditioning == "adaln" else None
        if cond is not None and ee_bias is not None:
            cond = cond + ee_bias
        # pi0-style concat time conditioning: MLP([action_emb ; time_emb]) (openpi
        # pi0.py:172-178, lingbot :475, InternVLA :564). Applied to the ACTION tokens
        # only -- before the foresight prepend -- matching InternVLA's suffix layout
        # "[state] [learnable] [action_time]" (:918). No-op unless expert_time_concat.
        if getattr(expert, "time_concat", False):
            action_hidden = expert.maybe_time_concat(action_hidden, times)

        # Match training: prepend learnable foresight tokens so the expert stream
        # sees the same prefix tokens it saw during _forward_expert / amortized.
        if expert.num_foresight_tokens > 0:
            lt_emb = expert.learnable_tokens_in_proj(expert.learnable_tokens)
            lt_emb = lt_emb.unsqueeze(0).expand(B, -1, -1)
            action_hidden = torch.cat([lt_emb, action_hidden], dim=1)
            A_total = expert.num_foresight_tokens + A
        else:
            A_total = A

        # Match training: add sinusoidal position embedding to the residual stream.
        action_pos_emb = expert.embed_action_positions(A_total, device, self.dtype)
        action_hidden = action_hidden + action_pos_emb

        # Match training (_forward_expert): [foresight | V camera tokens | actions], spliced
        # AFTER the sinusoidal positions so the camera token is the bare slot_seed_proj
        # output and the action tokens keep their flag-off phases. num_vc is folded into
        # A_total below, after the 1..A_total ramp has been built for the other tokens.
        vc_tokens, vc_valid = view_cond
        num_vc = 0
        if vc_tokens is not None:
            num_vc = vc_tokens.size(1)
            _lt = expert.num_foresight_tokens
            action_hidden = torch.cat(
                [action_hidden[:, :_lt], vc_tokens.type(action_hidden.dtype), action_hidden[:, _lt:]],
                dim=1,
            )

        # Match the joint path by continuing from one valid global M-RoPE maximum per sample.
        prefix_len = past_key_values.get_seq_length(0)
        offset = getattr(past_key_values, "prefix_last_position", None)
        if offset is not None:
            offset = offset.to(device=device, dtype=torch.long)
        elif action_position_ids is not None and action_position_ids.dim() == 3:
            raw_offset = (action_position_ids[..., :1].to(torch.long) - 1).to(device)
            offset = raw_offset.amax(dim=0, keepdim=True).expand(3, -1, -1)
        else:
            offset = torch.full((3, B, 1), prefix_len - 1, dtype=torch.long, device=device)
        _ramp = torch.arange(1, A_total + 1, dtype=torch.long, device=device)
        if num_vc > 0:
            # Camera RoPE positions restart at 1, exactly as in _forward_expert.
            _lt = expert.num_foresight_tokens
            _ramp = torch.cat([
                _ramp[:_lt],
                torch.arange(1, num_vc + 1, dtype=torch.long, device=device),
                _ramp[_lt:],
            ])
            A_total = A_total + num_vc
        action_rope_pos = _ramp.view(1, 1, A_total).repeat(3, B, 1) + offset
        cos, sin = self.language_model.rotary_emb(action_hidden, action_rope_pos)
        scaling = self.language_model.layers[0].self_attn.scaling

        # Match training and mask padded prefix keys for every cached decode.
        neg_inf = torch.finfo(self.dtype).min
        num_lt = expert.num_foresight_tokens
        allow = torch.ones(B, A_total, prefix_len + A_total, dtype=torch.bool, device=device)
        prefix_valid = getattr(past_key_values, "prefix_valid_mask", None)
        if prefix_valid is None:
            prefix_valid = torch.ones(B, prefix_len, dtype=torch.bool, device=device)
        else:
            prefix_valid = prefix_valid.to(device=device, dtype=torch.bool)
        allow[:, :, :prefix_len] &= prefix_valid[:, None, :]
        if num_lt > 0:
            allow[:, num_lt:, prefix_len + self.action_expert.foresight_vis_offset:prefix_len + num_lt] = False
        if num_vc > 0:
            # Same two rules as training: camera tokens are context (cannot read the
            # action tokens), and roles with no camera in this sample are key-masked.
            _v0, _v1 = num_lt, num_lt + num_vc
            allow[:, _v0:_v1, prefix_len + _v1:] = False
            allow[:, :, prefix_len + _v0:prefix_len + _v1] &= vc_valid[:, None, :]
        decode_attn_bias = torch.zeros(
            B, 1, A_total, prefix_len + A_total, dtype=self.dtype, device=device
        )
        decode_attn_bias.masked_fill_(~allow.unsqueeze(1), neg_inf)

        for layer_idx in range(self.config.text_config.num_hidden_layers):
            exp_layer = expert.layers[layer_idx]
            qa, ka, va = exp_layer.compute_qkv(action_hidden, cond)
            qa = qa.transpose(1, 2)
            ka = ka.transpose(1, 2)
            va = va.transpose(1, 2)
            qa, ka = apply_rotary_pos_emb(qa, ka, cos, sin)

            prefix_k = past_key_values.prefix_keys[layer_idx]
            prefix_v = past_key_values.prefix_values[layer_idx]
            k = torch.cat([prefix_k, ka], dim=-2)
            v = torch.cat([prefix_v, va], dim=-2)
            n_rep = qa.size(1) // k.size(1)
            attn_out = F.scaled_dot_product_attention(
                qa, repeat_kv(k, n_rep), repeat_kv(v, n_rep),
                attn_mask=None if decode_attn_bias is None else decode_attn_bias.to(qa.dtype),
                scale=scaling,
            ).transpose(1, 2)
            action_hidden = exp_layer.apply_output(action_hidden, attn_out, cond)

        action_hidden = expert.final(action_hidden, cond)
        _drop = expert.num_foresight_tokens + num_vc
        if _drop > 0:
            action_hidden = action_hidden[:, _drop:]

        latent_mode = self.use_latent_actions and self.latent_action_head is not None
        if latent_mode:
            pred_actions = self.latent_action_head(action_hidden)
        else:
            pred_actions = self._readout_action(action_hidden)
            pred_actions = pred_actions.view(actions.shape)

        return RynnBrainVLAModelOutputWithPast(
            last_hidden_state=action_hidden,
            past_key_values=past_key_values,
            rope_deltas=rope_deltas,
            actions=pred_actions,
            loss=None,
        )
