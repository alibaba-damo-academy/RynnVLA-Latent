# Portions of this file are derived from HuggingFace Transformers
# (https://github.com/huggingface/transformers), Copyright The HuggingFace Inc. team,
# licensed under the Apache License, Version 2.0. The license text is in LICENSE; the
# attribution is recorded in NOTICE.
# Upstream reference: src/transformers/models/qwen3_vl/configuration_qwen3_vl.py

from transformers.models.qwen3_vl import Qwen3VLConfig


class RynnBrainVLAConfig(Qwen3VLConfig):
    model_type = "rynn_brain_vla"

    def __init__(
        self,
        action_dim=6,
        action_chunk_size=20,
        state_token_id=-1,
        time_conditioning="adaln",
        knowledge_insulation=False,
        action_head_type="shared",
        expert_hidden_size=768,
        expert_intermediate_size=2752,
        expert_num_attention_heads=None,
        expert_train_repeat=1,
        expert_qk_norm=False,
        expert_backbone="custom",
        expert_per_layer_adanorm=True,
        expert_num_foresight_tokens=50,
        expert_foresight_visibility="first",
        expert_time_concat=False,
        use_latent_actions=False,
        use_latent_head_readout=False,
        bypass_latent_output_projection=False,
        latent_action_dim=256,
        latent_action_chunk_size=None,
        latent_action_stride=1,
        num_view_slots=1,
        use_view_role_embedding=False,
        use_view_cond_slots=False,
        use_depth_aux=False,
        depth_grid_size=16,
        depth_aux_num_layers=2,
        depth_aux_num_heads=8,
        predict_future_depth=True,
        depth_loss_weight=1.0,
        use_sf_align=False,
        sf_align_weight=0.5,
        sf_align_layer=None,
        sf_teacher_path=None,
        sf_teacher_code=None,
        sf_teacher_dim=1024,
        sf_teacher_layer=-1,
        sf_use_vlm_norm=True,
        use_lb_align=False,
        lb_align_weight=0.5,
        lb_align_loss="cosine",
        lb_align_layer=None,
        lb_num_queries=256,
        lb_num_task_tokens=8,
        lb_resampler_layers=1,
        lb_resampler_heads=4,
        lb_resampler_dim_head=32,
        lb_resampler_ff_mult=1,
        lb_query_init_std=1.0,
        use_ee_embedding=False,
        num_ee_types=8,
        use_history_cond=False,
        history_token_id=-1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.action_dim = action_dim
        self.action_chunk_size = action_chunk_size
        self.state_token_id = state_token_id
        # Action-head time conditioning: "adaln" (AdaLN/FiLM, DiT-style) or
        # "concat" (legacy concat-then-MLP). Defaults to "adaln".
        self.time_conditioning = time_conditioning
        # Knowledge insulation: when True, the prefix (VLM) key/value tensors that
        # the action (suffix) queries attend to are detached, so the action loss
        # gradient does not flow back into the VLM. Opt-in single-direction switch.
        self.knowledge_insulation = knowledge_insulation
        # The (monkey-patched) attention module's ``self.config`` is the text config,
        # so mirror the flag there for it to be readable during attention.
        if getattr(self, "text_config", None) is not None:
            self.text_config.knowledge_insulation = knowledge_insulation

        # Action head architecture: "shared" (action tokens run through the shared VLM
        # backbone) or "expert" (a separate narrow action-expert stack, run in two-stream
        # lockstep with the VLM per lingbot-vla-v2 / pi0). For "expert", the head layout
        # (head_dim / num_key_value_heads / num_hidden_layers / attention heads default)
        # is derived from text_config so the per-layer Q/K/V concat is valid; only the
        # residual width and MLP size below are expert-specific.
        self.action_head_type = action_head_type
        self.expert_hidden_size = expert_hidden_size
        self.expert_intermediate_size = expert_intermediate_size
        self.expert_num_attention_heads = expert_num_attention_heads
        # VLM-forward amortization: N independent noise/time draws per observation
        # share a single VLM prefix forward during training (expert path only).
        self.expert_train_repeat = expert_train_repeat
        # Per-head RMSNorm on the expert's Q/K before the joint softmax. Qwen3-VL's own
        # attention normalizes Q/K per head (q_norm/k_norm); without this flag the expert
        # side of the two-stream joint attention feeds RAW projections into the same
        # softmax as the VLM's normalized ones (scale asymmetry, no logit-explosion
        # guard). InternVLA-A1.5 applies q_norm/k_norm on both streams. Default False:
        # checkpoints trained without the norm must keep the old forward.
        self.expert_qk_norm = expert_qk_norm

        # ---- Expert architecture attribution axes (see std_expert_rynn_brain_vla.py) ----
        # Every default below reproduces the previously HARD-CODED behaviour exactly, so
        # existing checkpoints and scripts are unaffected.
        #
        # expert_backbone: "custom" = the hand-written ActionExpertLayer stack.
        #   "qwen3_vl" = a stock narrow Qwen3VLTextModel, i.e. the VLM's own family. This
        #   makes the joint softmax symmetric (q_norm/k_norm on both sides) by
        #   construction and removes the 117.6M (2B) / 150.9M (4B) dead `adaln`
        #   parameters of the custom stack -- but it is NOT what any peer literally
        #   instantiates.
        #   "qwen2" = lingbot-vla-v2's ACTUAL expert (modeling_lingbot_vla_v2.py:138,
        #   Qwen2ForCausalLM._from_config): no q_norm/k_norm and biased q/k/v, which
        #   reproduces their joint-softmax asymmetry against a Qwen3-VL prefix that does
        #   normalize (qwen3vl_in_vla.py:185-186). Use for a faithful lingbot arm.
        #   In every case the head layout (head_dim / num_key_value_heads /
        #   num_hidden_layers / heads) is derived from text_config, which is the peers'
        #   own rule -- see std_expert_rynn_brain_vla.build_stock_layers.
        self.expert_backbone = expert_backbone
        # Per-layer AdaRMSNorm time conditioning (pi0.5 / lingbot `adanorm_time: true`).
        self.expert_per_layer_adanorm = expert_per_layer_adanorm
        # InternVLA-style foresight tokens prepended to the expert suffix
        # (`modeling_internvla_a1_5.py:568`, their num_learnable_tokens=50). NOTE: in
        # InternVLA these are supervised by a frozen WAN video loss and their chunk_size
        # is also 50 (1:1); unsupervised they collapse to effective rank ~1 (measured on
        # E1/E2/E3 checkpoints), so 0 is the right value for any arm without that module.
        self.expert_num_foresight_tokens = expert_num_foresight_tokens
        # Which foresight tokens the ACTION tokens may attend to.
        #   "first" -- only foresight[0] (what this repo has always done)
        #   "all"   -- all of them, which is what InternVLA actually does
        #              (`att_masks += [1] + [0] * (num_lt - 1)`, :944)
        self.expert_foresight_visibility = expert_foresight_visibility
        # pi0-style concat time conditioning: MLP([action_emb ; time_emb]).
        #   pi0 / InternVLA-A1.5 : concat only
        #   lingbot-vla-v2       : concat AND per-layer AdaRMSNorm (both shipped yamls)
        #   pi0.5 / this repo    : AdaRMSNorm only  (expert_time_concat=False)
        self.expert_time_concat = expert_time_concat

        # Hierarchical VLA: when use_latent_actions=True, the model predicts latent actions
        # (robot-agnostic, from LAM) instead of robot-specific actions. Used for pretrain.
        # During finetune, set use_latent_actions=False and use an ActionDecoder to map
        # latent actions to robot-specific actions.
        self.use_latent_actions = use_latent_actions
        # fix-v7: reuse the pretrained latent_action_head as the direct readout trunk
        # (+ fresh latent_readout_proj). Declared here because config_overrides pass
        # through from_pretrained(**kwargs), whose from_dict drops undeclared keys.
        self.use_latent_head_readout = use_latent_head_readout
        # Stage-2 option: retain the pretrained head's first Linear + SiLU, then use a fresh
        # expert_hidden_size -> action_dim readout. Declared here for the same reason as
        # use_latent_head_readout above: from_dict drops undeclared keys, so an undeclared
        # flag could never be switched on. Default False preserves both the existing Stage-1
        # and Stage-2 checkpoint layouts.
        self.bypass_latent_output_projection = bypass_latent_output_projection
        self.latent_action_dim = latent_action_dim
        # Preserve None so the derived length follows HF's later action_chunk_size overrides.
        self.latent_action_chunk_size = latent_action_chunk_size
        self.latent_action_stride = latent_action_stride
        # NB: this is NOT constants.NUM_VIEW_SLOTS. Two different quantities share the name:
        #   config.num_view_slots     -> how many latent slots the multiview action stream
        #                                carries; read only when use_latent_actions=True
        #   constants.NUM_VIEW_SLOTS  -> how many camera ROLES exist, sizing the role
        #                                embedding and the V4 camera tokens; read by both
        # Direct fine-tuning leaves this at the default 1 and is unaffected; latent
        # pretraining sets it to one slot per role.
        self.num_view_slots = num_view_slots

        # B2 camera identity: add a learned per-camera-role embedding to that camera's image
        # tokens in the VLM prefix, so two observation images are no longer interchangeable.
        # Roles come from constants.VIEW_ROLES via the dataset's camera_slot_map; the table is
        # zero-initialised, so turning this on adds a module whose contribution starts at
        # exactly 0. Independent of num_view_slots / use_latent_actions: it lives in the
        # prefix, which the direct fine-tuning path shares with latent pretraining, and is the
        # only camera-identity mechanism that transfers between the two stages unchanged.
        self.use_view_role_embedding = use_view_role_embedding
        self.use_view_cond_slots = use_view_cond_slots

        # Auxiliary depth prediction (predictive-dynamics regularizer). When enabled and
        # paired depth targets are provided in the batch, a query-based readout head
        # predicts the current-frame depth (anchor, anti-collapse) and optionally the
        # future-frame depth (foresight) from the VLM's visual tokens, supervised by the
        # ground-truth depth maps + validity mask. The head is a pure readout (never
        # injected into the token sequence) and is dropped at inference.
        self.use_depth_aux = use_depth_aux
        self.depth_grid_size = depth_grid_size
        self.depth_aux_num_layers = depth_aux_num_layers
        self.depth_aux_num_heads = depth_aux_num_heads
        self.predict_future_depth = predict_future_depth
        self.depth_loss_weight = depth_loss_weight

        # Spatial-Forcing alignment (training-only): cosine-align mid-layer image-token
        # hidden states with a frozen DA3MONO teacher. sf_align_layer=None -> 2/3 depth.
        self.use_sf_align = use_sf_align
        self.sf_align_weight = sf_align_weight
        self.sf_align_layer = sf_align_layer
        self.sf_teacher_path = sf_teacher_path
        self.sf_teacher_code = sf_teacher_code
        self.sf_teacher_dim = sf_teacher_dim
        self.sf_teacher_layer = sf_teacher_layer
        self.sf_use_vlm_norm = sf_use_vlm_norm

        # LingBot-VLA 2.0 style query-readout distillation: learnable queries are
        # pooled into lb_num_task_tokens prefix tokens, and a Perceiver resampler reads
        # [primary-view image tokens + query hiddens] out of the trunk to predict the same
        # DA3MONO grid the SF arm uses. Unlike SF, these tokens are part of the prefix at
        # inference too, so the flag must persist in the checkpoint config.
        # lb_align_layer=None -> final trunk layer (LingBot reads the last hidden state).
        self.use_lb_align = use_lb_align
        self.lb_align_weight = lb_align_weight
        self.lb_align_loss = lb_align_loss
        self.lb_align_layer = lb_align_layer
        self.lb_num_queries = lb_num_queries
        self.lb_num_task_tokens = lb_num_task_tokens
        self.lb_resampler_layers = lb_resampler_layers
        self.lb_resampler_heads = lb_resampler_heads
        self.lb_resampler_dim_head = lb_resampler_dim_head
        self.lb_resampler_ff_mult = lb_resampler_ff_mult
        self.lb_query_init_std = lb_query_init_std

        # End-effector / embodiment type embedding: a learned codebook entry per EE
        # category (indexed by an ee_type_id in the batch) added to the action-head
        # time-conditioning signal, giving the model embodiment-specific action priors.
        self.use_ee_embedding = use_ee_embedding
        self.num_ee_types = num_ee_types

        # In-context history conditioning: when enabled and the batch provides past
        # (state, action) pairs plus <|history_pad|> slots in the prompt, the model
        # encodes them into history tokens (scattered like the state token) that the
        # action stream attends to -- enabling deployment-time adaptation / reactive
        # recovery. Inputs (past states/actions) are available at inference.
        self.use_history_cond = use_history_cond
        self.history_token_id = history_token_id

    # ---- derived: latent_action_chunk_size -----------------------------------------------
    # Lazy on purpose, so it stays correct no matter WHEN action_chunk_size is assigned.
    # Backed by _latent_action_chunk_size, in which None means "follow action_chunk_size".

    @property
    def latent_action_chunk_size(self):
        explicit = self.__dict__.get("_latent_action_chunk_size")
        return self.action_chunk_size if explicit is None else explicit

    @latent_action_chunk_size.setter
    def latent_action_chunk_size(self, value):
        self.__dict__["_latent_action_chunk_size"] = value

    def to_dict(self):
        out = super().to_dict()
        # The property lives on the class, so __dict__ -- and therefore super().to_dict() --
        # carries the private backing field. Swap it back to the public name, and serialise
        # the *sentinel* rather than the resolved number: writing out today's action_chunk_size
        # would freeze "follow action_chunk_size" into a literal and re-create exactly the
        # staleness this property exists to prevent. to_diff_dict() calls to_dict(), so
        # save_pretrained inherits this.
        out.pop("_latent_action_chunk_size", None)
        out["latent_action_chunk_size"] = self.__dict__.get("_latent_action_chunk_size")
        return out
