import json
import math
import os
from dataclasses import dataclass, field, fields
from datetime import timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import deepspeed
import torch
from packaging import version
from transformers import AutoConfig, PreTrainedConfig
from transformers.trainer_utils import IntervalStrategy, SaveStrategy
from transformers.training_args import OptimizerNames, SchedulerType

from . import parallel_state as mpu
from .registry import DATASET_REGISTRY
from .utils import logging
from .utils.pipeline_parallel import PipelineSchedule
from .constants import RotationRepresentation

logger = logging.get_logger(__name__)


@dataclass
class BaseArguments:
    def __post_init__(self):
        pass

    def to_dict(self):
        return {field.name: getattr(self, field.name) for field in fields(self) if field.init}

    def to_json_string(self):
        data_dict = self.to_dict()
        for key, value in data_dict.items():
            if isinstance(value, Enum):
                data_dict[key] = value.value
        return json.dumps(data_dict, indent=2)


@dataclass
class ModelArguments(BaseArguments):
    model_path: Optional[str] = field(default=None)
    trunk_model_path: Optional[str] = field(default=None)
    model_type: Optional[str] = field(default=None)

    config_overrides: Dict[str, Any] | str | None = field(default=None)
    processor_overrides: Dict[str, Any] | str | None = field(default=None)

    vision_encoder_path: Optional[str] = field(default=None)

    attn_implementation: Optional[str] = field(default="flash_attention_2")

    fp16: bool = field(default=False)
    bf16: bool = field(default=True)

    use_token_compression: Optional[bool] = field(default=False)

    def __post_init__(self):
        super().__post_init__()
        assert self.model_path is not None

        if isinstance(self.config_overrides, str):
            self.config_overrides = json.loads(self.config_overrides)
        elif self.config_overrides is None:
            self.config_overrides = {}

        if isinstance(self.processor_overrides, str):
            self.processor_overrides = json.loads(self.processor_overrides)
        elif self.processor_overrides is None:
            self.processor_overrides = {}

        if self.model_type is None:
            config = AutoConfig.from_pretrained(self.model_path)
            self.model_type = config.model_type

        if self.bf16:
            self.dtype = torch.bfloat16
        elif self.fp16:
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32


@dataclass
class ParallelismArguments(BaseArguments):
    pipeline_parallel_size: int = field(default=1)
    pipeline_parallel_schedule: Optional[str] = field(
        default=None, metadata={"choices": [item.value for item in PipelineSchedule]}
    )
    reduced_layers_in_stage_zero: int = field(default=0)

    expert_parallel_size: int = field(default=1)

    context_parallel_size: int = field(default=1)
    encoder_context_parallel_size: int = field(default=1)

    pp_broadcast_data: bool = field(default=False)
    cp_broadcast_data: bool = field(default=False)

    ddp_timeout: int = field(default=7200)

    def __post_init__(self):
        super().__post_init__()
        if self.expert_parallel_size != 1:
            raise ValueError("RynnVLA uses a dense action expert; expert_parallel_size must be 1")

        # These three knobs pass validation but do not work. Reject them here rather than
        # letting them fail late or degrade silently.
        #
        # context / encoder context parallelism divide data_parallel_size in parallel_state.py,
        # so anything above 1 halves dp_world_size -- and therefore the global batch and
        # throughput -- while nothing consumes the groups: rynnvla/models/ never references
        # context_parallel, and the only reader (utils/context_parallel.py) pads sequences to a
        # multiple of cp_size*2 without ever splitting them. The result is a smaller batch, no
        # memory saving, and no error.
        if self.context_parallel_size != 1:
            raise NotImplementedError(
                "context_parallel_size > 1 is not implemented: nothing splits the sequence, but "
                "parallel_state.py still divides data_parallel_size by it, silently shrinking the "
                "global batch. Leave it at 1."
            )
        if self.encoder_context_parallel_size != 1:
            raise NotImplementedError(
                "encoder_context_parallel_size > 1 is not implemented for the same reason as "
                "context_parallel_size. Leave it at 1."
            )
        # Pipeline parallelism fails even earlier than its NotImplementedError schedules:
        # build_model asserts hasattr(model, "apply_pipeline_parallel") and no model here
        # defines it.
        if self.pipeline_parallel_size != 1:
            raise NotImplementedError(
                "pipeline_parallel_size > 1 is not implemented: models/__init__.py asserts "
                "hasattr(model, 'apply_pipeline_parallel'), which no model in this repo defines, "
                "and every schedule except NO_PIPELINING raises NotImplementedError. Leave it at 1."
            )

        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device("cuda", self.local_rank)

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend="nccl",
                device_id=self.device,
                timeout=timedelta(seconds=self.ddp_timeout),
            )

        deepspeed.init_distributed(dist_backend="nccl")

        self.global_world_size = torch.distributed.get_world_size()
        self.global_rank = torch.distributed.get_rank()

        assert 1 <= self.pipeline_parallel_size <= torch.distributed.get_world_size()
        assert 1 <= self.expert_parallel_size <= torch.distributed.get_world_size()
        assert 1 <= self.context_parallel_size <= torch.distributed.get_world_size()
        assert 1 <= self.encoder_context_parallel_size <= torch.distributed.get_world_size()
        assert self.reduced_layers_in_stage_zero >= 0

        if self.pipeline_parallel_size > 1:
            assert self.pipeline_parallel_schedule is not None
        else:
            assert self.pipeline_parallel_schedule is None

        self.pipeline_parallel_schedule = PipelineSchedule(self.pipeline_parallel_schedule)

        mpu.initialize_model_parallel(
            pipeline_model_parallel_size=self.pipeline_parallel_size,
            expert_model_parallel_size=self.expert_parallel_size,
            context_parallel_size=self.context_parallel_size,
            encoder_context_parallel_size=self.encoder_context_parallel_size,
        )

        self.dp_group = mpu.get_data_parallel_group()
        self.dp_world_size = mpu.get_data_parallel_world_size()
        self.dp_rank = mpu.get_data_parallel_rank()

        self.dcp_group = mpu.get_data_parallel_group(with_context_parallel=True)
        self.dcp_world_size = mpu.get_data_parallel_world_size(with_context_parallel=True)
        self.dcp_rank = mpu.get_data_parallel_rank(with_context_parallel=True)

        self.cp_group = mpu.get_context_parallel_group()
        self.cp_world_size = mpu.get_context_parallel_world_size()
        self.cp_rank = mpu.get_context_parallel_rank()

        self.pp_group = mpu.get_pipeline_model_parallel_group()
        self.pp_world_size = mpu.get_pipeline_model_parallel_world_size()
        self.pp_rank = mpu.get_pipeline_model_parallel_rank()

        self.ep_group = mpu.get_expert_model_parallel_group()
        self.ep_world_size = mpu.get_expert_model_parallel_world_size()
        self.ep_rank = mpu.get_expert_model_parallel_rank()

        self.edp_group = mpu.get_expert_data_parallel_group()
        self.edp_world_size = mpu.get_expert_data_parallel_world_size()
        self.edp_rank = mpu.get_expert_data_parallel_rank()

        self._enforce_ddp_timeout()

    def _enforce_ddp_timeout(self):
        # DeviceMesh builds its subgroups by splitting the default group, and that path
        # reuses the parent's NCCL Options object while stamping its own 10-minute default
        # into it. Any timeout set before mesh creation is therefore silently lost, and a
        # checkpoint save that copies tens of GB to the shared mount keeps the other ranks
        # inside a collective for longer than that.
        timeout = timedelta(seconds=self.ddp_timeout)
        set_pg_timeout = getattr(torch.distributed.distributed_c10d, "_set_pg_timeout", None)
        if set_pg_timeout is None:
            logger.warning("torch.distributed has no _set_pg_timeout; ddp_timeout=%ds not enforced", self.ddp_timeout)
            return
        # Groups created later (the DeepSpeed engine builds its own) never pass through the
        # loop below, and a resume broadcast on one of them died on the 10-minute default
        # while rank 0 was still reading a checkpoint off the shared mount. new_group() reads
        # this module global whenever no explicit timeout is given, so patch it first.
        c10d = torch.distributed.distributed_c10d
        c10d.default_pg_nccl_timeout = timeout
        seen = set()
        for group in list(getattr(c10d._world, "pg_map", {})) + [c10d._get_default_group()]:
            if group is None or id(group) in seen:
                continue
            seen.add(id(group))
            set_pg_timeout(timeout, group)
        logger.info("Enforced ddp_timeout=%ds on %d process group(s)", self.ddp_timeout, len(seen))


@dataclass
class DataArguments(BaseArguments):
    data_type: str = field(default=None)
    data_path: str = field(default=None)
    data_mixture: Dict[str, Any] | str | None = field(default=None)

    # VLM processing configs
    model_max_length: Optional[int] = field(default=16384)
    mm_max_length: Optional[int] = field(default=10240)
    fps: Optional[int] = field(
        default=1,
        metadata={"help": (
            "VLM only: frames-per-second to sub-sample from input videos "
            "(processor extracts duration*fps frames, capped by max_frames). "
            "Ignored by VLA datasets — use target_fps for action/state resampling."
        )},
    )
    max_frames: Optional[int] = field(
        default=180,
        metadata={"help": "VLM only: hard cap on frames extracted per input video (pairs with fps)."},
    )

    # VLA processing configs
    action_chunk_size: int = field(default=20)
    # Default False on purpose. Turning this on makes base.py compute action - state, but the
    # robot corpora here already store delta actions: against LIBERO's world-frame EEF state
    # that subtraction yields garbage targets, and against a joint-space state it is a silent
    # no-op (Arm._apply leaves fields whose counterpart is None untouched). It never helps, so
    # it must not be the default. Both shipped recipes set it explicitly anyway.
    use_delta_action: bool = field(default=False)
    eef_rotation_repr: Optional[str] = field(default=None, metadata={"choices": [item.value for item in RotationRepresentation]})
    action_only: bool = field(default=False)
    target_fps: Optional[float] = field(
        default=None,
        metadata={"help": (
            "VLA only: resample action/state time-series to this rate "
            "(linear for joint/eef_pos, SLERP for rotation, nearest for gripper). "
            "None = use each episode's native fps (no resampling). "
            "Distinct from `fps`, which sub-samples VLM video frames."
        )},
    )
    use_visual_augmentation: bool = field(
        default=False,
        metadata={"help": (
            "VLA only: enable visual data augmentation (ColorJitter with "
            "brightness/contrast/saturation, lingbot-vla-v2 style). Improves "
            "robustness to lighting/color variations in real-world deployment."
        )},
    )
    emit_teacher_images: bool = field(
        default=False,
        metadata={"help": (
            "VLA only: attach un-augmented 224x224 copies of every camera frame to each "
            "sample (batch-ordered like image_grid_thw) as input for the frozen SF-alignment "
            "teacher. Required when the model config sets use_sf_align=true."
        )},
    )
    chunk_overlap_ratio: float = field(
        default=0.0,
        metadata={"help": (
            "VLA only: overlap ratio between consecutive action chunks during training. "
            "0.0 = no overlap (stride = chunk_size), 0.5 = 50%% overlap (stride = chunk_size/2). "
            "Overlapping chunks improve temporal consistency and smoothness. Must be in [0, 1)."
        )},
    )

    # Hierarchical VLA: latent action pretraining
    use_latent_actions: bool = field(
        default=False,
        metadata={"help": (
            "Hierarchical VLA: when True, the model predicts latent actions (robot-agnostic, "
            "from LAM) instead of robot-specific actions. Used for pretrain stage. "
            "During finetune, set to False and use an ActionDecoder to map latent -> robot actions."
        )},
    )
    latent_action_dim: int = field(
        default=256,
        metadata={"help": "Dimension of latent actions (default: 256, from LAM)."},
    )
    latent_action_path: Optional[str] = field(
        default=None,
        metadata={"help": (
            "Path to latent action targets (numpy .npy files, one per episode). "
            "Required when use_latent_actions=True. Each file should have shape "
            "(num_frames, latent_action_dim)."
        )},
    )
    num_view_slots: int = field(
        default=1,
        metadata={"help": (
            "Multi-view latent: max number of view-role slots K (slot i <-> a fixed "
            "semantic camera role, see constants.VIEW_ROLES). The VLA outputs K per-view "
            "latent actions; a sample only activates the slots for the cameras it has. "
            "K=1 = single-view (backward compatible)."
        )},
    )
    latent_stats_path: Optional[str] = field(
        default=None,
        metadata={"help": (
            "Path to merged latent normalization stats JSON (mean/std per dim), produced "
            "by scripts/build_latent_stats.py. Used to standardize latent actions before "
            "they become VLA flow-matching targets."
        )},
    )

    # ActionDecoder configuration (Stage 2 finetune)
    robot_type: Optional[str] = field(
        default=None,
        metadata={"help": "Robot type for ActionDecoder finetune (e.g., 'franka', 'aloha_agilex')."},
    )
    state_dim: int = field(
        default=7,
        metadata={"help": "Dimension of robot state (e.g., 7 for Franka joint positions)."},
    )
    action_dim: int = field(
        default=7,
        metadata={"help": "Dimension of robot action (e.g., 7 for Franka joint velocities)."},
    )
    decoder_hidden_dim: int = field(
        default=512,
        metadata={"help": "Hidden dimension of ActionDecoder MLP."},
    )
    decoder_num_layers: int = field(
        default=3,
        metadata={"help": "Number of MLP layers in ActionDecoder."},
    )
    decoder_actions_per_latent: int = field(
        default=1,
        metadata={"help": (
            "How many distinct actions each latent step decodes into (MultiViewActionDecoder). "
            "Set to latent_action_stride (e.g. 5) so one latent unpacks into its full motion "
            "segment instead of one action repeated stride times. Default 1 = legacy behavior."
        )},
    )
    decoder_use_step_attention: bool = field(
        default=False,
        metadata={"help": (
            "MultiViewActionDecoder fix-v3: decode the chunk as per-step tokens (repeat_interleave "
            "latents + step positional embedding + temporal self-attention + shared per-step head) "
            "instead of one Linear(hidden -> actions_per_latent*action_dim) per latent. Removes the "
            "measured curvature spike at every group boundary. Requires decoder_actions_per_latent > 1. "
            "Changes the decoder's parameter shapes: not loadable into/from legacy decoder checkpoints."
        )},
    )
    decoder_num_temporal_layers: int = field(
        default=2,
        metadata={"help": "Number of temporal self-attention layers when decoder_use_step_attention=True."},
    )
    decoder_flow_matching: bool = field(
        default=False,
        metadata={"help": (
            "fix-v4: drop the MLP decoder entirely and decode robot actions by flow matching in "
            "the action expert (pi0-style suffix of noisy per-step action tokens attending the "
            "full VLM prefix + the integrated latent context). Adds no new parameters: reuses "
            "action_in_proj/action_out_proj and the expert stack. Requires "
            "decoder_latent_num_steps >= 1 and freeze_vla=False."
        )},
    )
    decoder_action_num_steps: int = field(
        default=10,
        metadata={"help": (
            "Inference-time Euler steps for the robot-action flow when decoder_flow_matching=True. "
            "Recorded in action_decoder_config.json so the eval server replays the same setting."
        )},
    )
    decoder_latent_num_steps: int = field(
        default=0,
        metadata={"help": (
            "Euler flow-integration steps used to produce the latent fed to the decoder. "
            "0 = legacy: feed the raw single-forward velocity output (noise-contaminated). "
            ">=1 = integrate x += dt*v from a noise canvas down to a clean latent estimate "
            "(1 step cancels the noise at the same cost; 10 matches the direct variant). "
            "The same convention must be used at inference (stored in the decoder config)."
        )},
    )
    use_flow_matching: bool = field(
        default=False,
        metadata={"help": "Use flow matching in ActionDecoder for smoother action generation."},
    )
    num_flow_steps: int = field(
        default=5,
        metadata={"help": "Number of flow matching refinement steps (only used if use_flow_matching=True)."},
    )
    freeze_vla: bool = field(
        default=True,
        metadata={"help": (
            "Freeze the VLA backbone in the ActionDecoder wrapper (default: True, the standard "
            "Stage-2 decoder-only finetune). Set False to full fine-tune the VLA end-to-end "
            "together with the decoder (latent-bottleneck ablation)."
        )},
    )

    def __post_init__(self):
        super().__post_init__()

        if self.data_mixture is not None:
            assert self.data_type is None and self.data_path is None
            if isinstance(self.data_mixture, str):
                with open(self.data_mixture, "r") as f:
                    data_mixture = json.load(f)

                assert isinstance(data_mixture, list)
                for data_source in data_mixture:
                    assert isinstance(data_source, dict)

                self.data_mixture = data_mixture
                logger.info(f"Using data mixture: {data_mixture}")

        else:
            assert self.data_type is not None
            assert self.data_type in DATASET_REGISTRY, f"Available data types: {DATASET_REGISTRY.keys()}"

        if self.eef_rotation_repr is not None:
            self.eef_rotation_repr = RotationRepresentation(self.eef_rotation_repr)

@dataclass
class TrainingArguments(ModelArguments, ParallelismArguments, DataArguments, BaseArguments):
    # Efficiency-related configs
    deepspeed: str = field(default=None)

    gradient_checkpointing: bool = field(default=False)
    gradient_checkpointing_kwargs: Optional[Union[dict[str, Any], str]] = field(
        default=None,
        metadata={
            "help": "Gradient checkpointing key word arguments such as `use_reentrant`. Will be passed to `torch.utils.checkpoint.checkpoint` through `model.gradient_checkpointing_enable`."
        },
    )
    encoder_gradient_checkpointing_interval: Optional[int] = field(default=None)

    sequence_packing: bool = field(default=False)
    decoder_load_balancing: bool = field(default=False)

    dynamic_batching: bool = field(default=False)
    dynamic_batching_window_size: int = field(default=128)

    # Data configs
    micro_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=1)
    # The global batch this recipe's learning rate was calibrated for. Changing the rank count
    # does NOT rescale gradient accumulation or the LR, so the same recipe on a different GPU
    # count silently changes the global batch while leaving the LR untouched. Recording the
    # design point lets the trainer warn on deviation. None = no reference, no check.
    reference_global_batch: Optional[int] = field(default=None)

    num_train_epochs: float = field(default=3.0, metadata={"help": "Total number of training epochs to perform."})
    max_steps: int = field(
        default=-1,
        metadata={"help": "If > 0: set total number of training steps to perform. Override num_train_epochs."},
    )

    # Data loading configs
    sampler_shuffle: str = field(
        default="auto",
        metadata={"choices": ["auto", "global"],
                  "help": "auto preserves existing episode-locality order; global uniformly shuffles all samples."},
    )
    dataloader_num_workers: int = field(default=0)
    dataloader_drop_last: bool = field(default=False)
    dataloader_pin_memory: bool = field(default=False)
    dataloader_persistent_workers: bool = field(default=False)
    dataloader_prefetch_factor: Optional[int] = field(default=None)

    # Optimizer configs
    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for AdamW."})
    action_head_lr: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Peak learning rate for the randomly-initialized action modules listed in "
                "action_head_modules. The pretrained VLM keeps `learning_rate`. Leave unset (None) "
                "to use one rate for every parameter, which is the historical behaviour."
            )
        },
    )
    action_head_modules: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": (
                "Parameter-name prefixes that action_head_lr applies to. Defaults to the modules "
                "built from scratch on top of the pretrained VLM (action expert, projections, and "
                "the latent action decoder)."
            )
        },
    )
    reset_pretrained_modules: Optional[List[str]] = field(default=None)
    transfer_lr: Optional[float] = field(
        default=None, metadata={"help": "Independent peak LR for inherited modules listed in transfer_modules."},
    )
    transfer_modules: Optional[List[str]] = field(
        default=None, metadata={"help": "Module prefixes for transfer_lr; must not overlap action_head_modules."},
    )
    reset_pretrained_seed: int = field(default=42)
    frozen_parameters: Optional[List[str]] = field(default=None)

    lr_scheduler_type: Union[SchedulerType, str] = field(
        default="linear",
        metadata={"help": "The scheduler type to use."},
    )
    lr_scheduler_kwargs: Union[dict[str, Any], str] = field(
        default_factory=dict,
        metadata={
            "help": (
                "Extra parameters for the lr_scheduler such as {'num_cycles': 1} for the cosine with hard restarts. "
                "Not settable from the command line: HfArgumentParser reduces the annotation to `dict` and then "
                "calls dict() on the string. Use `min_lr_rate` for the common case."
            )
        },
    )
    min_lr_rate: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Floor for `cosine_with_min_lr`, as a fraction of each parameter group's own peak rate. "
                "Because the schedule is a LambdaLR multiplier, 0.05 floors a 1e-5 backbone at 5e-7 and a "
                "1e-4 action head at 5e-6, preserving their ratio for the whole run. Folded into "
                "lr_scheduler_kwargs; requires --lr_scheduler_type cosine_with_min_lr."
            )
        },
    )
    warmup_ratio: float = field(
        default=0.0, metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."}
    )
    warmup_steps: int = field(default=0, metadata={"help": "Linear warmup over warmup_steps."})

    optim: Union[OptimizerNames, str] = field(
        default="adamw_torch_fused" if version.parse(torch.__version__) >= version.parse("2.8") else "adamw_torch",
        metadata={"help": "The optimizer to use.", "choices": [item.value for item in OptimizerNames]},
    )
    optim_args: Optional[str] = field(default=None, metadata={"help": "Optional arguments to supply to optimizer."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for AdamW optimizer"})
    adam_beta2: float = field(default=0.95, metadata={"help": "Beta2 for AdamW optimizer (0.95 is the transformer/VLA standard; openpi/Qwen use 0.95)."})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})

    # EMA (exponential moving average) of trainable weights. When enabled, an EMA
    # shadow is kept in fp32 and saved as ema_model.bin / ema_model_pp_*.pt for
    # evaluation/inference (dropped from the live optimizer state). ZeRO-1/2 only
    # (params must be replicated, not sharded as in ZeRO-3).
    use_ema: bool = field(default=False, metadata={"help": "Maintain an EMA of trainable weights."})
    ema_decay: float = field(default=0.999, metadata={"help": "EMA decay (with early warmup)."})
    ema_device: Optional[str] = field(default=None, metadata={"help": "EMA storage device; cpu reduces GPU memory without changing precision."})
    export_ema: bool = field(
        default=False,
        metadata={
            "help": "Also write an `ema/` subdirectory in the final export, holding the EMA "
            "weights. Without this the run root only ever contains the live weights, so the "
            "per-checkpoint EMA copy is unreachable to anything loading the root as MODEL_PATH."
        },
    )

    # Loss configs
    loss_implementation: str = field(default="torch")
    loss_reduction_scope: str = field(default="sequence")
    average_tokens_across_devices: bool = field(default=True)

    # Eval configs
    eval_strategy: Union[IntervalStrategy, str] = field(
        default="no",
        metadata={"help": "The evaluation strategy to use."},
    )
    eval_steps: Optional[float] = field(default=None)

    # Log configs
    output_dir: str = field(default="outputs")
    log_flops: bool = field(default=False)
    log_seen_tokens: bool = field(default=False)
    report_to: Optional[List[str]] = field(
        default=None, metadata={"help": "The list of integrations to report the results and logs to."}
    )

    logging_strategy: Union[IntervalStrategy, str] = field(
        default="steps",
        metadata={"help": "The logging strategy to use."},
    )
    logging_steps: int = field(default=10)
    logging_first_step: bool = field(default=False, metadata={"help": "Log the first global_step"})
    log_level: str = field(default="info", metadata={"choices": [item.value for item in logging.LogLevel]})
    log_level_replica: str = field(default="warning", metadata={"choices": [item.value for item in logging.LogLevel]})
    disable_tqdm: bool = field(default=False)

    save_strategy: Union[SaveStrategy, str] = field(
        default="steps",
        metadata={"help": "The checkpoint save strategy to use."},
    )
    save_steps: int = field(default=1000)
    save_total_limit: Optional[int] = field(default=None)
    save_keep_every: Optional[int] = field(
        default=None,
        metadata={
            "help": "Step interval whose checkpoints are permanently exempt from rotation and "
            "do not consume the save_total_limit window. Lets a long run hold a short rolling "
            "window for crash recovery while keeping milestones a later stage may transfer "
            "better from. MUST be a multiple of save_steps, or no checkpoint ever lands on it."
        },
    )
    save_full_model: bool = field(default=True)

    restore_callback_states_from_checkpoint: bool = field(default=False)

    # Misc
    synchronize_experts_before_forward: bool = field(default=False)
    cleanup_before_optimizer_step: bool = field(default=False)

    # Reproducibility
    seed: int = field(default=42)
    full_determinism: bool = field(default=False)

    def __post_init__(self):
        super().__post_init__()

        # HF's create_scheduler splats lr_scheduler_kwargs as **scheduler_specific_kwargs,
        # and this dataclass does not inherit HF's TrainingArguments, so nothing else
        # decodes a string set programmatically (e.g. from a JSON config).
        if isinstance(self.lr_scheduler_kwargs, str):
            self.lr_scheduler_kwargs = json.loads(self.lr_scheduler_kwargs) if self.lr_scheduler_kwargs else {}
        if self.min_lr_rate is not None:
            self.lr_scheduler_kwargs = {**self.lr_scheduler_kwargs, "min_lr_rate": self.min_lr_rate}

        seed = torch.tensor(self.seed, device="cuda")
        torch.distributed.broadcast(seed, src=0)
        self.seed = seed.item()

        if self.encoder_gradient_checkpointing_interval is not None:
            assert self.gradient_checkpointing
            assert self.encoder_gradient_checkpointing_interval > 0

        if self.sequence_packing:
            # assert disabled for testing
            pass

        if self.decoder_load_balancing:
            assert self.sequence_packing, "DP load balancing requires batch flattening."
            assert not self.dynamic_batching, "DP load balancing and dynamic batching cannot be used together."

        if self.dynamic_batching:
            assert self.sequence_packing, "Dynamic batching requires batch flattening."
            assert not self.decoder_load_balancing, "Dynamic batching and workload balancing cannot be used together."

        assert self.loss_reduction_scope in ["batch", "sequence"], (
            f"Unsupported loss reduction scope: {self.loss_reduction_scope}"
        )
        if self.loss_reduction_scope == "sequence":
            assert self.average_tokens_across_devices

        self.logging_dir = self.output_dir
        self.log_level = logging.LogLevel(self.log_level)
        self.log_level_replica = logging.LogLevel(self.log_level_replica)
        log_level = self.log_level if self.global_rank == 0 else self.log_level_replica
        logging.set_verbosity(log_level)

        self.eval_strategy = IntervalStrategy(self.eval_strategy)
        self.logging_strategy = IntervalStrategy(self.logging_strategy)
        self.save_strategy = SaveStrategy(self.save_strategy)

        for attr in ["log_flops", "log_seen_tokens"]:
            if getattr(self, attr):
                logger.warn(f"The `{attr}` argument can only be used for debugging.")

        assert self.deepspeed is not None, "DeepSpeed config path is required."
        assert os.path.isfile(self.deepspeed)
        with open(self.deepspeed, "r") as f:
            deepspeed_config = json.load(f)
        self.deepspeed_config = self._process_deepspeed_config(deepspeed_config)

        if self.synchronize_experts_before_forward:
            assert self.ep_world_size > 1

    def _process_deepspeed_config(self, deepspeed_config: Dict[str, Any]):
        try:
            config = AutoConfig.from_pretrained(self.model_path)
            hidden_size = config.get_text_config().hidden_size
        except Exception:
            # A resumed RynnVLA checkpoint has a custom model_type, registered later
            # by build_processor. Bucket sizing only needs its text width, so read
            # the metadata without requiring AutoConfig registration or model imports.
            try:
                raw_config, _ = PreTrainedConfig.get_config_dict(self.model_path)
                text_config = raw_config.get("text_config") or raw_config
                hidden_size = text_config.get("hidden_size")
            except Exception:
                hidden_size = None

        def _process_auto(config, prefix=""):
            config = config.copy()
            for key, value in config.items():
                global_key = prefix + key
                if isinstance(value, dict):
                    config[key] = _process_auto(value, prefix=global_key + ".")
                elif value == "auto":
                    if global_key == "train_micro_batch_size_per_gpu":
                        config[key] = self.micro_batch_size
                    elif global_key == "gradient_accumulation_steps":
                        config[key] = self.gradient_accumulation_steps
                    elif global_key == "gradient_clipping":
                        config[key] = self.max_grad_norm
                    elif global_key == "fp16.enabled":
                        config[key] = self.fp16
                    elif global_key == "bf16.enabled":
                        config[key] = self.bf16
                    elif global_key == "zero_optimization.reduce_bucket_size":
                        if not isinstance(hidden_size, int) or hidden_size <= 0:
                            raise ValueError(
                                f"Cannot resolve {global_key}=auto: no valid text hidden_size in {self.model_path}"
                            )
                        config[key] = hidden_size * hidden_size
                    elif global_key == "zero_optimization.stage3_prefetch_bucket_size":
                        assert hidden_size is not None
                        config[key] = int(0.9 * hidden_size * hidden_size)
                    elif global_key == "zero_optimization.stage3_param_persistence_threshold":
                        assert hidden_size is not None
                        config[key] = 10 * hidden_size
                    else:
                        raise ValueError(f"Unsupported auto config: {key}")
            return config

        return _process_auto(deepspeed_config)

    def get_warmup_steps(self, num_training_steps: int):
        warmup_steps = (
            self.warmup_steps if self.warmup_steps > 0 else math.ceil(num_training_steps * self.warmup_ratio)
        )
        return warmup_steps


@dataclass
class EvaluationArguments(ModelArguments):
    benchmarks: List[str] = field(default=None)
    prompt_format: str = field(default=None)
    enable_thinking: bool = field(default=False)
    save_dir: str = field(default=None)
    save_rollout: bool = field(default=False)

    engine: str = field(default="hf", metadata={"choices": ["hf", "sglang", "vla"]})
    num_processor_workers: int = field(default=8)
    max_concurrency: int = field(default=128)
    max_running_requests: int = field(default=16)

    image_min_pixels: int = field(default=16 * 32 * 32)
    image_max_pixels: int = field(default=16384 * 32 * 32)
    video_min_pixels: int = field(default=16 * 32 * 32)
    video_max_pixels: int = field(default=16384 * 32 * 32)

    fps: int = field(default=1)
    max_frames: int = field(default=180)

    max_new_tokens: int = field(default=128)
    temperature: float = field(default=0.0)
    top_p: float = field(default=0.95)
    top_k: int = field(default=50)
    repetition_penalty: Optional[float] = field(default=None)

    tensor_parallel_size: int = field(default=1)
    expert_parallel_size: int = field(default=1)
    pipeline_parallel_size: int = field(default=1)

    def __post_init__(self):
        super().__post_init__()

        assert self.benchmarks is not None
        assert self.save_dir is not None

        self.processing_params = {
            "image_max_pixels": self.image_max_pixels,
            "image_min_pixels": self.image_min_pixels,
            "video_max_pixels": self.video_max_pixels,
            "video_min_pixels": self.video_min_pixels,
            "fps": self.fps,
            "max_frames": self.max_frames,
        }

        self.sampling_params = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
        }
        if self.repetition_penalty is not None:
            self.sampling_params["repetition_penalty"] = self.repetition_penalty

        if self.engine != "sglang":
            assert self.tensor_parallel_size == 1
            assert self.expert_parallel_size == 1

        self.parallel_params = {
            "tp_size": self.tensor_parallel_size,
            "ep_size": self.expert_parallel_size,
            "pp_size": self.pipeline_parallel_size,
        }


@dataclass
class ReplayArguments(ModelArguments, DataArguments):
    save_dir: Optional[str] = field(default=None)
    video_fps: int = field(default=20)
    video_height: int = field(default=720)
    render_size: int = field(default=320)
    num_segments: int = field(
        default=1,
        metadata={"help": "Total number of segments to replay. Each segment is sampled by first drawing a random episode, then a random segment within it."},
    )
    num_inference_steps: int = field(default=10)
    seed: int = field(default=42)
    episode_indices: Optional[List[int]] = field(
        default=None,
        metadata={"help": "If set, replay exactly these flat episode indices (overrides random seed sampling and num_episodes)."},
    )
    force_joint: bool = field(
        default=False,
        metadata={"help": "Force renderer to use recorded joint angles instead of IK-solving EEF poses."},
    )
    max_segments_per_episode: int = field(
        default=0,
        metadata={"help": "If >0, cap each replayed episode to this many segments (useful to keep demo runs bounded for very long episodes)."},
    )

    def __post_init__(self):
        # ``ModelArguments.__post_init__`` asserts ``model_path is not None``
        # and probes the checkpoint for ``model_type``. When replaying
        # training data only (no model), skip those steps.
        if self.model_path is not None:
            super().__post_init__()
            return

        DataArguments.__post_init__(self)

        if isinstance(self.config_overrides, str):
            self.config_overrides = json.loads(self.config_overrides)
        elif self.config_overrides is None:
            self.config_overrides = {}

        if isinstance(self.processor_overrides, str):
            self.processor_overrides = json.loads(self.processor_overrides)
        elif self.processor_overrides is None:
            self.processor_overrides = {}

        if self.bf16:
            self.dtype = torch.bfloat16
        elif self.fp16:
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32
