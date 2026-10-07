"""Portable, flat configuration for the released RynnLAM training path."""

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple


@dataclass
class RynnLAMConfig:
    # DA3-Large and the active self-attention architecture.
    patch_size: int = 14
    encoder_backbone: str = "large"
    embed_dim: int = 1024
    encoder_finetune_mode: str = "freeze"
    encoder_checkpoint_path: Optional[str] = None
    latent_dim: int = 64
    latent_encoder_type: str = "selfattn"
    latent_encoder_model_dim: int = 512
    latent_encoder_num_heads: int = 16
    latent_encoder_num_blocks: int = 6
    camera_pose_latent_dim: int = 32
    motion_hint_dim: int = 128
    motion_hints_dropout: float = 0.8
    flow_decoder_version: str = "v5"
    flow_decoder_model_dim: int = 512
    flow_decoder_num_heads: int = 16
    flow_decoder_dec_blocks: int = 8
    flow_mask_ratio: float = 0.5
    warp_num_sample_points: int = 8
    cam_dec_hidden_dim: int = 256
    use_k_token_recon: bool = True
    num_k_tokens: int = 8
    k_token_source: str = "features"
    k_token_bottleneck_dim: int = 0
    recon_decoder_model_dim: int = 512
    recon_decoder_num_heads: int = 16
    recon_decoder_num_blocks: int = 4
    adversarial_alpha: float = 2.0
    adversarial_warmup_steps: int = 2000
    use_rope: bool = False
    log_z_utilization: bool = True

    # Local safetensors are the default. No dataset location is inferred.
    data_root: str = ""
    depth_root: str = ""
    flow_root: str = ""
    safetensors_root: str = ""
    manifest_path: Optional[str] = None
    cache_dir: Optional[str] = None
    num_workers: int = 16
    batch_size: int = 16
    depth_grad_threshold: float = 0.02
    flow_max_radius: float = 20.0
    max_scenes: int = 0
    max_frame_stride: int = 10
    min_frame_stride: int = 5
    max_sample_stride: int = 15
    target_hw: Optional[Tuple[int, int]] = (238, 322)
    dataset_sampling_temperature: float = 2.0
    dataset_sampling_weights: Optional[Dict[str, float]] = None
    dataset_role_sampling_weights: Optional[Dict[str, float]] = None

    lr: float = 5e-5
    encoder_lr: float = 1e-5
    lr_warmup_steps: int = 1000  # Successful optimizer updates.
    gradient_accumulation_steps: int = 1
    target_effective_batch: int = 0
    # If target_effective_batch > 0: optimizer-update budget; otherwise batch budget.
    # global_step, loss warmups and save_interval_steps always count per-rank batches.
    max_steps: int = 0
    num_epochs: int = 1
    lambda_flow: float = 5.0
    lambda_pose: float = 5.0
    lambda_feat: float = 3.0
    lambda_adversarial: float = 2.0
    lambda_recon: float = 5.0
    use_z_residual_loss: bool = True
    z_recon_warmup_steps: int = 800
    flow_loss_beta: float = 0.05
    flow_warmup_steps: int = 2000
    use_contrastive_feat_loss: bool = True
    contrast_temperature: float = 0.2

    device: str = "auto"
    exp_name: str = "stage1"
    output_dir: str = "./runs"
    track: bool = False   # append metrics to <output_dir>/<exp_name>/metrics.jsonl
    log_interval: int = 50
    save_interval: int = 1
    save_interval_steps: int = 2000
    eval_interval: int = 1
    vis_interval: int = 100
    val_scene_id: Optional[str] = None
    resume: Optional[str] = None
    reset_optimizer: bool = (
        False  # Strict weights-only load; reset all training counters.
    )

    @classmethod
    def from_dict(cls, raw: Mapping, *, strict: bool = False) -> "RynnLAMConfig":
        """Read flat checkpoint config or nested YAML sections.

        Unknown historical checkpoint fields are ignored unless strict=True.
        YAML loading is strict so misspelled training options cannot be silent.
        """
        if not isinstance(raw, Mapping):
            raise ValueError("Configuration must be a mapping")
        flat = {}
        for section in ("data", "model", "training", "logging"):
            values = raw.get(section, {})
            if not isinstance(values, Mapping):
                raise ValueError(f"{section} must be a mapping")
            flat.update(values)
        flat.update(
            {
                k: v
                for k, v in raw.items()
                if k not in ("data", "model", "training", "logging")
            }
        )
        for old, new in (("meta_json", "data_root"), ("output_root", "output_dir")):
            if old in flat:
                flat.setdefault(new, flat.pop(old))
        known = {f.name: f for f in fields(cls)}
        unknown = set(flat) - set(known)
        if strict and unknown:
            raise ValueError(
                f"Unknown configuration fields: {', '.join(sorted(unknown))}"
            )
        values = {}
        for key, value in flat.items():
            if key not in known:
                continue
            kind = known[key].type
            if kind is bool and not isinstance(value, bool):
                raise ValueError(f"{key} must be a boolean")
            if kind in (int, float):
                value = kind(value)
            if (
                key in ("dataset_sampling_weights", "dataset_role_sampling_weights")
                and value is not None
            ):
                value = {str(k): float(v) for k, v in value.items()}
            if key == "target_hw" and value is not None:
                value = tuple(value)
            values[key] = value
        return cls(**values)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "RynnLAMConfig":
        import yaml

        with Path(yaml_path).open() as stream:
            return cls.from_dict(yaml.safe_load(stream), strict=True)

    def validate_training(self):
        """Validate before loading a model, data, or initializing distributed workers."""
        if (
            self.latent_encoder_type != "selfattn"
            or self.flow_decoder_version != "v5"
            or not self.use_k_token_recon
        ):
            raise ValueError(
                "Release supports selfattn, FlowDecoderV5 and K-token reconstruction only"
            )
        if self.encoder_finetune_mode not in ("freeze", "full"):
            raise ValueError("encoder_finetune_mode must be freeze or full")
        if self.encoder_finetune_mode == "full" and not self.resume:
            raise ValueError(
                "Stage 2 full finetuning requires explicit --resume or config resume"
            )
        if self.reset_optimizer and not self.resume:
            raise ValueError("reset_optimizer requires a resume checkpoint")
        if not self.manifest_path:
            raise ValueError("Set data.manifest_path explicitly")
        if not self.safetensors_root:
            raise ValueError("Set data.safetensors_root explicitly")
        if (
            self.batch_size < 1
            or self.gradient_accumulation_steps < 1
            or self.num_epochs < 1
        ):
            raise ValueError(
                "batch_size, gradient_accumulation_steps and num_epochs must be positive"
            )
        if (
            self.max_steps < 0
            or self.target_effective_batch < 0
            or self.num_workers < 0
        ):
            raise ValueError(
                "Step budgets, target_effective_batch and num_workers cannot be negative"
            )
        if self.log_interval < 1:
            raise ValueError("log_interval must be positive")
