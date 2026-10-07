#!/usr/bin/env python3
"""RynnLAM training with torchrun or single-process Python (including CPU).

batch_size is per rank. global_step, checkpoint names, save intervals and loss
warmups count per-rank batches. With target_effective_batch > 0, max_steps counts
successful optimizer updates; otherwise max_steps counts per-rank batches.
LR warmup always counts successful optimizer updates. Resetting the optimizer
loads weights strictly and restarts all counters; normal resume restores them.
"""

import os
import sys
import argparse
import hashlib
import json
import math
from collections import defaultdict, deque
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rynnlam.config import RynnLAMConfig
from rynnlam.model import build_model
from rynnlam.dataset_epic import (
    BucketedDistributedBatchSampler,
    LAMEpicDataset,
    WeightedMultiDatasetBatchSampler,
    lam_collate_fn,
)
from rynnlam.modules.losses_v5 import compute_lam_v5_losses, LAMv5LossConfig
from rynnlam.logger import logger

# ==============================================================
# DDP utilities
# ==============================================================


def setup_ddp(device="auto"):
    """Initialize torchrun workers; ordinary Python needs no process group."""
    from datetime import timedelta

    use_cuda = device != "cpu" and torch.cuda.is_available()
    if device == "cuda" and not use_cuda:
        raise RuntimeError("CUDA was requested but is unavailable")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if use_cuda:
        torch.cuda.set_device(local_rank)
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        dist.init_process_group(
            backend="nccl" if use_cuda else "gloo", timeout=timedelta(minutes=60)
        )
    return local_rank


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def get_rank():
    return dist.get_rank() if dist.is_initialized() else 0


def barrier():
    if dist.is_initialized():
        dist.barrier()


def is_main():
    return get_rank() == 0


def get_world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


# ==============================================================
# Data
# ==============================================================


def build_data(config):
    """Create dataset + DistributedSampler + DataLoader."""
    has_safetensors = bool(getattr(config, "safetensors_root", None))
    # When safetensors_root is provided, we can build the scene list directly
    # from manifest_*.json files and skip the legacy meta_json.
    meta_json = (
        config.data_root if config.data_root else (None if has_safetensors else None)
    )
    if meta_json is None and not has_safetensors:
        raise ValueError(
            "Either data_root (meta_json) or safetensors_root must be provided."
        )

    dataset = LAMEpicDataset(
        meta_json=meta_json,
        depth_root=config.depth_root,
        flow_root=config.flow_root,
        safetensors_root=getattr(config, "safetensors_root", None),
        flow_max_radius=config.flow_max_radius,
        depth_grad_threshold=config.depth_grad_threshold,
        max_scenes=config.max_scenes,
        max_frame_stride=config.max_frame_stride,
        min_frame_stride=getattr(config, "min_frame_stride", 1),
        max_sample_stride=getattr(config, "max_sample_stride", None),
        target_hw=config.target_hw,
        manifest_path=config.manifest_path,
        cache_dir=config.cache_dir,
    )

    # Preprocessed data may span several aspect-ratio buckets (e.g. RoboMIND has
    # both 238x322 and 210x364 scenes). lam_collate_fn stacks tensors, so a batch
    # must never mix resolutions — use the bucketed batch sampler in that case.
    resolutions = {
        (e.get("height"), e.get("width"))
        for e in getattr(dataset, "_st_meta", {}).values()
    }
    resolutions.discard((None, None))

    ds_temp = config.dataset_sampling_temperature
    ds_weights = config.dataset_sampling_weights
    role_weights = config.dataset_role_sampling_weights
    use_weighted = (abs(ds_temp - 1.0) > 1e-9) or bool(ds_weights) or bool(role_weights)
    if len(resolutions) > 1 or use_weighted:
        if use_weighted:
            if is_main():
                logger.info(
                    f"Dataset spans {len(resolutions)} resolutions, multi-dataset sampling "
                    f"(T={ds_temp}, multipliers={ds_weights}, role_multipliers={role_weights}) "
                    f"-> WeightedMultiDatasetBatchSampler"
                )
            sampler = WeightedMultiDatasetBatchSampler(
                dataset,
                batch_size=config.batch_size,
                num_replicas=get_world_size(),
                rank=get_rank(),
                temperature=ds_temp,
                dataset_multipliers=ds_weights,
                role_multipliers=role_weights,
                shuffle=True,
                drop_last=False,
            )
        else:
            if is_main():
                logger.info(
                    f"Dataset spans {len(resolutions)} resolutions {sorted(resolutions)} "
                    f"-> using BucketedDistributedBatchSampler"
                )
            sampler = BucketedDistributedBatchSampler(
                dataset,
                batch_size=config.batch_size,
                num_replicas=get_world_size(),
                rank=get_rank(),
                shuffle=True,
                drop_last=False,
            )
        # batch_sampler is mutually exclusive with batch_size/sampler/drop_last.
        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=config.num_workers,
            collate_fn=lam_collate_fn,
            pin_memory=True,
            persistent_workers=config.num_workers > 0,
            prefetch_factor=4 if config.num_workers > 0 else None,
        )
        return dataset, sampler, loader

    sampler = DistributedSampler(
        dataset,
        num_replicas=get_world_size(),
        rank=get_rank(),
        shuffle=True,
        drop_last=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.num_workers,
        collate_fn=lam_collate_fn,
        pin_memory=True,
        drop_last=False,
        persistent_workers=config.num_workers > 0,
        prefetch_factor=4 if config.num_workers > 0 else None,
    )

    return dataset, sampler, loader


# ==============================================================
# Visualization helpers (rank 0 only)
# ==============================================================

TRAJ_STRIDE = 16


def _to_uint8_rgb(img):
    if img.dtype == np.uint8:
        return img
    return (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)


def _flow_to_color(dx, dy):
    angle = np.arctan2(dy, dx)
    hue = int((angle + np.pi) / (2 * np.pi) * 179)
    hsv = np.array([[[hue, 255, 255]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return tuple(int(x) for x in bgr)


def _draw_pair_lines(frame_rgb, flow2d, mask2d, stride=TRAJ_STRIDE):
    H, W = frame_rgb.shape[:2]
    result = frame_rgb.copy()
    ys = np.arange(stride // 2, H, stride)
    xs = np.arange(stride // 2, W, stride)
    for y in ys:
        for x in xs:
            if not mask2d[y, x]:
                continue
            dx, dy = flow2d[y, x]
            x2, y2 = int(round(x + dx)), int(round(y + dy))
            if 0 <= x2 < W and 0 <= y2 < H:
                color = _flow_to_color(float(dx), float(dy))
                cv2.line(
                    result,
                    (int(x), int(y)),
                    (x2, y2),
                    color=color,
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )
    return result


def _features_to_pca_rgb(features, nH, nW, H, W, pca_components=None):
    N, D = features.shape
    mean = features.mean(axis=0, keepdims=True)
    centered = features - mean
    if pca_components is None:
        if D > 1024:
            rng = np.random.default_rng(42)
            proj = rng.standard_normal((D, 256)).astype(np.float32)
            proj /= np.linalg.norm(proj, axis=0, keepdims=True)
            reduced = centered @ proj
            _, _, Vt = np.linalg.svd(reduced, full_matrices=False)
            pca_components = Vt[:3] @ proj.T
        else:
            _, _, Vt = np.linalg.svd(centered, full_matrices=False)
            pca_components = Vt[:3]
    projected = centered @ pca_components.T
    for c in range(3):
        ch = projected[:, c]
        cmin, cmax = ch.min(), ch.max()
        projected[:, c] = (ch - cmin) / (cmax - cmin) if cmax - cmin > 1e-8 else 0.5
    feat_map = projected.reshape(nH, nW, 3)
    feat_rgb = cv2.resize(feat_map, (W, H), interpolation=cv2.INTER_NEAREST)
    return (feat_rgb * 255).astype(np.uint8), pca_components


def _cosine_sim_map(feat_a, feat_b, nH, nW, H, W):
    a_n = feat_a / (np.linalg.norm(feat_a, axis=-1, keepdims=True) + 1e-6)
    b_n = feat_b / (np.linalg.norm(feat_b, axis=-1, keepdims=True) + 1e-6)
    sim = np.clip((a_n * b_n).sum(axis=-1).reshape(nH, nW), 0, 1)
    sim_up = cv2.resize(sim, (W, H), interpolation=cv2.INTER_NEAREST)
    color = cv2.applyColorMap((sim_up * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.cvtColor(color, cv2.COLOR_BGR2RGB)


# ==============================================================
# DDP Trainer
# ==============================================================


class DDPTrainer:
    """Single-node multi-GPU DDP trainer for LAM."""

    def __init__(self, config, local_rank):
        self.config = config
        self.local_rank = local_rank
        self.rank = get_rank()
        self.world_size = get_world_size()
        use_cuda = config.device != "cpu" and torch.cuda.is_available()
        self.device = torch.device(f"cuda:{local_rank}" if use_cuda else "cpu")

        # Paths are caller-configured, relative to the working directory.
        self.save_dir = Path(config.output_dir) / config.exp_name
        if is_main():
            self.save_dir.mkdir(parents=True, exist_ok=True)

        # Metrics log (rank 0 only): one JSON object per line under the run directory.
        self.metrics_path = (
            self.save_dir / "metrics.jsonl"
            if getattr(config, "track", False) and is_main()
            else None
        )

        # A full resume checkpoint already includes the backbone weights.
        model_config = (
            replace(config, encoder_checkpoint_path=None) if config.resume else config
        )
        self.model_raw = build_model(model_config).to(self.device)
        # Diagnostic-only: enable the z-utilization measurement pass (no training change).
        self.model_raw.log_z_utilization = getattr(config, "log_z_utilization", False)
        self.patch_size = self.model_raw.patch_size  # save before DDP wrap

        self.model = self.model_raw
        if dist.is_initialized():
            self.model = DDP(
                self.model_raw,
                device_ids=[local_rank] if use_cuda else None,
                output_device=local_rank if use_cuda else None,
                find_unused_parameters=True,
            )

        # ---- Optimizer (supports encoder_lr param groups for stage-2) ----
        # NOTE: float() guards against YAML scientific notation like '1e-5' being
        # parsed as a str (YAML only treats '1.0e-5' as a float), which would make
        # a param-group lr a string and crash LambdaLR ("can't multiply sequence").
        encoder_lr = float(getattr(config, "encoder_lr", config.lr))
        encoder_params, other_params = [], []
        for name, p in self.model_raw.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("encoder."):
                encoder_params.append(p)
            else:
                other_params.append(p)
        param_groups = [{"params": other_params, "lr": float(config.lr)}]
        if len(encoder_params) > 0:
            param_groups.append({"params": encoder_params, "lr": encoder_lr})
            if is_main():
                logger.info(
                    f"Optimizer: {len(other_params)} decoder params @ lr={config.lr}, "
                    f"{len(encoder_params)} encoder params @ lr={encoder_lr}"
                )
        self.optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)

        # ---- LR Scheduler (with optional linear warmup) ----
        self.lr_warmup_steps = getattr(config, "lr_warmup_steps", 0)
        self.max_steps = int(getattr(config, "max_steps", 0) or 0)
        if self.lr_warmup_steps >= 0:
            # Per-update scheduler: optional linear warmup then cosine decay.
            # self._total_steps is provisional here; train() sets the real value
            # from len(train_loader) once the dataloader exists.
            self._base_lr = float(config.lr)
            self._total_steps = max(
                getattr(config, "num_epochs", 2) * 20000, self.lr_warmup_steps + 1
            )
            warmup = self.lr_warmup_steps

            def lr_lambda(step):
                if step < warmup:
                    return step / max(warmup, 1)
                progress = (step - warmup) / max(self._total_steps - warmup, 1)
                progress = min(progress, 1.0)  # clamp: monotonic decay, never rise back
                return max(
                    1e-6 / self._base_lr, 0.5 * (1 + math.cos(progress * math.pi))
                )

            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda
            )
            self.scheduler_per_step = True
        else:
            raise ValueError("lr_warmup_steps cannot be negative")

        # ---- AMP ----
        self.use_amp = use_cuda
        # bf16 has the same exponent range as fp32: no forward/backward overflow,
        # no GradScaler needed. fp16 stage2 runs collapsed (scaler scale -> 32,
        # per-rank NaN storms). Fall back to fp16 only if bf16 unsupported.
        self.amp_dtype = (
            torch.bfloat16
            if not use_cuda or torch.cuda.is_bf16_supported()
            else torch.float16
        )
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.use_amp and self.amp_dtype == torch.float16
        )
        if is_main():
            logger.info(
                f"AMP dtype: {self.amp_dtype}, GradScaler enabled: {self.scaler.is_enabled()}"
            )

        # ---- Loss ----
        self.loss_config = LAMv5LossConfig(
            lambda_pose=getattr(config, "lambda_pose", 1.0),
            lambda_flow=config.lambda_flow,
            lambda_feat=getattr(config, "lambda_feat", 1.0),
            lambda_adversarial=getattr(config, "lambda_adversarial", 0.5),
            flow_loss_beta=getattr(config, "flow_loss_beta", 0.05),
            use_contrastive_feat_loss=getattr(
                config, "use_contrastive_feat_loss", False
            ),
            contrast_temperature=getattr(config, "contrast_temperature", 0.1),
            lambda_recon=getattr(config, "lambda_recon", 0.0),
            use_z_residual_loss=getattr(config, "use_z_residual_loss", False),
            z_recon_warmup_steps=getattr(config, "z_recon_warmup_steps", 0),
        )
        self.flow_warmup_steps = getattr(config, "flow_warmup_steps", 2000)

        # ---- Adversarial alpha warmup (GRL strength ramps 0 -> target) ----
        self.adversarial_alpha_target = getattr(config, "adversarial_alpha", 1.0)
        self.adversarial_warmup_steps = getattr(config, "adversarial_warmup_steps", 0)
        if self.adversarial_warmup_steps > 0:
            self.model_raw.adversarial_alpha = 0.0  # start disabled, ramp up

        # ---- State ----
        self.global_step = 0
        self.optimizer_step = 0
        self.batches_in_epoch = 0
        self.current_epoch = 0
        self.log_step = 0
        self.best_loss = float("inf")
        self.accum_steps = getattr(config, "gradient_accumulation_steps", 1)

        self.target_eff_batch = int(config.target_effective_batch)
        self.max_optimizer_steps = self.max_steps if self.target_eff_batch else 0
        if self.target_eff_batch:
            base = config.batch_size * self.world_size
            if self.target_eff_batch < base or self.target_eff_batch % base:
                raise ValueError(
                    "target_effective_batch must be a positive multiple of batch_size * world_size"
                )
            self.accum_steps = self.target_eff_batch // base
            self.max_steps = 0  # budget is checked in optimizer units instead

        # ---- Loss tracking (sliding window for convergence monitoring) ----
        self._loss_window_size = 100  # rolling average window
        self._loss_windows = defaultdict(lambda: deque(maxlen=self._loss_window_size))
        # Per-dataset flow loss: flow_loss is an absolute metric-scale quantity and
        # per-dataset motion magnitudes differ by ~10x, so the global mean alone
        # can't tell a uniform regression from a few large-motion domains
        # dominating it. Diagnostic only — never feeds gradients.
        self._flow_by_dataset = defaultdict(
            lambda: deque(maxlen=self._loss_window_size)
        )

        # Data (set up later)
        self.dataset = None
        self.sampler = None
        self.train_loader = None

        if is_main():
            total_p = sum(p.numel() for p in self.model_raw.parameters())
            train_p = sum(
                p.numel() for p in self.model_raw.parameters() if p.requires_grad
            )
            eff_batch = config.batch_size * self.world_size * self.accum_steps
            logger.info(
                f"DDPTrainer: {self.world_size} GPUs, "
                f"per-GPU batch={config.batch_size}, accum={self.accum_steps}, "
                f"effective batch={eff_batch}"
                + (
                    f" [target_effective_batch={self.target_eff_batch}]"
                    if self.target_eff_batch > 0
                    else ""
                )
            )
            logger.info(f"  Params total={total_p:,}  trainable={train_p:,}")

    def _log_metrics(self, data, step):
        """Append one metrics record to the local JSONL log (rank 0, when tracking)."""
        if self.metrics_path is None:
            return
        record = {"step": step, **data}
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    # ============================================================
    # Data
    # ============================================================

    def setup_data(self):
        self.dataset, self.sampler, self.train_loader = build_data(self.config)
        self._resume_data_layout()
        if is_main():
            logger.info(
                f"Dataset: {len(self.dataset)} pairs, "
                f"{len(self.dataset.scene_list)} scenes, "
                f"stride=[{getattr(self.config, 'min_frame_stride', 1)}, "
                f"{self.config.max_frame_stride}]"
            )

    def _move_batch(self, batch):
        return {
            k: (
                v.to(self.device, non_blocking=True)
                if isinstance(v, torch.Tensor)
                else v
            )
            for k, v in batch.items()
        }

    # ============================================================
    # Training
    # ============================================================

    def budget_complete(self):
        return (self.max_steps > 0 and self.global_step >= self.max_steps) or (
            self.max_optimizer_steps > 0
            and self.optimizer_step >= self.max_optimizer_steps
        )

    def train_epoch(self, epoch):
        if self.budget_complete():
            return {}
        self.model.train()
        self.sampler.set_epoch(epoch)  # essential for proper shuffling
        self.optimizer.zero_grad()

        # Reset sliding windows each epoch
        self._loss_windows.clear()
        self._flow_by_dataset.clear()

        epoch_losses = []
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1}",
            disable=not is_main(),
        )

        pending_batches = 0
        resume_batches = self.batches_in_epoch
        checkpoint_due = False
        for batch_idx, batch in enumerate(pbar):
            if self.budget_complete():
                break
            if batch_idx < resume_batches:
                continue
            batch = self._move_batch(batch)
            self.batches_in_epoch = batch_idx + 1

            # Ramp GRL adversarial strength 0 -> target over warmup steps
            if self.adversarial_warmup_steps > 0:
                alpha = self.adversarial_alpha_target * min(
                    1.0, self.global_step / self.adversarial_warmup_steps
                )
                self.model_raw.adversarial_alpha = alpha

            # The final partial window must synchronize its backward pass too.
            last_batch = batch_idx + 1 == len(self.train_loader)
            last_budget_batch = (
                self.max_steps > 0 and self.global_step + 1 >= self.max_steps
            )
            is_accum_step = pending_batches + 1 < self.accum_steps and not (
                last_batch or last_budget_batch
            )
            sync_ctx = (
                self.model.no_sync
                if is_accum_step and isinstance(self.model, DDP)
                else nullcontext
            )

            with sync_ctx():
                with torch.amp.autocast(
                    self.device.type, dtype=self.amp_dtype, enabled=self.use_amp
                ):
                    output = self.model(
                        images=batch["images"],
                        extrinsics=batch["extrinsics"],
                        intrinsics=batch["intrinsics"],
                        depths=batch["depths"],
                    )
                    target = {
                        "flows": batch["flow"],
                        "mask": batch["mask"],
                        "extrinsics": batch["extrinsics"],
                    }
                    compute_loss_fn = compute_lam_v5_losses
                    losses = compute_loss_fn(
                        output=output,
                        target=target,
                        loss_config=self.loss_config,
                        global_step=self.global_step,
                        flow_warmup_steps=self.flow_warmup_steps,
                    )

                # Skip batches that produce NaN/Inf loss to protect optimizer state.
                # Must be a collective decision: if only one rank skips, the others
                # block in gradient allreduce until NCCL times out (SIGABRT).
                skip_flag = (~torch.isfinite(losses["total_loss"])).to(
                    dtype=torch.float32, device=losses["total_loss"].device
                )
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(skip_flag, op=dist.ReduceOp.MAX)
                if skip_flag.item() > 0:
                    self._consec_nan_skips = getattr(self, "_consec_nan_skips", 0) + 1
                    if is_main():
                        logger.warning(
                            f"NaN/Inf total_loss at step {self.global_step}, skipping batch (all ranks)"
                        )
                    if self._consec_nan_skips >= 100:
                        # Weights are almost certainly NaN; zombie-stepping would
                        # burn steps and overwrite milestone ckpts with garbage.
                        raise RuntimeError(
                            f"{self._consec_nan_skips} consecutive NaN/Inf batches at step "
                            f"{self.global_step}: model has diverged, aborting. Resume from an "
                            f"earlier checkpoint with a lower LR."
                        )
                    # Complete DDP's reducer cycle even when the loss is invalid.
                    # nan_to_num keeps every output graph connected; resulting gradients
                    # are discarded immediately on every rank.
                    if isinstance(self.model, DDP):
                        connected = [
                            v
                            for v in output.values()
                            if isinstance(v, torch.Tensor) and v.requires_grad
                        ]
                        if connected:
                            sum(
                                torch.nan_to_num(v).mul(0).sum() for v in connected
                            ).backward()
                    self.optimizer.zero_grad(set_to_none=True)
                    pending_batches = 0
                    self.global_step += 1
                    # Don't miss milestone checkpoints when the boundary step is skipped
                    save_interval_steps = getattr(self.config, "save_interval_steps", 0)
                    if (
                        save_interval_steps > 0
                        and self.global_step % save_interval_steps == 0
                    ):
                        if is_main():
                            self.save_step_checkpoint(epoch)
                        barrier()
                    continue
                self._consec_nan_skips = 0

                self.scaler.scale(losses["total_loss"] / self.accum_steps).backward()
                pending_batches += 1

            # Normalize a final partial window by its actual batch count.
            if not is_accum_step:
                self.scaler.unscale_(self.optimizer)
                if pending_batches != self.accum_steps:
                    for parameter in self.model_raw.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(self.accum_steps / pending_batches)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0
                )
                bad_grad = (~torch.isfinite(grad_norm)).float()
                if dist.is_initialized():
                    dist.all_reduce(bad_grad, op=dist.ReduceOp.MAX)
                if bad_grad.item():
                    if is_main():
                        logger.warning(
                            f"Non-finite grad_norm at step {self.global_step}, skipping optimizer step"
                        )
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.update()
                else:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.optimizer_step += 1
                    if self.scheduler_per_step:
                        self.scheduler.step()
                pending_batches = 0

            self.global_step += 1

            # Never checkpoint unsaved accumulated gradients: defer to the next boundary.
            save_interval_steps = self.config.save_interval_steps
            checkpoint_due |= (
                save_interval_steps > 0 and self.global_step % save_interval_steps == 0
            )
            if checkpoint_due and not pending_batches:
                if is_main():
                    self.save_step_checkpoint(epoch)
                barrier()
                checkpoint_due = False

            loss_dict = {k: float(v.item()) for k, v in losses.items()}
            epoch_losses.append(loss_dict)

            # Update sliding windows every step (rank 0 only)
            if is_main():
                for k, v in loss_dict.items():
                    self._loss_windows[k].append(v)

                # Attribute flow loss to each sample's source dataset (diagnostic).
                # Same masked Smooth-L1 / beta as the training loss so the numbers
                # are directly comparable to the logged flow_loss, but per sample.
                ds_names = batch.get("dataset")
                if ds_names is not None and "flow_3d" in output:
                    try:
                        with torch.no_grad():
                            pred = output["flow_3d"].detach().float()
                            gt = batch["flow"].to(pred.device).float()
                            m = batch["mask"].to(pred.device).float()
                            m = m * torch.isfinite(gt).all(dim=-1, keepdim=True).float()
                            gt = torch.nan_to_num(gt)
                            diff = torch.nn.functional.smooth_l1_loss(
                                pred,
                                gt,
                                reduction="none",
                                beta=self.loss_config.flow_loss_beta,
                            )
                            per = (diff * m).flatten(1).sum(1) / (
                                m.flatten(1).sum(1) * diff.shape[-1] + 1e-6
                            )
                            for name, val in zip(ds_names, per.tolist()):
                                if math.isfinite(val):
                                    self._flow_by_dataset[name].append(val)
                    except Exception as e:
                        logger.warn(f"per-dataset flow diag failed: {e}")

            # ---- Rank 0: visualization ----
            vis_interval = getattr(self.config, "vis_interval", 0)
            if is_main() and vis_interval > 0 and batch_idx % vis_interval == 0:
                try:
                    self._visualize_step(batch, output, epoch, batch_idx)
                except Exception as e:
                    logger.warn(f"Vis failed at step {batch_idx}: {e}")

            # ---- Rank 0: logging ----
            if is_main() and batch_idx % self.config.log_interval == 0:
                # Compute rolling averages
                avg_losses = {k: sum(w) / len(w) for k, w in self._loss_windows.items()}

                # tqdm postfix: compact summary
                pbar.set_postfix(
                    total=f"{loss_dict['total_loss']:.4f}",
                    flow=f"{loss_dict['flow_loss']:.4f}",
                    pose=f"{loss_dict.get('pose_loss', 0):.4f}",
                    feat=f"{loss_dict.get('feat_loss', 0):.4f}",
                )

                # Detailed loss log
                lr = self.optimizer.param_groups[0]["lr"]
                parts = [f"[Step {self.global_step}]"]
                # Print each loss: current value / rolling avg
                loss_keys = [
                    ("total_loss", "total"),
                    ("pose_loss", "pose"),
                    ("flow_loss", "flow"),
                    ("recon_loss", "recon"),
                    ("recon_cos_sim", "rcos"),
                    ("z_recon_rel", "zrel"),
                    ("z_recon_topk_rel", "ztopk"),
                    ("z_gain_recon", "zrg"),
                    ("feat_loss", "feat"),
                    ("feat_cos_sim", "cos_sim"),
                    ("feat_cos_sim_zeroz", "cos0"),
                    ("feat_z_gain", "z_gain"),
                    ("adversarial_loss", "adv"),
                ]
                for key, label in loss_keys:
                    if key in loss_dict:
                        cur = loss_dict[key]
                        avg = avg_losses.get(key, cur)
                        parts.append(f"{label}={cur:.4f}(avg:{avg:.4f})")
                parts.append(f"lr={lr:.2e}")
                logger.info("  ".join(parts))

                # Per-dataset flow loss, worst domain first.
                if self._flow_by_dataset:
                    per_ds = {
                        d: sum(w) / len(w)
                        for d, w in self._flow_by_dataset.items()
                        if w
                    }
                    if per_ds:
                        ranked = sorted(per_ds.items(), key=lambda kv: -kv[1])
                        logger.info(
                            "  [flow/dataset] "
                            + "  ".join(
                                f"{d}={v:.4f}(n{len(self._flow_by_dataset[d])})"
                                for d, v in ranked
                            )
                        )

                log_data = {f"train/{k}": v for k, v in loss_dict.items()}
                log_data.update(
                    {f"train_avg/{k}": v for k, v in avg_losses.items()}
                )
                log_data["train/lr"] = lr
                for d, w in self._flow_by_dataset.items():
                    if w:
                        log_data[f"flow_by_dataset/{d}"] = sum(w) / len(w)
                self._log_metrics(log_data, self.log_step)

            self.log_step += 1

        return (
            {k: float(np.mean([x[k] for x in epoch_losses])) for k in epoch_losses[0]}
            if epoch_losses
            else {}
        )

    # ============================================================
    # Visualization (rank 0 only)
    # ============================================================

    def _visualize_step(self, batch, output, epoch, step):
        vis_dir = self.save_dir / "train_vis"
        vis_dir.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            frame_t = batch["images"][0, 0].detach().cpu().float().numpy()
            frame_tn = batch["images"][0, 1].detach().cpu().float().numpy()
            frame_t_u8 = _to_uint8_rgb(frame_t)
            frame_tn_u8 = _to_uint8_rgb(frame_tn)

            H, W = frame_t_u8.shape[:2]
            ps = self.patch_size
            nH = (H + (-H % ps)) // ps
            nW = (W + (-W % ps)) // ps

            # Flow 3D -> 2D projection
            flow_pred_3d = output["flow_3d"][0, 0].detach().cpu()
            flow_gt_3d = batch["flow"][0, 0].detach().cpu()
            mask_gt = batch["mask"][0, 0, ..., 0].bool().detach().cpu()
            depth1 = batch["depths"][0, 0].detach().cpu()
            K1 = batch["intrinsics"][0, 0].detach().cpu()
            fx, fy = K1[0, 0], K1[1, 1]
            z = depth1.clamp(min=1e-6)

            flow_pred_2d = torch.zeros(H, W, 2)
            flow_pred_2d[..., 0] = flow_pred_3d[..., 0] * fx / z
            flow_pred_2d[..., 1] = flow_pred_3d[..., 1] * fy / z
            flow_gt_2d = torch.zeros(H, W, 2)
            flow_gt_2d[..., 0] = flow_gt_3d[..., 0] * fx / z
            flow_gt_2d[..., 1] = flow_gt_3d[..., 1] * fy / z

            mask_np = mask_gt.numpy()
            gt_lines = _draw_pair_lines(frame_t_u8, flow_gt_2d.numpy(), mask_np)
            pred_lines = _draw_pair_lines(frame_t_u8, flow_pred_2d.numpy(), mask_np)

            # Feature PCA + cosine similarity
            blank = np.zeros_like(frame_t_u8)
            src_pca = blank.copy()
            gt_pca = blank.copy()
            pred_pca = blank.copy()
            sim_gt = blank.copy()
            sim_src = blank.copy()

            if "refined_features" in output and "target_features" in output:
                pred_feat = output["refined_features"][0].detach().cpu().float().numpy()
                gt_feat = output["target_features"][0].detach().cpu().float().numpy()

                gt_pca, basis = _features_to_pca_rgb(gt_feat, nH, nW, H, W)
                pred_pca, _ = _features_to_pca_rgb(pred_feat, nH, nW, H, W, basis)
                sim_gt = _cosine_sim_map(pred_feat, gt_feat, nH, nW, H, W)

                if "source_features" in output:
                    src_feat = (
                        output["source_features"][0].detach().cpu().float().numpy()
                    )
                    src_pca, _ = _features_to_pca_rgb(src_feat, nH, nW, H, W, basis)
                    sim_src = _cosine_sim_map(pred_feat, src_feat, nH, nW, H, W)

            # 3x3 grid
            row1 = np.concatenate([frame_t_u8, frame_tn_u8, pred_pca], axis=1)
            row2 = np.concatenate([src_pca, gt_pca, sim_gt], axis=1)
            row3 = np.concatenate([sim_src, gt_lines, pred_lines], axis=1)
            grid = np.concatenate([row1, row2, row3], axis=0)

            # Labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            labels = [
                (10, 25, "frame_t"),
                (W + 10, 25, "frame_t+n (GT)"),
                (2 * W + 10, 25, "Recon (PCA)"),
                (10, H + 25, "src_t (PCA)"),
                (W + 10, H + 25, "GT t+n (PCA)"),
                (2 * W + 10, H + 25, "sim(recon,GT)"),
                (10, 2 * H + 25, "sim(recon,src)"),
                (W + 10, 2 * H + 25, "GT flow"),
                (2 * W + 10, 2 * H + 25, "Pred flow"),
            ]
            for x, y, text in labels:
                cv2.putText(
                    grid, text, (x + 1, y + 1), font, 0.5, (0, 0, 0), 2, cv2.LINE_AA
                )
                cv2.putText(
                    grid, text, (x, y), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA
                )

            # Mean cosine similarity values
            if "refined_features" in output and "target_features" in output:
                pn = pred_feat / (
                    np.linalg.norm(pred_feat, axis=-1, keepdims=True) + 1e-6
                )
                gn = gt_feat / (np.linalg.norm(gt_feat, axis=-1, keepdims=True) + 1e-6)
                cv2.putText(
                    grid,
                    f"{(pn * gn).sum(-1).mean():.3f}",
                    (2 * W + 10, 2 * H - 10),
                    font,
                    0.6,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                if "source_features" in output:
                    sn = src_feat / (
                        np.linalg.norm(src_feat, axis=-1, keepdims=True) + 1e-6
                    )
                    cv2.putText(
                        grid,
                        f"{(pn * sn).sum(-1).mean():.3f}",
                        (10, 3 * H - 10),
                        font,
                        0.6,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

            fname = f"e{epoch + 1:02d}_s{step:05d}.jpg"
            cv2.imwrite(
                str(vis_dir / fname),
                cv2.cvtColor(grid, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            )

    # ============================================================
    # Evaluation (rank 0 only)
    # ============================================================

    @torch.no_grad()
    def evaluate(self, epoch):
        if not is_main():
            return {}

        # Do not call DDP.forward here: other ranks are waiting, so buffer
        # broadcasts in the wrapper would deadlock.
        self.model_raw.eval()
        if not self.dataset.scene_list:
            return {}

        scene_id = getattr(self.config, "val_scene_id", None)
        if not scene_id or scene_id not in self.dataset.scene_list:
            fallback = self.dataset.scene_list[0]
            if scene_id and scene_id != fallback:
                logger.warning(
                    f"val_scene_id '{scene_id}' not found in dataset, "
                    f"falling back to '{fallback}'"
                )
            scene_id = fallback

        scene_samples = self.dataset.get_scene(scene_id)
        logger.info(f"Evaluating scene '{scene_id}' ({len(scene_samples)} pairs)")

        vis_dir = (
            self.save_dir / f"epoch_{epoch + 1:03d}" / "eval" / f"scene_{scene_id}"
        )
        vis_dir.mkdir(parents=True, exist_ok=True)

        losses_list = []
        cosine_list = []

        for sample in tqdm(scene_samples, desc=f"Eval {scene_id}"):
            batch = {
                k: (
                    v.unsqueeze(0).to(self.device)
                    if isinstance(v, torch.Tensor)
                    else [v]
                )
                for k, v in sample.items()
            }

            with torch.amp.autocast(
                self.device.type, dtype=self.amp_dtype, enabled=self.use_amp
            ):
                output = self.model_raw(
                    images=batch["images"],
                    extrinsics=batch["extrinsics"],
                    intrinsics=batch["intrinsics"],
                    depths=batch["depths"],
                )
                target = {
                    "flows": batch["flow"],
                    "mask": batch["mask"],
                    "extrinsics": batch["extrinsics"],
                }
                compute_loss_fn = compute_lam_v5_losses
                losses = compute_loss_fn(
                    output=output,
                    target=target,
                    loss_config=self.loss_config,
                    global_step=self.global_step,
                    flow_warmup_steps=self.flow_warmup_steps,
                )

            losses_list.append({k: float(v.item()) for k, v in losses.items()})

            if "refined_features" in output and "target_features" in output:
                pred = torch.nn.functional.normalize(
                    output["refined_features"][0], dim=-1
                )
                gt = torch.nn.functional.normalize(output["target_features"][0], dim=-1)
                cosine_list.append((pred * gt).sum(-1).mean().item())

        if not losses_list:
            self.model_raw.train()
            return {}
        avg = {k: float(np.mean([x[k] for x in losses_list])) for k in losses_list[0]}
        avg["feat_cosine_sim"] = float(np.mean(cosine_list)) if cosine_list else 0.0

        with open(vis_dir / "metrics.json", "w") as f:
            json.dump({"scene_id": scene_id, "avg_losses": avg}, f, indent=2)

        logger.info("Eval: " + ", ".join(f"{k}={v:.4f}" for k, v in avg.items()))

        self._log_metrics({f"eval/{k}": v for k, v in avg.items()}, epoch)

        self.model.train()
        return avg

    # ============================================================
    # Checkpoint
    # ============================================================

    def _resume_data_layout(self):
        """Snapshot the sampling layout once; never stat individual tensor files."""
        if hasattr(self, "_data_layout"):
            return self._data_layout
        if self.dataset is None or self.sampler is None or self.train_loader is None:
            raise RuntimeError("Set up data before saving or restoring optimizer state")
        config = self.config
        # Match LAMEpicDataset's local manifest discovery. Its cache identity is
        # currently local to __init__, so hash the source JSON files here once.
        root = config.manifest_path or config.safetensors_root
        manifests = []
        if root:
            root = Path(root)
            if root.is_file():
                manifests = [root]
            elif root.is_dir():
                manifests = sorted(
                    set(root.glob("manifest_*.json"))
                    | set(root.glob("*/manifest_*.json"))
                    | set(root.glob("manifest.json"))
                )
            if not manifests:
                raise ValueError(f"Cannot fingerprint resume manifests at {root}")
        if config.data_root:
            meta_path = Path(config.data_root)
            if meta_path not in manifests:
                manifests.append(meta_path)
        if not manifests:
            raise ValueError(
                "Cannot verify resume data layout without a local manifest"
            )
        fingerprints = []
        for path in manifests:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            fingerprints.append((str(path.resolve()), digest.hexdigest()))
        self._data_layout = {
            "version": 1,
            "world_size": self.world_size,
            "batch_size": config.batch_size,
            "accum_steps": self.accum_steps,
            "min_frame_stride": config.min_frame_stride,
            "max_frame_stride": config.max_frame_stride,
            "max_sample_stride": (
                config.max_sample_stride
                if config.max_sample_stride is not None
                else config.max_frame_stride
            ),
            "max_scenes": config.max_scenes,
            "target_hw": (
                tuple(config.target_hw) if config.target_hw is not None else None
            ),
            "dataset_sampling_temperature": config.dataset_sampling_temperature,
            "dataset_sampling_weights": dict(config.dataset_sampling_weights or {}),
            "dataset_role_sampling_weights": dict(
                config.dataset_role_sampling_weights or {}
            ),
            "sampler_type": type(self.sampler).__name__,
            "sampler_seed": self.sampler.seed,
            "sampler_shuffle": self.sampler.shuffle,
            "sampler_drop_last": self.sampler.drop_last,
            "batches_per_epoch": len(self.train_loader),
            "depth_root": (
                str(Path(config.depth_root).resolve()) if config.depth_root else None
            ),
            "flow_root": (
                str(Path(config.flow_root).resolve()) if config.flow_root else None
            ),
            "manifest_fingerprints": fingerprints,
        }
        return self._data_layout

    def _validate_resume_data_layout(self, ckpt):
        saved = ckpt.get("resume_data_layout")
        if saved is None:
            if ckpt.get("batches_in_epoch", 0):
                raise ValueError(
                    "Checkpoint has a partial-epoch position but no resume data-layout "
                    "metadata; use reset_optimizer=True for a weights-only restart"
                )
            logger.warning(
                "Checkpoint lacks resume data-layout metadata; sample-exact resume "
                "cannot be verified. Continuing at the epoch boundary."
            )
            return
        current = self._resume_data_layout()
        changed = sorted(
            key
            for key in saved.keys() | current.keys()
            if saved.get(key) != current.get(key)
        )
        if changed:
            raise ValueError(
                "Resume data layout changed: "
                + ", ".join(changed)
                + "; use reset_optimizer=True for a weights-only restart"
            )
        position = ckpt.get("batches_in_epoch", 0)
        if not 0 <= position <= len(self.train_loader):
            raise ValueError(
                "Checkpoint sampler position is outside the current epoch; "
                "use reset_optimizer=True for a weights-only restart"
            )

    def _write_checkpoint_sidecars(self, ckpt_dir, ckpt, label):
        """Save portable resolved configuration and metrics only."""
        ckpt_dir = Path(ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        with (ckpt_dir / "config_resolved.json").open("w") as stream:
            json.dump(self.config.__dict__, stream, indent=2)
        with (ckpt_dir / "metrics.json").open("w") as stream:
            json.dump(
                {
                    "label": label,
                    "global_step": self.global_step,
                    "optimizer_step": self.optimizer_step,
                    "metrics": ckpt.get("metrics", {}),
                },
                stream,
                indent=2,
            )

    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Save checkpoint (rank 0 only). Compatible with single-GPU loaders."""
        if not is_main():
            return

        ckpt = {
            "epoch": epoch
            + 1,  # next epoch to train (completed epochs are skipped on resume)
            "model_state_dict": self.model_raw.state_dict(),  # unwrap DDP
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "metrics": metrics,
            "config": self.config.__dict__,
            "best_loss": self.best_loss,
            "log_step": self.log_step,
            "global_step": self.global_step,
            "model_version": "rynnlam",
            "optimizer_step": self.optimizer_step,
            "accum_steps": self.accum_steps,
            "batches_in_epoch": self.batches_in_epoch,
            "resume_data_layout": self._resume_data_layout(),
        }

        epoch_dir = self.save_dir / f"epoch_{epoch + 1:03d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, epoch_dir / "checkpoint.pt")
        torch.save(ckpt, self.save_dir / "latest.pt")

        # Sidecar files — same content next to epoch checkpoint AND latest.pt
        self._write_checkpoint_sidecars(epoch_dir, ckpt, label=f"epoch_{epoch + 1:03d}")
        self._write_checkpoint_sidecars(self.save_dir, ckpt, label="latest")

        if is_best:
            torch.save(ckpt, self.save_dir / "best.pt")
            self._write_checkpoint_sidecars(self.save_dir, ckpt, label="best")

        logger.info(
            f"Saved checkpoint: epoch {epoch + 1}" + (" [BEST]" if is_best else "")
        )

    def save_step_checkpoint(self, epoch):
        """Save a mid-epoch checkpoint at step boundaries (rank 0 only)."""
        # Never persist corrupted weights: a NaN model overwriting latest.pt /
        # a milestone step ckpt destroys the only good resume point.
        for name, p in self.model_raw.named_parameters():
            if not torch.isfinite(p).all():
                logger.error(
                    f"REFUSING to save step checkpoint at step {self.global_step}: "
                    f"non-finite weights detected (e.g. {name}). Model has diverged."
                )
                return
        ckpt = {
            "epoch": epoch,  # current epoch (0-indexed), resume continues this epoch
            "model_state_dict": self.model_raw.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "metrics": {},
            "config": self.config.__dict__,
            "best_loss": self.best_loss,
            "log_step": self.log_step,
            "global_step": self.global_step,
            "model_version": "rynnlam",
            "optimizer_step": self.optimizer_step,
            "accum_steps": self.accum_steps,
            "batches_in_epoch": self.batches_in_epoch,
            "resume_data_layout": self._resume_data_layout(),
        }
        step_dir = self.save_dir / "steps"
        step_dir.mkdir(parents=True, exist_ok=True)
        step_ckpt_path = step_dir / f"step_{self.global_step:06d}.pt"
        torch.save(ckpt, step_ckpt_path)
        torch.save(ckpt, self.save_dir / "latest.pt")

        # Sidecar files — also next to the step checkpoint
        step_sidecar_dir = step_dir / f"step_{self.global_step:06d}"
        self._write_checkpoint_sidecars(
            step_sidecar_dir, ckpt, label=f"step_{self.global_step:06d}"
        )
        self._write_checkpoint_sidecars(self.save_dir, ckpt, label="latest")

        logger.info(
            f"Saved step checkpoint: step {self.global_step} (epoch {epoch + 1})"
        )

    def load_checkpoint(self, path):
        """Load checkpoint on all ranks."""
        ckpt = torch.load(path, map_location="cpu", weights_only=True)

        reset_opt = getattr(self.config, "reset_optimizer", False)
        if reset_opt:
            self.model_raw.load_state_dict(ckpt["model_state_dict"], strict=True)
            if is_main():
                logger.info(
                    f"Loaded model weights strictly from {path}; reset_optimizer=True: "
                    "fresh optimizer/scheduler/scaler, epoch=0, global_step=0, optimizer_step=0."
                )
            return

        self._validate_resume_data_layout(ckpt)
        self.model_raw.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])

        self.current_epoch = ckpt["epoch"]
        self.best_loss = ckpt.get("best_loss", float("inf"))
        self.log_step = ckpt.get("log_step", 0)
        self.global_step = ckpt.get("global_step", 0)
        self.optimizer_step = ckpt.get(
            "optimizer_step",
            (
                max(0, self.scheduler.last_epoch)
                if self.scheduler_per_step
                else self.global_step // self.accum_steps
            ),
        )
        self.batches_in_epoch = ckpt.get("batches_in_epoch", 0)
        if is_main():
            logger.info(
                f"Resumed model/optimizer/scheduler/scaler from {path}, "
                f"epoch={self.current_epoch}, batch={self.global_step}, optimizer_step={self.optimizer_step}. "
                "Random augmentations are not bitwise replayed."
            )
            if "resume_data_layout" in ckpt and "batches_in_epoch" in ckpt:
                logger.info("Sampler position restored after data-layout validation.")
            elif "batches_in_epoch" not in ckpt:
                logger.warning(
                    "Legacy checkpoint lacks sampler position: replaying its current epoch"
                )

    # ============================================================
    # Main loop
    # ============================================================

    def train(self):
        self.setup_data()

        if self.config.resume:
            self.load_checkpoint(self.config.resume)

        if not len(self.train_loader):
            raise ValueError("Training loader is empty")
        if self.budget_complete():
            logger.info(
                "Checkpoint already reached the configured budget; no optimizer or scheduler step"
            )
            return
        if self.scheduler_per_step:
            steps_per_epoch = math.ceil(len(self.train_loader) / self.accum_steps)
            remaining_epochs = max(0, self.config.num_epochs - self.current_epoch)
            remaining_batches = max(0, len(self.train_loader) - self.batches_in_epoch)
            self._total_steps = (
                self.optimizer_step
                + math.ceil(remaining_batches / self.accum_steps)
                + max(0, remaining_epochs - 1) * steps_per_epoch
            )
            if self.max_optimizer_steps:
                self._total_steps = min(self._total_steps, self.max_optimizer_steps)
            elif self.max_steps:
                self._total_steps = min(
                    self._total_steps,
                    self.optimizer_step
                    + math.ceil(
                        max(0, self.max_steps - self.global_step) / self.accum_steps
                    ),
                )
            logger.info(
                f"LR schedule: {self._total_steps} optimizer updates, "
                f"warmup={self.lr_warmup_steps}; global_step counts per-rank batches"
            )

        for epoch in range(self.current_epoch, self.config.num_epochs):
            if self.budget_complete():
                break
            self.current_epoch = epoch

            if is_main():
                logger.info(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")

            before_updates = self.optimizer_step
            avg = self.train_epoch(epoch)
            epoch_complete = self.batches_in_epoch >= len(self.train_loader)
            if not self.scheduler_per_step and self.optimizer_step > before_updates:
                self.scheduler.step()

            if is_main():
                summary_parts = [f"Epoch {epoch + 1} summary:"]
                for k, v in avg.items():
                    summary_parts.append(f"  {k}={v:.6f}")
                logger.info("\n".join(summary_parts))

            do_eval = (
                self.config.eval_interval > 0
                and (epoch + 1) % self.config.eval_interval == 0
            )
            do_save = (
                self.config.save_interval > 0
                and (epoch + 1) % self.config.save_interval == 0
            )

            is_best = False
            eval_metrics = {}

            if do_eval:
                eval_metrics = self.evaluate(epoch)
                if is_main():
                    cur = eval_metrics.get("total_loss", float("inf"))
                    if cur < self.best_loss:
                        self.best_loss = cur
                        is_best = True

            if epoch_complete:
                self.batches_in_epoch = 0
            if do_save:
                if epoch_complete:
                    self.save_checkpoint(epoch, eval_metrics, is_best=is_best)
                elif is_main():
                    self.save_step_checkpoint(epoch)

            # Sync all ranks before next epoch
            barrier()


# ==============================================================
# Main
# ==============================================================


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to YAML configuration")
    parser.add_argument("--output-dir", help="Run root directory (default: ./runs)")
    parser.add_argument(
        "--resume", help="Trusted checkpoint to resume; required for stage 2"
    )
    parser.add_argument("--encoder-checkpoint", help="Local DA3 checkpoint path")
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Optimizer updates when target_effective_batch > 0; otherwise per-rank batches",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument(
        "--reset-optimizer",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Weights-only initialization, resetting optimizer/scheduler and all counters",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = RynnLAMConfig.from_yaml(args.config)
    for arg, field in (
        ("output_dir", "output_dir"),
        ("resume", "resume"),
        ("encoder_checkpoint", "encoder_checkpoint_path"),
        ("max_steps", "max_steps"),
        ("device", "device"),
        ("reset_optimizer", "reset_optimizer"),
    ):
        value = getattr(args, arg)
        if value is not None:
            setattr(config, field, value)
    config.validate_training()
    local_rank = setup_ddp(config.device)
    try:
        trainer = DDPTrainer(config, local_rank)
        trainer.train()
    finally:
        cleanup_ddp()


if __name__ == "__main__":
    main()
