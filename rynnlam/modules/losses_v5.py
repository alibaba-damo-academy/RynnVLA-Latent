"""Losses and diagnostics for the retained RynnLAM training recipe."""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict
from .losses import compute_relative_pose_target


@dataclass
class LAMv5LossConfig:
    lambda_pose: float = 1.0
    lambda_flow: float = 2.0
    lambda_feat: float = 1.0
    lambda_adversarial: float = 0.5
    flow_loss_beta: float = 0.05
    use_contrastive_feat_loss: bool = False
    contrast_temperature: float = 0.1
    lambda_recon: float = 0.0
    use_z_residual_loss: bool = False
    z_recon_warmup_steps: int = 0


def compute_lam_v5_losses(
    output: Dict,
    target: Dict,
    loss_config: LAMv5LossConfig,
    global_step: int = 0,
    flow_warmup_steps: int = 0,
) -> Dict[str, torch.Tensor]:
    losses = {}
    latent_action = output["latent_action"]
    total_loss = torch.tensor(0.0, device=latent_action.device)
    pose_gt = None
    if "extrinsics" in target:
        pose_gt = compute_relative_pose_target(target["extrinsics"])
        if pose_gt.shape[1] == 1:
            pose_gt = pose_gt.squeeze(1)
    if "pose_pred" in output and pose_gt is not None:
        pose_pred = output["pose_pred"]
        pose_loss = F.mse_loss(pose_pred, pose_gt.detach())
        losses["pose_loss"] = pose_loss
        total_loss = total_loss + loss_config.lambda_pose * pose_loss
    flow_3d_gt = target["flows"]
    flow_3d_pred = output["flow_3d"].clamp(-2.0, 2.0)
    flow_3d_gt = flow_3d_gt.clamp(-2.0, 2.0)
    mask = target.get("mask", None)
    gt_finite = torch.isfinite(flow_3d_gt).all(dim=-1, keepdim=True).float()
    if mask is not None:
        mask = mask * gt_finite
    else:
        mask = gt_finite
    beta = loss_config.flow_loss_beta
    flow_diff = F.smooth_l1_loss(flow_3d_pred, flow_3d_gt, reduction="none", beta=beta)
    gt_motion = flow_3d_gt.norm(dim=-1, keepdim=True)
    motion_w = 1.0 + 20.0 * gt_motion.clamp(max=0.2) / 0.2
    weight = motion_w
    if mask is not None:
        weight = mask * weight
    flow_loss = (flow_diff * weight).sum() / (
        weight.sum() * flow_diff.shape[-1] + 1e-06
    )
    flow_loss = flow_loss.clamp(max=100.0)
    losses["flow_loss"] = flow_loss
    total_loss = total_loss + loss_config.lambda_flow * flow_loss
    if "refined_features" in output and "target_features" in output:
        pred_feat = output["refined_features"]
        gt_feat = output["target_features"]
        pred_norm = F.normalize(pred_feat, dim=-1, eps=1e-06)
        gt_norm = F.normalize(gt_feat, dim=-1, eps=1e-06)
        B, N, _ = pred_feat.shape
        motion_map = gt_motion.squeeze(-1).squeeze(1)
        H_full, W_full = motion_map.shape[-2:]
        ps_h = int(round((H_full * W_full / max(N, 1)) ** 0.5))
        nH = max(1, H_full // ps_h)
        nW = max(1, N // nH)
        motion_patch = F.adaptive_avg_pool2d(motion_map.unsqueeze(1), (nH, nW))
        motion_patch = motion_patch.reshape(B, nH * nW)
        if motion_patch.shape[1] != N:
            patch_w = torch.ones(B, N, device=pred_feat.device, dtype=pred_feat.dtype)
        else:
            patch_w = 1.0 + 5.0 * motion_patch.clamp(max=0.5) / 0.5
        if pred_feat.shape[-1] != gt_feat.shape[-1]:
            cos_sim = torch.tensor(0.0, device=pred_feat.device)
            feat_loss = torch.tensor(1.0, device=pred_feat.device)
        elif loss_config.use_contrastive_feat_loss and "source_features" in output:
            src_feat = output["source_features"]
            src_norm = F.normalize(src_feat, dim=-1, eps=1e-06)
            cos_sim_pos = (pred_norm * gt_norm).sum(dim=-1)
            cos_sim_neg = (pred_norm * src_norm).sum(dim=-1)
            tau = loss_config.contrast_temperature
            logits = torch.stack([cos_sim_pos / tau, cos_sim_neg / tau], dim=-1)
            labels = torch.zeros(B * N, dtype=torch.long, device=logits.device)
            ce = F.cross_entropy(logits.reshape(-1, 2), labels, reduction="none")
            ce = ce.reshape(B, N)
            feat_loss = (ce * patch_w).sum() / (patch_w.sum() + 1e-06)
            cos_sim = cos_sim_pos
        else:
            cos_sim = (pred_norm * gt_norm).sum(dim=-1)
            feat_loss = ((1.0 - cos_sim) * patch_w).sum() / (patch_w.sum() + 1e-06)
        losses["feat_loss"] = feat_loss
        losses["feat_cos_sim"] = cos_sim.mean().detach()
        pred0 = output.get("refined_features_zeroz", None)
        if pred0 is not None and pred0.shape[-1] == gt_feat.shape[-1]:
            pred0_norm = F.normalize(pred0, dim=-1, eps=1e-06)
            cos0 = (pred0_norm * gt_norm).sum(dim=-1)
            losses["feat_cos_sim_zeroz"] = cos0.mean().detach()
            losses["feat_z_gain"] = (cos_sim.mean() - cos0.mean()).detach()
        feat_weight = loss_config.lambda_feat
        if flow_warmup_steps > 0 and global_step < flow_warmup_steps:
            feat_weight = feat_weight * (global_step / flow_warmup_steps)
        total_loss = total_loss + feat_weight * feat_loss
    if (
        loss_config.lambda_recon > 0
        and output.get("recon_features") is not None
        and ("target_features" in output)
    ):
        rec_pred = output["recon_features"]
        rec_gt = output["target_features"]
        rec_pred_norm = F.normalize(rec_pred, dim=-1, eps=1e-06)
        rec_gt_norm = F.normalize(rec_gt, dim=-1, eps=1e-06)
        rec_cos = (rec_pred_norm * rec_gt_norm).sum(dim=-1)
        B_r, N_r, _ = rec_pred.shape
        motion_map_r = gt_motion.squeeze(-1).squeeze(1)
        H_r, W_r = motion_map_r.shape[-2:]
        ps_h_r = int(round((H_r * W_r / max(N_r, 1)) ** 0.5))
        nH_r = max(1, H_r // ps_h_r)
        nW_r = max(1, N_r // nH_r)
        motion_patch_r = F.adaptive_avg_pool2d(motion_map_r.unsqueeze(1), (nH_r, nW_r))
        motion_patch_r = motion_patch_r.reshape(B_r, nH_r * nW_r)
        if motion_patch_r.shape[1] != N_r:
            patch_w_r = torch.ones(
                B_r, N_r, device=rec_pred.device, dtype=rec_pred.dtype
            )
        else:
            patch_w_r = 1.0 + 5.0 * motion_patch_r.clamp(max=0.5) / 0.5
        if loss_config.use_z_residual_loss and "source_features" in output:
            src_r = output["source_features"]
            res_pred = rec_pred - src_r
            res_tgt = rec_gt - src_r
            num = ((res_pred - res_tgt) ** 2).sum(dim=-1)
            den = (res_tgt**2).sum(dim=-1) + 1e-06
            rel = num / den
            recon_loss = (rel * patch_w_r).sum() / (patch_w_r.sum() + 1e-06)
            losses["z_recon_rel"] = rel.mean().detach()
            with torch.no_grad():
                k = max(1, N_r // 10)
                mot = res_tgt.norm(dim=-1)
                idx = mot.topk(k, dim=1).indices
                losses["z_recon_topk_rel"] = rel.gather(1, idx).mean().detach()
                rec0 = output.get("recon_features_zeroz", None)
                if rec0 is not None:
                    rel0 = ((rec0 - src_r - res_tgt) ** 2).sum(dim=-1) / den
                    loss0 = (rel0 * patch_w_r).sum() / (patch_w_r.sum() + 1e-06)
                    losses["z_gain_recon"] = (loss0 - recon_loss).detach()
        else:
            recon_loss = ((1.0 - rec_cos) * patch_w_r).sum() / (patch_w_r.sum() + 1e-06)
        losses["recon_loss"] = recon_loss
        losses["recon_cos_sim"] = rec_cos.mean().detach()
        _lam_recon = loss_config.lambda_recon
        if loss_config.z_recon_warmup_steps > 0:
            _lam_recon = _lam_recon * min(
                1.0, global_step / loss_config.z_recon_warmup_steps
            )
        total_loss = total_loss + _lam_recon * recon_loss
    if "pose_from_action" in output and pose_gt is not None:
        pose_from_action = output["pose_from_action"]
        adv_loss = F.mse_loss(pose_from_action, pose_gt.detach())
        losses["adversarial_loss"] = adv_loss
        total_loss = total_loss + loss_config.lambda_adversarial * adv_loss
    losses["total_loss"] = total_loss
    return losses
