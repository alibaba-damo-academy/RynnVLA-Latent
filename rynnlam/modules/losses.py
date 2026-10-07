"""Loss functions for LAM training."""

import torch
import torch.nn.functional as F
from typing import Dict


def _rotation_matrix_to_rotvec(R: torch.Tensor) -> torch.Tensor:
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    cos_angle = torch.clamp((trace - 1) / 2, -1.0 + 1e-7, 1.0 - 1e-7)
    angle = torch.acos(cos_angle)
    rx = R[:, 2, 1] - R[:, 1, 2]
    ry = R[:, 0, 2] - R[:, 2, 0]
    rz = R[:, 1, 0] - R[:, 0, 1]
    sin_angle = torch.sin(angle).clamp(min=1e-8)
    scale = angle / (2 * sin_angle)
    small = angle.abs() < 1e-6
    scale = torch.where(small, torch.full_like(scale, 0.5), scale)
    return torch.stack([rx * scale, ry * scale, rz * scale], dim=-1)


def compute_relative_pose_target(extrinsics: torch.Tensor) -> torch.Tensor:
    """Compute relative pose (6DoF) between consecutive frames.

    Args:
        extrinsics: [B, T, 4, 4] world-to-camera transforms

    Returns:
        [B, T-1, 6] = [rotvec(3), translation(3)]
    """
    E1 = extrinsics[:, :-1]
    E2 = extrinsics[:, 1:]
    B, Tm1 = E1.shape[:2]
    T_rel = E2 @ torch.linalg.inv(E1)
    R_rel = T_rel[:, :, :3, :3]
    t_rel = T_rel[:, :, :3, 3]
    R_flat = R_rel.reshape(B * Tm1, 3, 3)
    rotvec = _rotation_matrix_to_rotvec(R_flat).reshape(B, Tm1, 3)
    return torch.cat([rotvec, t_rel], dim=-1)
