"""RynnLAM latent-action, flow, and K-token reconstruction modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal
from .lam import DA3ViTLargeEncoder
from .blocks import patchify, PositionalEncoding, SelfAttention


class StaticFlowFromPose(nn.Module):

    def __init__(self):
        super().__init__()

    @staticmethod
    def rotvec_to_matrix(rotvec: torch.Tensor) -> torch.Tensor:
        theta = rotvec.norm(dim=-1, keepdim=True).clamp(min=1e-08)
        axis = rotvec / theta
        cos_t = torch.cos(theta).unsqueeze(-1)
        sin_t = torch.sin(theta).unsqueeze(-1)
        kx, ky, kz = (axis[:, 0], axis[:, 1], axis[:, 2])
        zero = torch.zeros_like(kx)
        K = torch.stack([zero, -kz, ky, kz, zero, -kx, -ky, kx, zero], dim=-1).reshape(
            -1, 3, 3
        )
        I = torch.eye(3, device=rotvec.device, dtype=rotvec.dtype).unsqueeze(0)
        return I + sin_t * K + (1 - cos_t) * (K @ K)

    def forward(self, pose_pred, depths, intrinsics):
        B = pose_pred.shape[0]
        depth_t = depths[:, 0]
        K = intrinsics[:, 0]
        H, W = (depth_t.shape[1], depth_t.shape[2])
        ys, xs = torch.meshgrid(
            torch.arange(H, device=depth_t.device, dtype=depth_t.dtype),
            torch.arange(W, device=depth_t.device, dtype=depth_t.dtype),
            indexing="ij",
        )
        fx = K[:, 0, 0].view(B, 1, 1)
        fy = K[:, 1, 1].view(B, 1, 1)
        cx = K[:, 0, 2].view(B, 1, 1)
        cy = K[:, 1, 2].view(B, 1, 1)
        z = depth_t
        x = (xs.unsqueeze(0) - cx) * z / (fx + 1e-08)
        y = (ys.unsqueeze(0) - cy) * z / (fy + 1e-08)
        pts = torch.stack([x, y, z], dim=-1)
        rotvec = pose_pred[:, :3]
        t = pose_pred[:, 3:]
        R = self.rotvec_to_matrix(rotvec)
        pts_flat = pts.reshape(B, -1, 3)
        pts_transformed = torch.bmm(pts_flat, R.transpose(1, 2)) + t.unsqueeze(1)
        static_flow = (pts_transformed - pts_flat).reshape(B, H, W, 3)
        return static_flow.unsqueeze(1)


class CrossAttentionActionEncoderV5(nn.Module):

    def __init__(
        self,
        in_dim: int,
        model_dim: int,
        latent_action_dim: int,
        num_heads: int,
        camera_pose_latent_dim: int = 32,
        num_blocks: int = 4,
        dropout: float = 0.0,
        motion_hint_dim: int = 16,
    ) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.latent_action_dim = latent_action_dim
        self.camera_pose_latent_dim = camera_pose_latent_dim
        self.motion_hint_dim = motion_hint_dim
        self.proj_f = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, model_dim))
        self.pos_enc = PositionalEncoding(model_dim)
        self.diff_token = nn.Parameter(torch.empty(1, 1, model_dim))
        nn.init.uniform_(self.diff_token, a=-1, b=1)
        self.camera_pose_token = nn.Parameter(torch.empty(1, 1, model_dim))
        nn.init.uniform_(self.camera_pose_token, a=-1, b=1)
        self.frame_type_emb = nn.Embedding(2, model_dim)
        self.diff_gate = nn.Sequential(nn.Linear(model_dim, model_dim), nn.Sigmoid())
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(
                nn.ModuleDict(
                    {
                        "self_norm": nn.LayerNorm(model_dim),
                        "self_attn": SelfAttention(model_dim, num_heads, dropout),
                        "ffn_norm": nn.LayerNorm(model_dim),
                        "ffn": nn.Sequential(
                            nn.Linear(model_dim, model_dim * 4),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(model_dim * 4, model_dim),
                        ),
                    }
                )
            )
        self.out_norm = nn.LayerNorm(model_dim)
        self.out_proj = nn.Linear(model_dim, latent_action_dim)
        self.cam_pose_norm = nn.LayerNorm(model_dim)
        self.cam_pose_proj = nn.Linear(model_dim, camera_pose_latent_dim)
        self.motion_hint_proj = nn.Sequential(
            nn.LayerNorm(model_dim), nn.Linear(model_dim, motion_hint_dim)
        )

    def forward(self, f1, f2):
        B, N, _ = f1.shape
        f1_proj = self.proj_f(f1)
        f2_proj = self.proj_f(f2)
        f_diff = f2_proj - f1_proj
        gate = self.diff_gate(f_diff)
        f1_gated = f1_proj + gate * f_diff
        pos = self.pos_enc.pos_enc[:N].to(f1_proj.device)
        f1_pos = f1_gated + pos.unsqueeze(0)
        f2_pos = f2_proj + pos.unsqueeze(0)
        frame_ids = torch.tensor([0, 1], device=f1.device)
        f1_type = self.frame_type_emb(frame_ids[0]).unsqueeze(0).unsqueeze(0)
        f2_type = self.frame_type_emb(frame_ids[1]).unsqueeze(0).unsqueeze(0)
        f1_pos = f1_pos + f1_type.expand(B, N, -1)
        f2_pos = f2_pos + f2_type.expand(B, N, -1)
        diff_tok = self.diff_token.expand(B, -1, -1)
        cam_tok = self.camera_pose_token.expand(B, -1, -1)
        x = torch.cat([diff_tok, cam_tok, f1_pos, f2_pos], dim=1)
        for block in self.blocks:
            x_norm = block["self_norm"](x)
            x = x + block["self_attn"](x_norm)
            x_norm = block["ffn_norm"](x)
            x = x + block["ffn"](x_norm)
        diff_out = x[:, 0, :]
        diff_out = self.out_norm(diff_out)
        latent_action = self.out_proj(diff_out)
        cam_out = x[:, 1, :]
        cam_out = self.cam_pose_norm(cam_out)
        camera_pose_latent = self.cam_pose_proj(cam_out)
        f1_attended = x[:, 2 : N + 2, :]
        motion_hints = self.motion_hint_proj(f1_attended)
        return (latent_action, camera_pose_latent, motion_hints)


class FeatureWarpV5(nn.Module):

    def __init__(self, num_sample_points: int = 4):
        super().__init__()
        self.num_sample_points = num_sample_points

    def forward(
        self,
        features_t: torch.Tensor,
        flow_3d: torch.Tensor,
        depths: torch.Tensor,
        intrinsics: torch.Tensor,
        patch_size: int,
        H: int,
        W: int,
    ) -> torch.Tensor:
        B = features_t.shape[0]
        D = features_t.shape[-1]
        device = features_t.device
        dtype = features_t.dtype
        h_pad = -H % patch_size
        w_pad = -W % patch_size
        H_padded = H + h_pad
        W_padded = W + w_pad
        nH = H_padded // patch_size
        nW = W_padded // patch_size
        N = nH * nW
        depth1 = depths[:, 0]
        K1 = intrinsics[:, 0]
        K2 = intrinsics[:, 1]
        half = patch_size / 4.0
        offsets = torch.tensor(
            [[0, 0], [-half, -half], [half, -half], [-half, half]],
            device=device,
            dtype=dtype,
        )[: self.num_sample_points]
        cy_patches = (
            torch.arange(nH, device=device, dtype=dtype) * patch_size + patch_size / 2.0
        )
        cx_patches = (
            torch.arange(nW, device=device, dtype=dtype) * patch_size + patch_size / 2.0
        )
        cy_grid, cx_grid = torch.meshgrid(cy_patches, cx_patches, indexing="ij")
        cx_center = cx_grid.reshape(-1)
        cy_center = cy_grid.reshape(-1)
        cx_multi = cx_center.unsqueeze(1) + offsets[:, 0].unsqueeze(0)
        cy_multi = cy_center.unsqueeze(1) + offsets[:, 1].unsqueeze(0)
        cx_multi = cx_multi.clamp(0, W - 1)
        cy_multi = cy_multi.clamp(0, H - 1)
        cx_idx = cx_multi.long()
        cy_idx = cy_multi.long()
        depth_multi = depth1[:, cy_idx, cx_idx]
        fx1 = K1[:, 0, 0].unsqueeze(1).unsqueeze(2)
        fy1 = K1[:, 1, 1].unsqueeze(1).unsqueeze(2)
        cx1 = K1[:, 0, 2].unsqueeze(1).unsqueeze(2)
        cy1 = K1[:, 1, 2].unsqueeze(1).unsqueeze(2)
        z = depth_multi
        x = (cx_multi.unsqueeze(0) - cx1) * z / (fx1 + 1e-08)
        y = (cy_multi.unsqueeze(0) - cy1) * z / (fy1 + 1e-08)
        pts3d = torch.stack([x, y, z], dim=-1)
        flow = flow_3d[:, 0]
        flow_perm = flow.permute(0, 3, 1, 2)
        cx_sample_norm = 2.0 * cx_multi / max(W - 1, 1) - 1.0
        cy_sample_norm = 2.0 * cy_multi / max(H - 1, 1) - 1.0
        flow_grid = torch.stack([cx_sample_norm, cy_sample_norm], dim=-1)
        flow_grid = flow_grid.unsqueeze(0).expand(B, -1, -1, -1)
        flow_sampled = F.grid_sample(
            flow_perm,
            flow_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        flow_multi = flow_sampled.permute(0, 2, 3, 1)
        pts3d_displaced = pts3d + flow_multi
        fx2 = K2[:, 0, 0].unsqueeze(1).unsqueeze(2)
        fy2 = K2[:, 1, 1].unsqueeze(1).unsqueeze(2)
        cx2 = K2[:, 0, 2].unsqueeze(1).unsqueeze(2)
        cy2 = K2[:, 1, 2].unsqueeze(1).unsqueeze(2)
        z2 = pts3d_displaced[:, :, :, 2].clamp(min=1e-06)
        u2 = fx2 * pts3d_displaced[:, :, :, 0] / z2 + cx2
        v2 = fy2 * pts3d_displaced[:, :, :, 1] / z2 + cy2
        u2_avg = u2.mean(dim=2)
        v2_avg = v2.mean(dim=2)
        u2_feat = u2_avg / patch_size
        v2_feat = v2_avg / patch_size
        u2_norm = 2.0 * u2_feat / max(nW - 1, 1) - 1.0
        v2_norm = 2.0 * v2_feat / max(nH - 1, 1) - 1.0
        grid = torch.stack([u2_norm, v2_norm], dim=-1).reshape(B, nH, nW, 2)
        feat_spatial = features_t.permute(0, 2, 1).reshape(B, D, nH, nW)
        warped_spatial = F.grid_sample(
            feat_spatial,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        warped = warped_spatial.reshape(B, D, -1).permute(0, 2, 1)
        return warped


class FlowDecoderV5(nn.Module):

    def __init__(
        self,
        model_dim: int = 512,
        latent_dim: int = 128,
        patch_size: int = 14,
        dec_blocks: int = 8,
        num_heads: int = 16,
        in_dim: int = 0,
        motion_hint_dim: int = 16,
        dropout: float = 0.0,
        use_rope: bool = False,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.patch_size = patch_size
        patch_token_dim = 3 * patch_size * patch_size
        actual_in_dim = in_dim if in_dim > 0 else patch_token_dim
        total_in_dim = actual_in_dim + motion_hint_dim
        self.patch_embed = nn.Linear(total_in_dim, model_dim)
        from .blocks import AdaLNSpatioTransformer

        self.transformer = AdaLNSpatioTransformer(
            in_dim=model_dim,
            model_dim=model_dim,
            out_dim=model_dim,
            num_blocks=dec_blocks,
            num_heads=num_heads,
            dropout=dropout,
            latent_dim=latent_dim,
            use_rope=use_rope,
        )
        self.decoder_norm = nn.LayerNorm(model_dim)
        self.flow_head = nn.Linear(model_dim, 3 * patch_size * patch_size)

    def forward(self, patches, latent_z, motion_hints, H, W):
        from .blocks import unpatchify

        B, T_m1, N, _ = patches.shape
        nH = -(-H // self.patch_size)
        nW = -(-W // self.patch_size)
        hints = motion_hints.unsqueeze(1).expand(-1, T_m1, -1, -1)
        x = torch.cat([patches, hints], dim=-1)
        x = self.patch_embed(x)
        x = self.transformer(x, latent_z, grid_hw=(nH, nW))
        x = self.decoder_norm(x)
        x_flat = x.reshape(B * T_m1, N, self.model_dim)
        flow_patches = self.flow_head(x_flat)
        flow_patches = flow_patches.reshape(
            B, T_m1, N, 3 * self.patch_size * self.patch_size
        )
        flow_3d = unpatchify(flow_patches, self.patch_size, H, W)
        return {"dynamic_flow_3d": flow_3d}


class CameraDecV5(nn.Module):

    def __init__(self, latent_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_rot = nn.Linear(hidden_dim, 3)
        self.fc_trans = nn.Linear(hidden_dim, 3)

    def forward(self, latent_z):
        feat = self.backbone(latent_z)
        rot = self.fc_rot(feat)
        trans = self.fc_trans(feat)
        return torch.cat([rot, trans], dim=-1)


class _GradientReversal(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return (-ctx.alpha * grad_output, None)


def gradient_reversal(x, alpha=1.0):
    return _GradientReversal.apply(x, alpha)


class _ReconBlock(nn.Module):

    def __init__(self, model_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(model_dim, elementwise_affine=False, eps=1e-06)
        self.attn = SelfAttention(model_dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(model_dim, elementwise_affine=False, eps=1e-06)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 4, model_dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class HintCompressor(nn.Module):

    def __init__(self, hint_dim: int, num_tokens: int, num_heads: int = 8,
                 bottleneck_dim: int = 0):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, num_tokens, hint_dim) * 0.02)
        self.attn = nn.MultiheadAttention(hint_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hint_dim)
        # DreamDojo-style narrow bottleneck (mirrors LAM): project each token
        # hint_dim -> d so the delivered latent is K*d. 0 = off (full hint_dim).
        self.bottleneck_dim = bottleneck_dim
        self.out_dim = bottleneck_dim if bottleneck_dim > 0 else hint_dim
        if bottleneck_dim > 0:
            self.bottleneck = nn.Linear(hint_dim, bottleneck_dim)

    def forward(self, motion_hints: torch.Tensor) -> torch.Tensor:
        q = self.queries.expand(motion_hints.shape[0], -1, -1)
        o, _ = self.attn(q, motion_hints, motion_hints)
        o = self.norm(o)
        if self.bottleneck_dim > 0:
            o = self.bottleneck(o)
        return o


class ReconDecoderKToken(nn.Module):

    def __init__(
        self,
        in_dim,
        model_dim,
        out_dim,
        hint_dim,
        num_tokens,
        num_blocks=4,
        num_heads=16,
        dropout=0.0,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.feat_proj = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, model_dim)
        )
        self.token_proj = nn.Linear(hint_dim, model_dim)
        self.pos_enc = PositionalEncoding(model_dim)
        self.blocks = nn.ModuleList(
            [
                _ReconBlock(model_dim, num_heads, dropout=dropout)
                for _ in range(num_blocks)
            ]
        )
        self.out_norm = nn.LayerNorm(model_dim, elementwise_affine=False, eps=1e-06)
        self.out_proj = nn.Linear(model_dim, out_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self, source_features: torch.Tensor, k_tokens: torch.Tensor
    ) -> torch.Tensor:
        x = self.feat_proj(source_features)
        k = self.token_proj(k_tokens)
        seq = torch.cat([x, k], dim=1)
        seq = self.pos_enc(seq.unsqueeze(1)).squeeze(1)
        for blk in self.blocks:
            seq = blk(seq)
        delta = self.out_proj(self.out_norm(seq[:, : x.shape[1]]))
        return source_features + delta


class RynnLAM(nn.Module):
    """DA3-Large pair encoder with flow and K-token reconstruction heads."""

    def __init__(
        self,
        patch_size: int = 14,
        encoder_backbone: Literal["large"] = "large",
        embed_dim: int = 1024,
        encoder_finetune_mode: Literal["freeze", "full"] = "freeze",
        encoder_checkpoint_path: Optional[str] = None,
        latent_dim: int = 64,
        camera_pose_latent_dim: int = 32,
        latent_encoder_model_dim: int = 512,
        latent_encoder_num_heads: int = 16,
        latent_encoder_num_blocks: int = 6,
        flow_decoder_model_dim: int = 512,
        flow_decoder_num_heads: int = 16,
        flow_decoder_dec_blocks: int = 8,
        flow_mask_ratio: float = 0.5,
        motion_hint_dim: int = 128,
        warp_num_sample_points: int = 8,
        cam_dec_hidden_dim: int = 256,
        adversarial_alpha: float = 2.0,
        use_rope: bool = False,
        motion_hints_dropout: float = 0.8,
        recon_decoder_model_dim: int = 512,
        recon_decoder_num_heads: int = 16,
        recon_decoder_num_blocks: int = 4,
        num_k_tokens: int = 8,
        k_token_source: Literal["features", "hints"] = "features",
        k_token_bottleneck_dim: int = 0,
        log_z_utilization: bool = True,
    ):
        super().__init__()
        if encoder_backbone != "large" or embed_dim != 1024:
            raise ValueError("RynnLAM requires DA3-Large (embed_dim=1024)")
        if encoder_finetune_mode not in ("freeze", "full"):
            raise ValueError("encoder_finetune_mode must be freeze or full")
        if patch_size != 14:
            raise ValueError("DA3-Large requires patch_size=14")
        if k_token_source not in ("features", "hints"):
            raise ValueError("k_token_source must be features or hints")
        if num_k_tokens < 1:
            raise ValueError("num_k_tokens must be positive")
        if not 0 <= motion_hints_dropout < 1 or not 0 <= flow_mask_ratio <= 1:
            raise ValueError("invalid motion_hints_dropout or flow_mask_ratio")
        self.latent_dim = latent_dim
        self.camera_pose_latent_dim = camera_pose_latent_dim
        self.patch_size = patch_size
        self.flow_mask_ratio = flow_mask_ratio
        self.adversarial_alpha = adversarial_alpha
        self.use_rope = use_rope
        self.motion_hints_dropout = motion_hints_dropout
        self.embed_dim = embed_dim
        self.log_z_utilization = log_z_utilization
        feature_dim = embed_dim * 2

        # Registration and initialization order match production checkpoints.
        self.encoder = DA3ViTLargeEncoder(
            finetune_mode=encoder_finetune_mode,
            checkpoint_path=encoder_checkpoint_path,
        )
        self.latent_encoder = CrossAttentionActionEncoderV5(
            in_dim=feature_dim,
            model_dim=latent_encoder_model_dim,
            latent_action_dim=latent_dim,
            camera_pose_latent_dim=camera_pose_latent_dim,
            num_heads=latent_encoder_num_heads,
            num_blocks=latent_encoder_num_blocks,
            motion_hint_dim=motion_hint_dim,
        )
        self.cam_dec = CameraDecV5(camera_pose_latent_dim, cam_dec_hidden_dim)
        self.cam_dec_adversarial = CameraDecV5(latent_dim, cam_dec_hidden_dim)
        self.flow_decoder = FlowDecoderV5(
            model_dim=flow_decoder_model_dim,
            latent_dim=latent_dim,
            patch_size=patch_size,
            dec_blocks=flow_decoder_dec_blocks,
            num_heads=flow_decoder_num_heads,
            in_dim=feature_dim + patch_size * patch_size,
            motion_hint_dim=motion_hint_dim,
            use_rope=use_rope,
        )
        self.flow_mask_token = nn.Parameter(torch.zeros(1, 1, 1, feature_dim))
        nn.init.normal_(self.flow_mask_token, std=0.02)
        self.feature_warp = FeatureWarpV5(num_sample_points=warp_num_sample_points)
        self.target_feature_dim = feature_dim
        self.num_k_tokens = num_k_tokens
        self.k_token_source = k_token_source
        self.k_token_bottleneck_dim = k_token_bottleneck_dim
        token_dim = motion_hint_dim if k_token_source == "hints" else feature_dim
        # With a narrow bottleneck the compressor emits k_token_bottleneck_dim per
        # token, so the recon decoder's token_proj input dim must match it.
        token_out_dim = (
            k_token_bottleneck_dim if k_token_bottleneck_dim > 0 else token_dim
        )
        self.k_token_out_dim = token_out_dim
        self.hint_compressor = HintCompressor(
            token_dim, num_k_tokens, bottleneck_dim=k_token_bottleneck_dim
        )
        self.recon_decoder_ktoken = ReconDecoderKToken(
            in_dim=feature_dim,
            model_dim=recon_decoder_model_dim,
            out_dim=feature_dim,
            hint_dim=token_out_dim,
            num_tokens=num_k_tokens,
            num_blocks=recon_decoder_num_blocks,
            num_heads=recon_decoder_num_heads,
        )
        self.static_flow_from_pose = StaticFlowFromPose()

    def _apply_mask(self, x, mask_ratio, mask_token):
        if mask_ratio <= 0:
            return x
        B, T, N, D = x.shape
        num_mask = int(N * mask_ratio)
        if num_mask == 0:
            return x
        noise = torch.rand(B, T, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=2)
        ids_mask = ids_shuffle[:, :, :num_mask]
        mask = torch.zeros(B, T, N, 1, device=x.device, dtype=x.dtype)
        mask.scatter_(2, ids_mask.unsqueeze(-1), 1.0)
        return x * (1 - mask) + mask_token.expand(B, T, N, -1) * mask

    def encode_pair(self, images, *, normalize=True):
        """Encode float RGB [B,2,H,W,3] in [0,1]; return unflattened latents.

        normalize=False reproduces the historical unnormalized extractor.
        This method preserves autograd and does not change train/eval mode.
        """
        if images.ndim != 5 or images.shape[1] != 2 or images.shape[-1] != 3:
            raise ValueError("images must have shape [B,2,H,W,3]")
        if not images.is_floating_point():
            raise TypeError("images must be floating-point RGB in [0,1]")
        if normalize:
            mean = images.new_tensor([0.485, 0.456, 0.406])
            std = images.new_tensor([0.229, 0.224, 0.225])
            images = (images - mean) / std
        outputs, _ = self.encoder(images)
        features = outputs[-1][0]
        source, target = features[:, 0], features[:, 1]
        action, camera, hints = self.latent_encoder(source, target)
        # The feature-source reconstruction branch does not update DA3.
        compressor_input = (
            target.detach() if self.k_token_source == "features" else hints
        )
        tokens = self.hint_compressor(compressor_input)
        return {
            "source_features": source,
            "target_features": target,
            "latent_action": action,
            "camera_pose_latent": camera,
            "motion_hints": hints,
            "k_tokens": tokens,
        }

    def forward(self, images, extrinsics, intrinsics, depths):
        H, W = images.shape[2:4]
        encoded = self.encode_pair(images, normalize=True)
        features_t = encoded["source_features"]
        features_tn = encoded["target_features"]
        latent_action = encoded["latent_action"]
        camera_pose_latent = encoded["camera_pose_latent"]
        motion_hints = encoded["motion_hints"]
        k_tokens = encoded["k_tokens"]
        if self.training and self.motion_hints_dropout > 0:
            keep = (
                torch.rand(
                    motion_hints.shape[0],
                    motion_hints.shape[1],
                    1,
                    device=motion_hints.device,
                )
                >= self.motion_hints_dropout
            )
            motion_hints = motion_hints * keep / (1.0 - self.motion_hints_dropout)
        latent_action_expanded = latent_action.unsqueeze(1)
        pose_pred = self.cam_dec(camera_pose_latent)
        pose_from_action = self.cam_dec_adversarial(
            gradient_reversal(latent_action, self.adversarial_alpha)
        )
        flow_features = features_t.unsqueeze(1).detach()
        if self.training and self.flow_mask_ratio > 0:
            flow_features = self._apply_mask(
                flow_features, self.flow_mask_ratio, self.flow_mask_token
            )
        depth_patches = patchify(depths[:, :-1].unsqueeze(-1), self.patch_size)
        flow_input = torch.cat([flow_features, depth_patches], dim=-1)
        dynamic_flow = self.flow_decoder(
            flow_input, latent_action_expanded, motion_hints, H, W
        )["dynamic_flow_3d"]
        static_flow = self.static_flow_from_pose(pose_pred, depths, intrinsics)
        recon_features = self.recon_decoder_ktoken(features_t.detach(), k_tokens)
        recon_features_zeroz = None
        if self.training:
            with torch.no_grad():
                recon_features_zeroz = self.recon_decoder_ktoken(
                    features_t.detach(), torch.zeros_like(k_tokens)
                )
        warped_features = self.feature_warp(
            features_t=features_t.detach(),
            flow_3d=dynamic_flow,
            depths=depths,
            intrinsics=intrinsics,
            patch_size=self.patch_size,
            H=H,
            W=W,
        )
        return {
            "latent_action": latent_action_expanded,
            "camera_pose_latent": camera_pose_latent,
            "pose_pred": pose_pred,
            "pose_from_action": pose_from_action,
            "dynamic_flow": dynamic_flow,
            "static_flow": static_flow,
            "flow_3d": dynamic_flow,
            "refined_features": warped_features,
            "refined_features_zeroz": (
                warped_features if self.log_z_utilization else None
            ),
            "warped_features": warped_features,
            "recon_features": recon_features,
            "recon_features_zeroz": recon_features_zeroz,
            "target_features": features_tn.detach(),
            "source_features": features_t.detach(),
            "motion_hints": motion_hints,
            "k_tokens": k_tokens,
        }
