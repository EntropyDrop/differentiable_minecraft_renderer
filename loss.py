import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from differentiable_renderer import DifferentiableRenderer

class MinecraftRenderLoss(nn.Module):
    def __init__(self, mappings_dir=None, bg_color=(1/255, 1/255, 1/255), 
                 lambda_lpips=0.0, lambda_mse=1.0, use_lpips=False, views=None,
                 foreground_weight=0.0):
        super().__init__()
        self.renderer = DifferentiableRenderer(mappings_dir=mappings_dir, bg_color=bg_color)
        self.lambda_lpips = lambda_lpips
        self.lambda_mse = lambda_mse
        self.foreground_weight = foreground_weight
        
        if views is not None:
            if isinstance(views, str):
                views = [v.strip() for v in views.split(",") if v.strip()]
            self.views = [v for v in views if v in self.renderer.views]
            missing_views = [v for v in views if v not in self.renderer.views]
            if missing_views:
                print(f"WARNING: Ignoring unknown Minecraft render views: {missing_views}")
            if len(self.views) == 0:
                raise ValueError(f"No valid Minecraft render views selected. Available views: {self.renderer.views}")
            print(f"MinecraftRenderLoss configured with subset of views: {self.views}")
        else:
            self.views = self.renderer.views
        self.view_count = max(1, len(self.views))

        self.lpips_loss_fn = None
        if use_lpips:
            try:
                import lpips
                # We use the standard AlexNet backbone for LPIPS as it is fast and perceptually accurate
                self.lpips_loss_fn = lpips.LPIPS(net='alex')
                self.lpips_loss_fn.eval()
                self.lpips_loss_fn.requires_grad_(False)
                print("LPIPS loss module loaded successfully.")
            except ImportError:
                print("WARNING: 'lpips' package not found. Falling back to MSE-only loss.")
                print("Please install lpips via: pip install lpips")

    def forward(self, skins_pred, skins_gt):
        """
        Calculates the combined rendering alignment loss (LPIPS + MSE) across the selected views.
        Args:
            skins_pred: Predicted skin tensor of shape (B, 4, 64, 64) with range [0, 1].
            skins_gt: Ground truth skin tensor of shape (B, 4, 64, 64) with range [0, 1].
        Returns:
            dict with 'loss_total', 'loss_lpips', and 'loss_mse'.
        """
        B, C, H, W = skins_pred.shape
        assert skins_pred.shape == skins_gt.shape, "Shape mismatch between prediction and ground truth skins"
        
        # Render both skins across all selected views
        # Note: self.renderer does forward for all views, let's optimize to only forward_view for selected views
        renders_pred = {}
        renders_gt = {}
        for view in self.views:
            renders_pred[view] = self.renderer.forward_view(skins_pred, view)
            renders_gt[view] = self.renderer.forward_view(skins_gt, view)
        
        loss_lpips = torch.tensor(0.0, device=skins_pred.device)
        loss_mse = torch.tensor(0.0, device=skins_pred.device)
        
        for view in self.views:
            pred_view = renders_pred[view] # (B, 4, H_out, W_out)
            gt_view = renders_gt[view]     # (B, 4, H_out, W_out)
            
            # 1. Compute MSE Loss (on RGB channels). The foreground term prevents gray background
            # pixels from washing out character differences.
            full_mse = F.mse_loss(pred_view[:, :3], gt_view[:, :3])
            if self.foreground_weight > 0:
                fg_mask = torch.maximum(pred_view[:, 3:4], gt_view[:, 3:4]).detach()
                fg_denom = fg_mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
                fg_mse = (((pred_view[:, :3] - gt_view[:, :3]) ** 2) * fg_mask).sum(dim=(1, 2, 3)) / (fg_denom * 3.0)
                loss_mse += full_mse + self.foreground_weight * fg_mse.mean()
            else:
                loss_mse += full_mse
            
            # 2. Compute LPIPS Loss (on RGB channels)
            if self.lpips_loss_fn is not None:
                # LPIPS expects input tensors to be normalized to [-1, 1]
                pred_rgb = pred_view[:, :3] * 2.0 - 1.0
                gt_rgb = gt_view[:, :3] * 2.0 - 1.0
                
                # Compute LPIPS score
                # The output has shape (B, 1, 1, 1), we take the mean over the batch
                loss_lpips += self.lpips_loss_fn(pred_rgb, gt_rgb).mean()
                
        loss_lpips = loss_lpips / self.view_count
        loss_mse = loss_mse / self.view_count
        loss_total = self.lambda_lpips * loss_lpips + self.lambda_mse * loss_mse
        
        return {
            "loss_total": loss_total,
            "loss_lpips": loss_lpips,
            "loss_mse": loss_mse
        }
