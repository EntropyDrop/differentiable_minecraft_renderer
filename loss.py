import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from differentiable_renderer import DifferentiableRenderer

class MinecraftRenderLoss(nn.Module):
    def __init__(self, mappings_dir=None, bg_color=(1/255, 1/255, 1/255), 
                 lambda_lpips=1.0, lambda_mse=1.0, use_lpips=True):
        super().__init__()
        self.renderer = DifferentiableRenderer(mappings_dir=mappings_dir, bg_color=bg_color)
        self.lambda_lpips = lambda_lpips
        self.lambda_mse = lambda_mse
        
        self.lpips_loss_fn = None
        if use_lpips:
            try:
                import lpips
                # We use the standard AlexNet backbone for LPIPS as it is fast and perceptually accurate
                self.lpips_loss_fn = lpips.LPIPS(net='alex')
                print("LPIPS loss module loaded successfully.")
            except ImportError:
                print("WARNING: 'lpips' package not found. Falling back to MSE-only loss.")
                print("Please install lpips via: pip install lpips")

    def forward(self, skins_pred, skins_gt):
        """
        Calculates the combined rendering alignment loss (LPIPS + MSE) across the 4 static views.
        Args:
            skins_pred: Predicted skin tensor of shape (B, 4, 64, 64) with range [0, 1].
            skins_gt: Ground truth skin tensor of shape (B, 4, 64, 64) with range [0, 1].
        Returns:
            dict with 'loss_total', 'loss_lpips', and 'loss_mse'.
        """
        B, C, H, W = skins_pred.shape
        assert skins_pred.shape == skins_gt.shape, "Shape mismatch between prediction and ground truth skins"
        
        # Render both skins across all 4 views
        renders_pred = self.renderer(skins_pred)
        renders_gt = self.renderer(skins_gt)
        
        loss_lpips = torch.tensor(0.0, device=skins_pred.device)
        loss_mse = torch.tensor(0.0, device=skins_pred.device)
        
        for view in self.renderer.views:
            pred_view = renders_pred[view] # (B, 4, H_out, W_out)
            gt_view = renders_gt[view]     # (B, 4, H_out, W_out)
            
            # 1. Compute MSE Loss (on RGB channels)
            loss_mse += F.mse_loss(pred_view[:, :3], gt_view[:, :3])
            
            # 2. Compute LPIPS Loss (on RGB channels)
            if self.lpips_loss_fn is not None:
                # LPIPS expects input tensors to be normalized to [-1, 1]
                pred_rgb = pred_view[:, :3] * 2.0 - 1.0
                gt_rgb = gt_view[:, :3] * 2.0 - 1.0
                
                # Compute LPIPS score
                # The output has shape (B, 1, 1, 1), we take the mean over the batch
                loss_lpips += self.lpips_loss_fn(pred_rgb, gt_rgb).mean()
                
        loss_total = self.lambda_lpips * loss_lpips + self.lambda_mse * loss_mse
        
        return {
            "loss_total": loss_total,
            "loss_lpips": loss_lpips,
            "loss_mse": loss_mse
        }
