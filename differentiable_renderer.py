import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class DifferentiableRenderer(nn.Module):
    def __init__(self, mappings_dir=None, bg_color=(128/255, 128/255, 128/255)):
        super().__init__()
        if mappings_dir is None:
            mappings_dir = os.path.join(os.path.dirname(__file__), "mappings")
            
        self.mappings_dir = mappings_dir
        self.register_buffer("bg_color", torch.tensor(bg_color, dtype=torch.float32))
        
        self.views = []
        
        # Load and pre-process mappings dynamically
        for file_name in sorted(os.listdir(self.mappings_dir)):
            if file_name.endswith("_mapping.pt"):
                view_name = file_name[:-11]  # Remove "_mapping.pt"
                self.views.append(view_name)
                
                mapping_path = os.path.join(self.mappings_dir, file_name)
                data = torch.load(mapping_path)
                
                # Extract and process inner layer
                inner_uv = data["inner_uv_map"]  # (H, W, 2)
                inner_mask = data["inner_mask"]  # (H, W)
                
                # Extract and process outer layer
                outer_uv = data["outer_uv_map"]  # (H, W, 2)
                outer_mask = data["outer_mask"]  # (H, W)
                
                # Normalize UV coordinates [0, 63] to [-1, 1] for grid_sample
                inner_grid = torch.zeros_like(inner_uv)
                inner_grid[..., 0] = (inner_uv[..., 0] / 63.0) * 2.0 - 1.0
                inner_grid[..., 1] = (inner_uv[..., 1] / 63.0) * 2.0 - 1.0
                
                outer_grid = torch.zeros_like(outer_uv)
                outer_grid[..., 0] = (outer_uv[..., 0] / 63.0) * 2.0 - 1.0
                outer_grid[..., 1] = (outer_uv[..., 1] / 63.0) * 2.0 - 1.0
                
                # Register buffer so they are moved to device automatically
                self.register_buffer(f"{view_name}_inner_grid", inner_grid)
                self.register_buffer(f"{view_name}_inner_mask", inner_mask)
                self.register_buffer(f"{view_name}_outer_grid", outer_grid)
                self.register_buffer(f"{view_name}_outer_mask", outer_mask)

    def forward_view(self, skins, view_name):
        """
        Render a single view for a batch of skins.
        Args:
            skins: PyTorch tensor of shape (B, 4, 64, 64) with values in range [0, 1].
                   The channel order is RGBA.
            view_name: One of ["front", "back", "left", "right"].
        Returns:
            rendered: PyTorch tensor of shape (B, 3, H, W) or (B, 4, H, W) depending on background blending.
                      Returns RGBA image.
        """
        B, C, H_in, W_in = skins.shape
        assert C == 4, "Skins must have 4 channels (RGBA)"
        assert H_in == 64 and W_in == 64, "Skins must be 64x64"
        
        # Retrieve buffers
        inner_grid = getattr(self, f"{view_name}_inner_grid").unsqueeze(0).expand(B, -1, -1, -1)
        inner_mask = getattr(self, f"{view_name}_inner_mask").unsqueeze(0).unsqueeze(1).expand(B, -1, -1, -1) # (B, 1, H, W)
        
        outer_grid = getattr(self, f"{view_name}_outer_grid").unsqueeze(0).expand(B, -1, -1, -1)
        outer_mask = getattr(self, f"{view_name}_outer_mask").unsqueeze(0).unsqueeze(1).expand(B, -1, -1, -1) # (B, 1, H, W)
        
        # 1. Sample inner layer
        # sampled has shape (B, 4, H_out, W_out)
        inner_sampled = F.grid_sample(skins, inner_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        # Apply static visibility mask
        inner_sampled = inner_sampled * inner_mask
        
        # 2. Sample outer layer
        outer_sampled = F.grid_sample(skins, outer_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        # Apply static visibility mask
        outer_sampled = outer_sampled * outer_mask
        
        # 3. Alpha blend outer over inner
        # We extract RGB and Alpha for inner and outer layers
        inner_rgb = inner_sampled[:, :3, :, :]
        inner_alpha = inner_sampled[:, 3:4, :, :]
        
        outer_rgb = outer_sampled[:, :3, :, :]
        outer_alpha = outer_sampled[:, 3:4, :, :] # This is the dynamic alpha of the outer layer
        
        # Background color
        bg = self.bg_color.view(1, 3, 1, 1).expand(B, -1, inner_rgb.shape[2], inner_rgb.shape[3])
        
        # Composite inner layer over background
        # Note: Minecraft skin inner layers are usually opaque, but we blend using alpha to support transparency if any.
        inner_composite = inner_alpha * inner_rgb + (1.0 - inner_alpha) * bg
        
        # Composite outer layer over inner composite
        final_rgb = outer_alpha * outer_rgb + (1.0 - outer_alpha) * inner_composite
        
        # Calculate final alpha mask (for background/foreground separation)
        final_alpha = outer_alpha + (1.0 - outer_alpha) * inner_alpha
        
        return torch.cat([final_rgb, final_alpha], dim=1)

    def forward(self, skins):
        """
        Renders all 4 views for a batch of skins.
        Args:
            skins: PyTorch tensor of shape (B, 4, 64, 64) with values in range [0, 1].
        Returns:
            dict of rendered views. Each view has shape (B, 4, H, W).
        """
        results = {}
        for view_name in self.views:
            results[view_name] = self.forward_view(skins, view_name)
        return results
