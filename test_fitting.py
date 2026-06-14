import os
import sys
import torch
import torch.optim as optim
from PIL import Image
import numpy as np

from differentiable_renderer import DifferentiableRenderer

def run_fitting():
    # 1. Load target skin
    skin_path = os.path.join(os.path.dirname(__file__), "skins", "408097.png")
    if not os.path.exists(skin_path):
        raise FileNotFoundError(f"Target skin not found at: {skin_path}")
        
    skin_img = Image.open(skin_path).convert("RGBA")
    
    # Run voxel consistency resolver if any
    from mc_skin_utils.mc_voxel_texture_resolver import resolve_voxel_consistency
    skin_img = resolve_voxel_consistency(skin_img)
    
    skin_np = np.array(skin_img)
    target_skin = torch.tensor(skin_np, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
    
    # 2. Render target views using DifferentiableRenderer (to act as ground truth)
    renderer = DifferentiableRenderer(bg_color=(1/255, 1/255, 1/255))
    target_renders = renderer(target_skin)
    
    # 3. Initialize random learnable skin UV map
    # We initialize with a noisy version of the target or pure random noise to fit
    learnable_skin = torch.rand((1, 4, 64, 64), dtype=torch.float32, requires_grad=True)
    
    # Setup optimizer
    optimizer = optim.Adam([learnable_skin], lr=0.02)
    
    print("Starting gradient-based optimization loop...")
    print("Goal: Reconstruct the original 64x64 skin UV map using ONLY the 4 rendered views.")
    
    for step in range(501):
        optimizer.zero_grad()
        
        # Render current prediction from the 4 views
        pred_renders = renderer(learnable_skin)
        
        # Compute loss (MSE across all 4 views)
        loss = 0.0
        for view in renderer.views:
            loss += torch.mean((pred_renders[view] - target_renders[view]) ** 2)
            
        loss.backward()
        optimizer.step()
        
        # Project back to valid range [0, 1]
        with torch.no_grad():
            learnable_skin.clamp_(0.0, 1.0)
            
        if step % 50 == 0:
            print(f"Step {step:03d} | Loss: {loss.item():.6f}")
            
    # Save target and reconstructed skins for visual verification
    output_dir = os.path.join(os.path.dirname(__file__), "test_outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    reconstructed_np = (learnable_skin.squeeze(0).permute(1, 2, 0).detach().numpy() * 255.0).astype(np.uint8)
    Image.fromarray(skin_np).save(os.path.join(output_dir, "fitting_target.png"))
    Image.fromarray(reconstructed_np).save(os.path.join(output_dir, "fitting_reconstructed.png"))
    
    print("\nFitting completed successfully!")
    print(f"Target skin saved to: {os.path.join(output_dir, 'fitting_target.png')}")
    print(f"Reconstructed skin saved to: {os.path.join(output_dir, 'fitting_reconstructed.png')}")

if __name__ == "__main__":
    run_fitting()
