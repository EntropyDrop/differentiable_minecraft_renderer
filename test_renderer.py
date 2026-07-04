import os
import sys
import numpy as np
import torch
from PIL import Image

from mc_skin_utils import mc_render
from differentiable_renderer import DifferentiableRenderer

from config import views, walk_rot, static_rot, walk_offset, static_offset, get_views, parse_sizes

def run_tests(selected_views=None, size_arg=None):
    if size_arg is not None:
        parsed = parse_sizes(size_arg)
        test_views = get_views(parsed[0])
    else:
        test_views = views

    # Find a test skin
    skin_path = os.path.join(os.path.dirname(__file__), "skins", "a5c3a615940c35cf.png")
    if not os.path.exists(skin_path):
        raise FileNotFoundError(f"Test skin not found at: {skin_path}")
        
    print(f"Loading test skin: {skin_path}")
    # Load and preprocess using standard PIL
    skin_img = Image.open(skin_path).convert("RGBA")
    skin_np = np.array(skin_img)
    
    # Clean semi-transparency to match ensure_valid_skin preprocessing in build_target_img
    alpha = skin_np[..., 3]
    semi_transparent = (alpha > 0) & (alpha < 255)
    skin_np[semi_transparent, 3] = 255
    
    # Prepare PyTorch tensor (B, C, H, W), normalized to [0, 1]
    skin_tensor = torch.tensor(skin_np, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
    
    # Determine mapping dir for test_views size
    sample_param = next(iter(test_views.values()))
    w_out, h_out = sample_param["output_size"]
    target_mappings_dir = os.path.join(os.path.dirname(__file__), f"mappings_{w_out}x{h_out}")
    if not os.path.exists(target_mappings_dir):
        target_mappings_dir = None

    # Initialize PyTorch differentiable renderer
    renderer = DifferentiableRenderer(mappings_dir=target_mappings_dir, bg_color=(1/255, 1/255, 1/255))
    
    # Create output directories
    output_dir = os.path.join(os.path.dirname(__file__), "test_outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    if selected_views is not None:
        selected_views = [view.strip() for view in selected_views.split(",") if view.strip()]
        missing_views = [view for view in selected_views if view not in test_views]
        if missing_views:
            raise ValueError(f"Unknown views {missing_views}. Available views: {', '.join(test_views)}")
        selected_views = set(selected_views)

    view_count = len(selected_views) if selected_views is not None else len(test_views)
    print(f"Running rendering comparison for {view_count} views (size {w_out}x{h_out})...")
    for view_name, params in test_views.items():
        if selected_views is not None and view_name not in selected_views:
            continue
        print(f"\n--- Testing View: {view_name} ---")
        rot_args = walk_rot if params["walk"] else static_rot
        offset_args = walk_offset if params["walk"] else static_offset
        output_size = params["output_size"]
        
        # 1. Render using PyVista (Ground Truth)
        pv_render = mc_render.render_skin(
            skin=Image.fromarray(skin_np),
            output_size=output_size,
            cam_front=params["cam_front"],
            zoom=params["zoom"],
            look_at_y=params["look_at_y"],
            use_voxels=False,
            light=False,
            transparent_background=True,
            core_display=params["core_display"],
            decor_display=params["decor_display"],
            rot_args=rot_args,
            offset_args=offset_args,
            ortho=params.get("ortho", False),
            off_screen=True
        )
        
        # 2. Render using PyTorch (Differentiable)
        pt_render_tensor = renderer.forward_view(skin_tensor, view_name)
        # Convert to numpy uint8 (H, W, 4)
        pt_render_np = (pt_render_tensor.squeeze(0).permute(1, 2, 0).detach().numpy() * 255.0).astype(np.uint8)
        
        # Save output images
        pv_img_path = os.path.join(output_dir, f"{view_name}_pyvista.png")
        pt_img_path = os.path.join(output_dir, f"{view_name}_pytorch.png")
        
        Image.fromarray(pv_render).save(pv_img_path)
        Image.fromarray(pt_render_np).save(pt_img_path)
        print(f"Saved PyVista: {pv_img_path}")
        print(f"Saved PyTorch: {pt_img_path}")
        
        # 3. Calculate metrics
        pv_float = pv_render.astype(np.float32) / 255.0
        pt_float = pt_render_np.astype(np.float32) / 255.0
        
        mse = np.mean((pv_float - pt_float) ** 2)
        print(f"MSE between PyVista and PyTorch for {view_name}: {mse:.6f}")
        
        if mse < 0.005:
            print(f"PASS: PyTorch renderer matches PyVista for {view_name}.")
        else:
            print(f"FAIL: PyTorch renderer deviates from PyVista for {view_name}.")
            
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare PyVista and PyTorch differentiable renderer outputs.")
    parser.add_argument("--views", default=None, help="Optional comma-separated subset of view names to test.")
    parser.add_argument("--size", default=None, help="Optional size specification, e.g. 512x1024.")
    args = parser.parse_args()
    run_tests(selected_views=args.views, size_arg=args.size)
