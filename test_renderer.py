import os
import sys
import numpy as np
import torch
from PIL import Image

from mc_skin_utils import mc_render
from differentiable_renderer import DifferentiableRenderer

def run_tests():
    # Find a test skin
    skin_path = os.path.join(os.path.dirname(__file__), "skins", "lowpoly", "408097.png")
    if not os.path.exists(skin_path):
        raise FileNotFoundError(f"Test skin not found at: {skin_path}")
        
    print(f"Loading test skin: {skin_path}")
    # Load and preprocess using standard PIL
    skin_img = Image.open(skin_path).convert("RGBA")
    
    # Run voxel consistency resolver if any
    from mc_skin_utils.mc_voxel_texture_resolver import resolve_voxel_consistency
    skin_img = resolve_voxel_consistency(skin_img)
    skin_np = np.array(skin_img)
    
    # Clean semi-transparency to match ensure_valid_skin preprocessing in build_target_img
    # Any alpha between 0 and 255 is made fully opaque (255)
    alpha = skin_np[..., 3]
    semi_transparent = (alpha > 0) & (alpha < 255)
    skin_np[semi_transparent, 3] = 255
    
    # Prepare PyTorch tensor (B, C, H, W), normalized to [0, 1]
    skin_tensor = torch.tensor(skin_np, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
    
    # Initialize PyTorch differentiable renderer
    renderer = DifferentiableRenderer(bg_color=(1/255, 1/255, 1/255))
    
    # Create output directories
    output_dir = os.path.join(os.path.dirname(__file__), "test_outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    full_part = ['head', 'body', 'left_arm', 'right_arm', 'left_leg', 'right_leg']
    
    # View configuration (matching generate_mappings.py)
    views = {
        "walk_perspective": {
            "cam_front": (0.3, 0.4, 0.5), "zoom": 0.22, "look_at_y": 12, "walk": True,
            "core_display": full_part, "decor_display": full_part, "output_size": (768, 920)
        },
        "walk_perspective_back": {
            "cam_front": (-0.3, 0.4, -0.5), "zoom": 0.22, "look_at_y": 12, "walk": True,
            "core_display": full_part, "decor_display": full_part, "output_size": (768, 920)
        },
        "walk_perspective_ortho": {
            "cam_front": (0.3, 0.1, 0.5), "zoom": 0.23, "look_at_y": 16, "walk": True,
            "core_display": full_part, "decor_display": full_part, "output_size": (768, 920), "ortho": True
        },
        "walk_perspective_back_ortho": {
            "cam_front": (-0.3, 0.1, -0.5), "zoom": 0.23, "look_at_y": 16, "walk": True,
            "core_display": full_part, "decor_display": full_part, "output_size": (768, 920), "ortho": True
        },
        "static_front": {
            "cam_front": (0.0, 0.0, 0.5), "zoom": 0.35, "look_at_y": 12, "walk": False,
            "core_display": full_part, "decor_display": full_part, "output_size": (306, 512)
        },
        "static_back": {
            "cam_front": (-0.0, -0.0, -0.5), "zoom": 0.35, "look_at_y": 12, "walk": False,
            "core_display": full_part, "decor_display": full_part, "output_size": (306, 512)
        },
        "front": {
            "cam_front": (0.0, 0.0, 0.5), "zoom": 0.35, "look_at_y": 12, "walk": False,
            "core_display": full_part, "decor_display": full_part, "output_size": (512, 512)
        },
        "back": {
            "cam_front": (0.0, 0.0, -0.5), "zoom": 0.35, "look_at_y": 12, "walk": False,
            "core_display": full_part, "decor_display": full_part, "output_size": (512, 512)
        },
        "left": {
            "cam_front": (0.5, 0.0, 0.0), "zoom": 0.35, "look_at_y": 12, "walk": False,
            "core_display": full_part, "decor_display": full_part, "output_size": (512, 512)
        },
        "right": {
            "cam_front": (-0.5, 0.0, 0.0), "zoom": 0.35, "look_at_y": 12, "walk": False,
            "core_display": full_part, "decor_display": full_part, "output_size": (512, 512)
        },
        "top_front_45": {
            "cam_front": (0.0, 0.4, 0.5), "zoom": 0.28, "look_at_y": 12, "walk": False,
            "core_display": full_part, "decor_display": full_part, "output_size": (306, 512)
        },
        "top_back_45": {
            "cam_front": (0.0, 0.4, -0.5), "zoom": 0.28, "look_at_y": 12, "walk": False,
            "core_display": full_part, "decor_display": full_part, "output_size": (306, 512)
        },
        "first_layer_front": {
            "cam_front": (0.3, 0.4, 0.5), "zoom": 0.28, "look_at_y": 12, "walk": False,
            "core_display": full_part, "decor_display": [], "output_size": (306, 512)
        },
        "first_layer_back": {
            "cam_front": (-0.5, -0.4, -0.5), "zoom": 0.28, "look_at_y": 12, "walk": False,
            "core_display": full_part, "decor_display": [], "output_size": (306, 512)
        },
        "left_front": {
            "cam_front": (0.3, 0.4, 0.5), "zoom": 0.22, "look_at_y": 16, "walk": False,
            "core_display": full_part, "decor_display": full_part, "output_size": (306, 512)
        },
        "left_back": {
            "cam_front": (-0.5, -0.4, -0.5), "zoom": 0.22, "look_at_y": 16, "walk": False,
            "core_display": full_part, "decor_display": full_part, "output_size": (306, 512)
        },
        "right_front": {
            "cam_front": (0.3, -0.4, 0.5), "zoom": 0.22, "look_at_y": 16, "walk": False,
            "core_display": full_part, "decor_display": full_part, "output_size": (306, 512)
        },
        "right_back": {
            "cam_front": (-0.5, 0.4, -0.5), "zoom": 0.22, "look_at_y": 16, "walk": False,
            "core_display": full_part, "decor_display": full_part, "output_size": (306, 512)
        },
        "head_front_right": {
            "cam_front": (-0.3, -0.4, 0.5), "zoom": 0.09, "look_at_y": 28, "walk": False,
            "core_display": ['head'], "decor_display": ['head'], "output_size": (306, 192)
        },
        "head_front_left": {
            "cam_front": (0.3, -0.4, 0.5), "zoom": 0.09, "look_at_y": 28, "walk": False,
            "core_display": ['head'], "decor_display": ['head'], "output_size": (306, 192)
        },
        "head_back_left": {
            "cam_front": (-0.5, -0.4, -0.5), "zoom": 0.09, "look_at_y": 28, "walk": False,
            "core_display": ['head'], "decor_display": ['head'], "output_size": (306, 192)
        },
        "head_back_right": {
            "cam_front": (0.5, -0.4, -0.5), "zoom": 0.09, "look_at_y": 28, "walk": False,
            "core_display": ['head'], "decor_display": ['head'], "output_size": (306, 192)
        },
        "body_front": {
            "cam_front": (0.3, 0.4, 0.5), "zoom": 0.25, "look_at_y": 12, "walk": False,
            "core_display": ['body'], "decor_display": ['body'], "output_size": (384, 768)
        },
        "body_back": {
            "cam_front": (-0.3, 0.4, 0.5), "zoom": 0.25, "look_at_y": 12, "walk": False,
            "core_display": ['body'], "decor_display": ['body'], "output_size": (384, 768)
        },
        "hat_front": {
            "cam_front": (0.3, 0.3, 0), "zoom": 0.25, "look_at_y": 22, "walk": False,
            "core_display": [], "decor_display": ['head'], "output_size": (306, 340)
        },
        "hat_back": {
            "cam_front": (-0.3, 0.3, 0), "zoom": 0.25, "look_at_y": 22, "walk": False,
            "core_display": [], "decor_display": ['head'], "output_size": (306, 340)
        },
    }
    
    # Default walk rotation angles matching build_target_img.py
    walk_rot = {
        'rot_head': (0,0,0),
        'rot_arm_right': (-30,0,0),
        'rot_arm_left': (30,0,0),
        'rot_leg_right': (30,0,0),
        'rot_leg_left': (-30,0,0),
    }
    
    # Default static rotation angles
    static_rot = {
        'rot_head': (0,0,0),
        'rot_arm_right': (0,0,0),
        'rot_arm_left': (0,0,0),
        'rot_leg_right': (0,0,0),
        'rot_leg_left': (0,0,0),
    }
    
    print(f"Running rendering comparison for all {len(views)} views...")
    for view_name, params in views.items():
        print(f"\n--- Testing View: {view_name} ---")
        rot_args = walk_rot if params["walk"] else static_rot
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
        # Convert both to float for MSE calculation
        pv_float = pv_render.astype(np.float32) / 255.0
        pt_float = pt_render_np.astype(np.float32) / 255.0
        
        # Compute MSE
        mse = np.mean((pv_float - pt_float) ** 2)
        print(f"MSE between PyVista and PyTorch for {view_name}: {mse:.6f}")
        
        # Check if they are virtually identical
        # (A tiny threshold is fine due to PyTorch grid_sample bilinear vs VTK cell coloring subpixel interpolation)
        if mse < 0.005:
            print(f"PASS: PyTorch renderer matches PyVista for {view_name}.")
        else:
            print(f"FAIL: PyTorch renderer deviates from PyVista for {view_name}.")
            
if __name__ == "__main__":
    run_tests()
