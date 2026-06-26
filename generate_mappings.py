import os
import sys
import numpy as np
import torch
from PIL import Image

from mc_skin_utils import mc_render

def generate_coordinate_skins():
    # Load masks
    mask_path = os.path.join(os.path.dirname(__file__), "skin-mask.png")
    decor_mask_path = os.path.join(os.path.dirname(__file__), "skin-decor-mask.png")
    
    if not os.path.exists(mask_path) or not os.path.exists(decor_mask_path):
        raise FileNotFoundError("Skin mask files not found in local directory. Please ensure you are running in the correct workspace.")
        
    skin_mask = np.array(Image.open(mask_path).convert("RGBA"))
    skin_decor_mask = np.array(Image.open(decor_mask_path).convert("RGBA"))
    
    # Create coordinate encoding skin for inner layer
    inner_skin = np.zeros((64, 64, 4), dtype=np.uint8)
    for y in range(64):
        for x in range(64):
            if skin_mask[y, x, 3] > 0:
                # Store coordinates: Red = U (x), Green = V (y), Blue = 255 (valid flag), Alpha = 255
                inner_skin[y, x] = [x, y, 255, 255]
                
    # Create coordinate encoding skin for outer layer
    outer_skin = np.zeros((64, 64, 4), dtype=np.uint8)
    for y in range(64):
        for x in range(64):
            if skin_decor_mask[y, x, 3] > 0:
                outer_skin[y, x] = [x, y, 255, 255]
                
    return inner_skin, outer_skin

def render_and_save_mappings():
    inner_skin, outer_skin = generate_coordinate_skins()
    
    full_part = ['head', 'body', 'left_arm', 'right_arm', 'left_leg', 'right_leg']
    
    # Define all renderer views from build_target_img.py
    views = {
        "walk_perspective": {
            "cam_front": (0.3, 0.4, 0.5), "zoom": 0.22, "look_at_y": 12, "walk": True,
            "core_display": full_part, "decor_display": full_part, "output_size": (768, 920)
        },
        "walk_perspective_back": {
            "cam_front": (-0.3, 0.4, -0.5), "zoom": 0.22, "look_at_y": 12, "walk": True,
            "core_display": full_part, "decor_display": full_part, "output_size": (768, 920)
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
    
    mappings_dir = os.path.join(os.path.dirname(__file__), "mappings")
    os.makedirs(mappings_dir, exist_ok=True)
    
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
    
    for view_name, params in views.items():
        print(f"Generating mapping for view: {view_name}...")
        rot_args = walk_rot if params["walk"] else static_rot
        output_size = params["output_size"]
        
        # Render inner layer
        inner_rendered = mc_render.render_skin(
            skin=Image.fromarray(inner_skin),
            output_size=output_size,
            cam_front=params["cam_front"],
            zoom=params["zoom"],
            look_at_y=params["look_at_y"],
            use_voxels=False,
            light=False,
            transparent_background=True,
            core_display=params["core_display"],
            decor_display=[], # We render core only for inner mapping
            rot_args=rot_args,
            off_screen=True
        )
        
        # Render outer layer
        outer_rendered = mc_render.render_skin(
            skin=Image.fromarray(outer_skin),
            output_size=output_size,
            cam_front=params["cam_front"],
            zoom=params["zoom"],
            look_at_y=params["look_at_y"],
            use_voxels=False,
            light=False,
            transparent_background=True,
            core_display=[], # We render overlay only for outer mapping
            decor_display=params["decor_display"],
            rot_args=rot_args,
            off_screen=True
        )
        
        W, H = output_size
        
        # Extract inner mapping
        inner_uv_map = np.zeros((H, W, 2), dtype=np.float32) - 1.0
        inner_mask = np.zeros((H, W), dtype=np.float32)
        
        if inner_rendered is not None:
            valid_inner = (inner_rendered[..., 2] > 200) & (inner_rendered[..., 3] > 0)
            inner_uv_map[valid_inner, 0] = inner_rendered[valid_inner, 0]
            inner_uv_map[valid_inner, 1] = inner_rendered[valid_inner, 1]
            inner_mask[valid_inner] = 1.0
        
        # Extract outer mapping
        outer_uv_map = np.zeros((H, W, 2), dtype=np.float32) - 1.0
        outer_mask = np.zeros((H, W), dtype=np.float32)
        
        if outer_rendered is not None:
            valid_outer = (outer_rendered[..., 2] > 200) & (outer_rendered[..., 3] > 0)
            outer_uv_map[valid_outer, 0] = outer_rendered[valid_outer, 0]
            outer_uv_map[valid_outer, 1] = outer_rendered[valid_outer, 1]
            outer_mask[valid_outer] = 1.0
        
        # Convert to PyTorch tensors and save
        mapping_data = {
            "inner_uv_map": torch.tensor(inner_uv_map),
            "inner_mask": torch.tensor(inner_mask),
            "outer_uv_map": torch.tensor(outer_uv_map),
            "outer_mask": torch.tensor(outer_mask)
        }
        
        torch.save(mapping_data, os.path.join(mappings_dir, f"{view_name}_mapping.pt"))
        print(f"Saved {view_name}_mapping.pt (W={W}, H={H})")

if __name__ == "__main__":
    render_and_save_mappings()
