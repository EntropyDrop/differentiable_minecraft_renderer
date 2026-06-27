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
                
    return inner_skin, outer_skin, skin_decor_mask

def render_and_save_mappings(selected_views=None, outer_layers=1, composite_layers=6):
    inner_skin, outer_skin, skin_decor_mask = generate_coordinate_skins()
    composite_skin = np.maximum(inner_skin, outer_skin)
    
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
    
    mappings_dir = os.path.join(os.path.dirname(__file__), "mappings")
    os.makedirs(mappings_dir, exist_ok=True)

    if selected_views is not None:
        selected_views = [view.strip() for view in selected_views.split(",") if view.strip()]
        missing_views = [view for view in selected_views if view not in views]
        if missing_views:
            raise ValueError(f"Unknown views {missing_views}. Available views: {', '.join(views)}")
        selected_views = set(selected_views)
    
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
        if selected_views is not None and view_name not in selected_views:
            continue
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
            ortho=params.get("ortho", False),
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
        
        # Extract outer mapping layers by repeatedly peeling the visible overlay UV cells.
        outer_uv_layers = []
        outer_mask_layers = []
        remaining_outer_skin = outer_skin.copy()
        for layer_index in range(max(outer_layers, 1)):
            outer_rendered = mc_render.render_skin(
                skin=Image.fromarray(remaining_outer_skin),
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
                ortho=params.get("ortho", False),
                off_screen=True
            )

            outer_uv_map = np.zeros((H, W, 2), dtype=np.float32) - 1.0
            outer_mask = np.zeros((H, W), dtype=np.float32)
            if outer_rendered is not None:
                valid_outer = (outer_rendered[..., 2] > 200) & (outer_rendered[..., 3] > 0)
                outer_uv_map[valid_outer, 0] = outer_rendered[valid_outer, 0]
                outer_uv_map[valid_outer, 1] = outer_rendered[valid_outer, 1]
                outer_mask[valid_outer] = 1.0

                visible_uv = np.unique(outer_rendered[valid_outer, :2].astype(np.int16), axis=0)
                visible_uv = visible_uv[
                    (visible_uv[:, 0] >= 0)
                    & (visible_uv[:, 0] < 64)
                    & (visible_uv[:, 1] >= 0)
                    & (visible_uv[:, 1] < 64)
                ]
                remaining_outer_skin[visible_uv[:, 1], visible_uv[:, 0], 3] = 0

            outer_uv_layers.append(outer_uv_map)
            outer_mask_layers.append(outer_mask)
            print(f"  outer layer {layer_index}: {int(outer_mask.sum())} pixels")
            if outer_mask.sum() == 0:
                break

        while len(outer_uv_layers) < max(outer_layers, 1):
            outer_uv_layers.append(np.zeros((H, W, 2), dtype=np.float32) - 1.0)
            outer_mask_layers.append(np.zeros((H, W), dtype=np.float32))

        outer_uv_layers = np.stack(outer_uv_layers, axis=0)
        outer_mask_layers = np.stack(outer_mask_layers, axis=0)

        composite_uv_layers = []
        composite_mask_layers = []
        remaining_composite_skin = composite_skin.copy()
        for layer_index in range(max(composite_layers, 1)):
            composite_rendered = mc_render.render_skin(
                skin=Image.fromarray(remaining_composite_skin),
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

            composite_uv_map = np.zeros((H, W, 2), dtype=np.float32) - 1.0
            composite_mask = np.zeros((H, W), dtype=np.float32)
            if composite_rendered is not None:
                valid_composite = (composite_rendered[..., 2] > 200) & (composite_rendered[..., 3] > 0)
                composite_uv_map[valid_composite, 0] = composite_rendered[valid_composite, 0]
                composite_uv_map[valid_composite, 1] = composite_rendered[valid_composite, 1]
                composite_mask[valid_composite] = 1.0

                visible_uv = np.unique(composite_rendered[valid_composite, :2].astype(np.int16), axis=0)
                visible_uv = visible_uv[
                    (visible_uv[:, 0] >= 0)
                    & (visible_uv[:, 0] < 64)
                    & (visible_uv[:, 1] >= 0)
                    & (visible_uv[:, 1] < 64)
                ]
                remaining_composite_skin[visible_uv[:, 1], visible_uv[:, 0], 3] = 0

            composite_uv_layers.append(composite_uv_map)
            composite_mask_layers.append(composite_mask)
            print(f"  composite layer {layer_index}: {int(composite_mask.sum())} pixels")
            if composite_mask.sum() == 0:
                break

        while len(composite_uv_layers) < max(composite_layers, 1):
            composite_uv_layers.append(np.zeros((H, W, 2), dtype=np.float32) - 1.0)
            composite_mask_layers.append(np.zeros((H, W), dtype=np.float32))

        composite_uv_layers = np.stack(composite_uv_layers, axis=0)
        composite_mask_layers = np.stack(composite_mask_layers, axis=0)
        composite_is_decor_layers = np.zeros_like(composite_mask_layers, dtype=np.bool_)
        decor_mask = skin_decor_mask[:, :, 3] > 0
        for layer_index in range(composite_uv_layers.shape[0]):
            layer_mask = composite_mask_layers[layer_index] > 0
            u = np.clip(composite_uv_layers[layer_index, ..., 0].astype(np.int16), 0, 63)
            v = np.clip(composite_uv_layers[layer_index, ..., 1].astype(np.int16), 0, 63)
            composite_is_decor_layers[layer_index] = layer_mask & decor_mask[v, u]
        
        # Convert to PyTorch tensors and save
        mapping_data = {
            "inner_uv_map": torch.tensor(inner_uv_map),
            "inner_mask": torch.tensor(inner_mask),
            "outer_uv_map": torch.tensor(outer_uv_layers[0]),
            "outer_mask": torch.tensor(outer_mask_layers[0]),
            "composite_uv_layers": torch.tensor(composite_uv_layers.astype(np.int16)),
            "composite_masks": torch.tensor(composite_mask_layers.astype(np.bool_)),
            "composite_is_decor_layers": torch.tensor(composite_is_decor_layers),
        }
        if outer_layers > 1:
            mapping_data["outer_uv_layers"] = torch.tensor(outer_uv_layers.astype(np.int16))
            mapping_data["outer_masks"] = torch.tensor(outer_mask_layers.astype(np.bool_))
        
        torch.save(mapping_data, os.path.join(mappings_dir, f"{view_name}_mapping.pt"))
        print(f"Saved {view_name}_mapping.pt (W={W}, H={H})")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate differentiable renderer UV mapping files.")
    parser.add_argument("--views", default=None, help="Optional comma-separated subset of view names to generate.")
    parser.add_argument("--outer_layers", type=int, default=1, help="Number of depth-peeled overlay-only UV layers.")
    parser.add_argument("--composite_layers", type=int, default=6, help="Number of depth-peeled combined UV layers.")
    args = parser.parse_args()
    render_and_save_mappings(
        selected_views=args.views,
        outer_layers=args.outer_layers,
        composite_layers=args.composite_layers,
    )
