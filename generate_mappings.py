import os
import sys
import numpy as np
import torch
from PIL import Image

from mc_skin_utils import mc_render

_UV_PALETTE = None
_UV_PALETTE_FLAT = None
_UV_PALETTE_UVS = None
_UV_PALETTE_TREE = None

_MESH_ENTRIES = (
    ("core", "head", 0),
    ("core", "body", 1),
    ("core", "left_arm", 2),
    ("core", "right_arm", 3),
    ("core", "left_leg", 4),
    ("core", "right_leg", 5),
    ("decor", "head", 6),
    ("decor", "body", 7),
    ("decor", "left_arm", 8),
    ("decor", "right_arm", 9),
    ("decor", "left_leg", 10),
    ("decor", "right_leg", 11),
)


def _get_uv_palette():
    """Return a stable, high-contrast RGB code for each 64x64 UV texel."""
    global _UV_PALETTE, _UV_PALETTE_FLAT, _UV_PALETTE_UVS
    if _UV_PALETTE is None:
        idx = np.arange(64 * 64, dtype=np.uint16)
        # Odd multiplier makes this a permutation of the 12-bit color cube.
        # Neighboring UV texels land far apart in RGB, which is much more robust
        # at mesh/texel boundaries than directly encoding u and v as 0..63.
        code = ((idx * 2053 + 1381) & 0x0FFF).astype(np.uint16)
        colors = np.stack(
            [
                ((code >> 8) & 0x0F) * 17,
                ((code >> 4) & 0x0F) * 17,
                (code & 0x0F) * 17,
            ],
            axis=1,
        ).astype(np.uint8)
        uvs = np.stack([idx % 64, idx // 64], axis=1).astype(np.int16)
        _UV_PALETTE = colors.reshape(64, 64, 3)
        _UV_PALETTE_FLAT = colors.astype(np.float32)
        _UV_PALETTE_UVS = uvs
    return _UV_PALETTE, _UV_PALETTE_FLAT, _UV_PALETTE_UVS


def _nearest_palette_indices(rgb):
    global _UV_PALETTE_TREE
    _, palette_flat, _ = _get_uv_palette()
    try:
        from scipy.spatial import cKDTree

        if _UV_PALETTE_TREE is None:
            _UV_PALETTE_TREE = cKDTree(palette_flat)
        return _UV_PALETTE_TREE.query(rgb.astype(np.float32), workers=-1)[1]
    except Exception:
        nearest = np.empty(rgb.shape[0], dtype=np.int64)
        palette_norm = np.sum(palette_flat * palette_flat, axis=1)
        chunk_size = 8192
        rgb = rgb.astype(np.float32)
        for start in range(0, rgb.shape[0], chunk_size):
            chunk = rgb[start : start + chunk_size]
            distances = (
                np.sum(chunk * chunk, axis=1, keepdims=True)
                + palette_norm[None, :]
                - 2.0 * chunk @ palette_flat.T
            )
            nearest[start : start + chunk.shape[0]] = np.argmin(distances, axis=1)
        return nearest


def _decode_coordinate_render(rendered):
    _, _, palette_uvs = _get_uv_palette()
    valid = rendered[..., 3] > 0
    uv = np.zeros((*rendered.shape[:2], 2), dtype=np.int16) - 1
    if np.any(valid):
        nearest = _nearest_palette_indices(rendered[..., :3][valid])
        uv[valid] = palette_uvs[nearest]
    return valid, uv


def _extract_uv_mapping(rendered, output_shape):
    H, W = output_shape
    uv_map = np.zeros((H, W, 2), dtype=np.float32) - 1.0
    mask = np.zeros((H, W), dtype=np.float32)
    visible_uv = np.empty((0, 2), dtype=np.int16)
    if rendered is not None:
        valid, uv = _decode_coordinate_render(rendered)
        uv_map[valid, 0] = uv[valid, 0]
        uv_map[valid, 1] = uv[valid, 1]
        mask[valid] = 1.0
        visible_uv = np.unique(uv[valid], axis=0)
        visible_uv = visible_uv[
            (visible_uv[:, 0] >= 0)
            & (visible_uv[:, 0] < 64)
            & (visible_uv[:, 1] >= 0)
            & (visible_uv[:, 1] < 64)
        ]
    return uv_map, mask, visible_uv


def _new_plotter(params, output_size):
    import pyvista as pv

    plotter = pv.Plotter(off_screen=True, window_size=output_size)
    plotter.background_color = [1, 1, 1]
    plotter.camera.position = (
        params["cam_front"][0] * 70,
        params["cam_front"][1] * 70 + 20,
        params["cam_front"][2] * 70,
    )
    plotter.camera.focal_point = (0, params["look_at_y"], 0)
    plotter.camera.up = (0, 1, 0)
    plotter.camera.zoom(.14 / params["zoom"])
    if params.get("ortho", False):
        plotter.enable_parallel_projection()
    plotter.disable_anti_aliasing()
    return plotter


def _render_mesh_mapping(mesh, params, output_size):
    plotter = _new_plotter(params, output_size)
    plotter.add_mesh(
        mesh,
        scalars="RGBA",
        rgb=True,
        show_scalar_bar=False,
        smooth_shading=False,
        lighting=False,
        show_edges=False,
    )
    rendered = plotter.screenshot(None, transparent_background=True)
    depth = plotter.get_image_depth(fill_value=-np.inf).astype(np.float32)
    plotter.close()
    plotter.deep_clean()
    return rendered, depth


def _iter_display_meshes(meshes, core_display, decor_display):
    core_display = set(core_display)
    decor_display = set(decor_display)
    for layer_type, part_name, mesh_index in _MESH_ENTRIES:
        if layer_type == "core" and part_name not in core_display:
            continue
        if layer_type == "decor" and part_name not in decor_display:
            continue
        mesh = meshes[mesh_index]
        if mesh is not None:
            yield mesh

def generate_coordinate_skins():
    # Load masks
    mask_path = os.path.join(os.path.dirname(__file__), "skin-mask.png")
    decor_mask_path = os.path.join(os.path.dirname(__file__), "skin-decor-mask.png")
    
    if not os.path.exists(mask_path) or not os.path.exists(decor_mask_path):
        raise FileNotFoundError("Skin mask files not found in local directory. Please ensure you are running in the correct workspace.")
        
    skin_mask = np.array(Image.open(mask_path).convert("RGBA"))
    skin_decor_mask = np.array(Image.open(decor_mask_path).convert("RGBA"))
    
    uv_palette, _, _ = _get_uv_palette()

    # Create coordinate encoding skin for inner layer
    inner_skin = np.zeros((64, 64, 4), dtype=np.uint8)
    for y in range(64):
        for x in range(64):
            if skin_mask[y, x, 3] > 0:
                inner_skin[y, x, :3] = uv_palette[y, x]
                inner_skin[y, x, 3] = 255
                
    # Create coordinate encoding skin for outer layer
    outer_skin = np.zeros((64, 64, 4), dtype=np.uint8)
    for y in range(64):
        for x in range(64):
            if skin_decor_mask[y, x, 3] > 0:
                outer_skin[y, x, :3] = uv_palette[y, x]
                outer_skin[y, x, 3] = 255
                
    return inner_skin, outer_skin, skin_decor_mask

from config import sizes as config_sizes, get_views, walk_rot, static_rot, walk_offset, static_offset, parse_sizes

def render_and_save_mappings(selected_views=None, outer_layers=1, composite_layers=6, target_sizes=None):
    if target_sizes is None:
        target_sizes = config_sizes
    elif isinstance(target_sizes, str):
        target_sizes = parse_sizes(target_sizes)

    inner_skin, outer_skin, skin_decor_mask = generate_coordinate_skins()
    composite_skin = np.maximum(inner_skin, outer_skin)
    
    for size_tuple in target_sizes:
        W, H = size_tuple
        mappings_dir_name = f"mappings_{W}x{H}"
        mappings_dir = os.path.join(os.path.dirname(__file__), mappings_dir_name)
        os.makedirs(mappings_dir, exist_ok=True)

        # Sync to default 'mappings' directory if size is (256, 512) or the first size
        default_mappings_dir = os.path.join(os.path.dirname(__file__), "mappings")
        os.makedirs(default_mappings_dir, exist_ok=True)

        current_views = get_views((W, H))

        if selected_views is not None:
            if isinstance(selected_views, str):
                views_list = [view.strip() for view in selected_views.split(",") if view.strip()]
            else:
                views_list = list(selected_views)
            missing_views = [view for view in views_list if view not in current_views]
            if missing_views:
                raise ValueError(f"Unknown views {missing_views}. Available views: {', '.join(current_views)}")
            views_set = set(views_list)
        else:
            views_set = None

        print(f"\n==========================================")
        print(f"Generating mappings for size: W={W}, H={H} -> {mappings_dir_name}/")
        print(f"==========================================")

        for view_name, params in current_views.items():
            if views_set is not None and view_name not in views_set:
                continue
            print(f"Generating mapping for view: {view_name} ({W}x{H})...")
            rot_args = walk_rot if params["walk"] else static_rot
            offset_args = walk_offset if params["walk"] else static_offset
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
                offset_args=offset_args,
                ortho=params.get("ortho", False),
                off_screen=True
            )
            
            # Extract inner mapping
            inner_uv_map, inner_mask, _ = _extract_uv_mapping(inner_rendered, (H, W))
            
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
                    offset_args=offset_args,
                    ortho=params.get("ortho", False),
                    off_screen=True
                )

                outer_uv_map, outer_mask, visible_uv = _extract_uv_mapping(outer_rendered, (H, W))
                if visible_uv.size:
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
                    offset_args=offset_args,
                    ortho=params.get("ortho", False),
                    off_screen=True
                )

                composite_uv_map, composite_mask, visible_uv = _extract_uv_mapping(composite_rendered, (H, W))
                if visible_uv.size:
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

            geometry_uv_layers = []
            geometry_mask_layers = []
            geometry_depth_layers = []
            geometry_meshes = mc_render.build_minecraft_model(
                composite_skin,
                core_display=params["core_display"],
                decor_display=params["decor_display"],
                use_voxels=False,
                offset_args=offset_args,
                **rot_args,
            )
            for geometry_mesh in _iter_display_meshes(
                geometry_meshes,
                params["core_display"],
                params["decor_display"],
            ):
                geometry_rendered, geometry_depth = _render_mesh_mapping(
                    geometry_mesh,
                    params,
                    output_size,
                )
                geometry_uv_map, geometry_mask, _ = _extract_uv_mapping(geometry_rendered, (H, W))
                geometry_uv_layers.append(geometry_uv_map)
                geometry_mask_layers.append(geometry_mask)
                geometry_depth_layers.append(np.where(geometry_mask > 0, geometry_depth, -np.inf))

            if geometry_uv_layers:
                geometry_uv_layers = np.stack(geometry_uv_layers, axis=0)
                geometry_mask_layers = np.stack(geometry_mask_layers, axis=0)
                geometry_depth_layers = np.stack(geometry_depth_layers, axis=0).astype(np.float32)
                geometry_sort_indices = np.argsort(geometry_depth_layers, axis=0).astype(np.int16)
            else:
                geometry_uv_layers = np.zeros((0, H, W, 2), dtype=np.float32) - 1.0
                geometry_mask_layers = np.zeros((0, H, W), dtype=np.float32)
                geometry_sort_indices = np.zeros((0, H, W), dtype=np.int16)
            print(f"  geometry layers: {geometry_uv_layers.shape[0]}")
            
            # Convert to PyTorch tensors and save
            mapping_data = {
                "inner_uv_map": torch.tensor(inner_uv_map),
                "inner_mask": torch.tensor(inner_mask),
                "outer_uv_map": torch.tensor(outer_uv_layers[0]),
                "outer_mask": torch.tensor(outer_mask_layers[0]),
                "composite_uv_layers": torch.tensor(composite_uv_layers.astype(np.int16)),
                "composite_masks": torch.tensor(composite_mask_layers.astype(np.bool_)),
                "composite_is_decor_layers": torch.tensor(composite_is_decor_layers),
                "geometry_uv_layers": torch.tensor(geometry_uv_layers.astype(np.int16)),
                "geometry_masks": torch.tensor(geometry_mask_layers.astype(np.bool_)),
                "geometry_sort_indices": torch.tensor(geometry_sort_indices),
            }
            if outer_layers > 1:
                mapping_data["outer_uv_layers"] = torch.tensor(outer_uv_layers.astype(np.int16))
                mapping_data["outer_masks"] = torch.tensor(outer_mask_layers.astype(np.bool_))
            
            mapping_file_name = f"{view_name}_mapping.pt"
            torch.save(mapping_data, os.path.join(mappings_dir, mapping_file_name))
            print(f"Saved {mapping_file_name} in {mappings_dir_name} (W={W}, H={H})")

            if size_tuple == (256, 512) or size_tuple == target_sizes[0]:
                torch.save(mapping_data, os.path.join(default_mappings_dir, mapping_file_name))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate differentiable renderer UV mapping files.")
    parser.add_argument("--views", default=None, help="Optional comma-separated subset of view names to generate.")
    parser.add_argument("--sizes", default=None, help="Optional comma-separated sizes, e.g. 256x512,512x1024.")
    parser.add_argument("--outer_layers", type=int, default=1, help="Number of depth-peeled overlay-only UV layers.")
    parser.add_argument("--composite_layers", type=int, default=6, help="Number of depth-peeled combined UV layers.")
    args = parser.parse_args()
    render_and_save_mappings(
        selected_views=args.views,
        outer_layers=args.outer_layers,
        composite_layers=args.composite_layers,
        target_sizes=args.sizes,
    )
