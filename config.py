# Shared configurations for differentiable_minecraft_renderer
import os
import re

full_part = ['head', 'body', 'left_arm', 'right_arm', 'left_leg', 'right_leg']

# 0.28
zoom = 0.23

def parse_size_tuple(s):
    s = s.strip().strip("()[]")
    for sep in ["x", "X", ","]:
        if sep in s:
            parts = [p.strip() for p in s.split(sep) if p.strip()]
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                return (int(parts[0]), int(parts[1]))
    parts = s.split()
    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
        return (int(parts[0]), int(parts[1]))
    raise ValueError(f"Invalid size specification: '{s}'")

def parse_sizes(env_val):
    if not env_val:
        return [(256, 512),(512, 1024), (1024, 2048)]
        #return [(344, 768),(344*2, 768*2)]
    
    pattern = r'\(?\s*(\d+)\s*[\times,X\s]\s*(\d+)\s*\)?'
    matches = re.findall(pattern, env_val)
    if matches:
        res = [(int(w), int(h)) for w, h in matches]
        if len(res) > 0:
            return res

    clean_val = env_val.replace(";", ",")
    items = [item.strip() for item in clean_val.split(",") if item.strip()]
    parsed = []
    for item in items:
        try:
            parsed.append(parse_size_tuple(item))
        except ValueError:
            pass
    if parsed:
        return parsed
        
    return [(256, 512), (512, 1024), (1024, 2048)]

env_size_str = (
    os.environ.get("RENDERER_SIZES")
    or os.environ.get("RENDERER_SIZE")
    or os.environ.get("RENDER_SIZE")
    or os.environ.get("SIZE")
)
sizes = parse_sizes(env_size_str)
size = sizes[0]

def create_views(output_size):
    return {
        "front_left": {
            "cam_front": (-0.3, 0.1, 0.5), "zoom": zoom, "look_at_y": 16, "walk": True,
            "core_display": full_part, "decor_display": full_part, "output_size": output_size, "ortho": True
        },
        "back_left": {
            "cam_front": (0.3, 0.1, -0.5), "zoom": zoom, "look_at_y": 16, "walk": True,
            "core_display": full_part, "decor_display": full_part, "output_size": output_size, "ortho": True
        },
        "front_right": {
            "cam_front": (0.3, 0.1, 0.5), "zoom": zoom, "look_at_y": 16, "walk": True,
            "core_display": full_part, "decor_display": full_part, "output_size": output_size, "ortho": True
        },
        "back_right": {
            "cam_front": (-0.3, 0.1, -0.5), "zoom": zoom, "look_at_y": 16, "walk": True,
            "core_display": full_part, "decor_display": full_part, "output_size": output_size, "ortho": True
        },
        "front_left_overlay": {
            "cam_front": (-0.3, 0.1, 0.5), "zoom": zoom, "look_at_y": 16, "walk": True,
            "core_display": [], "decor_display": full_part, "output_size": output_size, "ortho": True
        },
        "back_left_overlay": {
            "cam_front": (0.3, 0.1, -0.5), "zoom": zoom, "look_at_y": 16, "walk": True,
            "core_display": [], "decor_display": full_part, "output_size": output_size, "ortho": True
        },
        "front_right_overlay": {
            "cam_front": (0.3, 0.1, 0.5), "zoom": zoom, "look_at_y": 16, "walk": True,
            "core_display": [], "decor_display": full_part, "output_size": output_size, "ortho": True
        },
        "back_right_overlay": {
            "cam_front": (-0.3, 0.1, -0.5), "zoom": zoom, "look_at_y": 16, "walk": True,
            "core_display": [], "decor_display": full_part, "output_size": output_size, "ortho": True
        },

        #"base_front2": {
        #    "cam_front": (0.3, 0.1, 0.5), "zoom": zoom, "look_at_y": 16, "walk": True,
        #    "core_display": full_part, "decor_display": [], "output_size": output_size, "ortho": True
        #},
        #"base_back2": {
        #    "cam_front": (-0.3, 0.1, -0.5), "zoom": zoom, "look_at_y": 16, "walk": True,
        #    "core_display": full_part, "decor_display": [], "output_size": output_size, "ortho": True
        #},
        #"base_front1": {
        #    "cam_front": (-0.3, 0.1, 0.5), "zoom": zoom, "look_at_y": 16, "walk": True,
        #    "core_display": full_part, "decor_display": [], "output_size": output_size, "ortho": True
        #},
        #"base_back1": {
        #    "cam_front": (0.3, 0.1, -0.5), "zoom": zoom, "look_at_y": 16, "walk": True,
        #    "core_display": full_part, "decor_display": [], "output_size": output_size, "ortho": True
        #},
        #"static_front": {
        #    "cam_front": (-0.3, 0.1, 0.5), "zoom": zoom, "look_at_y": 16, "walk": False,
        #    "core_display": full_part, "decor_display": full_part, "output_size": output_size, "ortho": True
        #},
        #"static_back": {
        #    "cam_front": (0.3, 0.1, -0.5), "zoom": zoom, "look_at_y": 16, "walk": False,
        #    "core_display": full_part, "decor_display": full_part, "output_size": output_size, "ortho": True
        #},
    }

def get_views(output_size=None):
    if output_size is None:
        output_size = size
    return create_views(output_size)

views = create_views(size)

# Default walk rotation angles matching build_target_img.py
walk_rot = {
    'rot_head': (0,0,0),
    'rot_arm_right': (-10,0,0),
    'rot_arm_left': (10,0,0),
    'rot_leg_right': (10,0,0),
    'rot_leg_left': (-10,0,0),
}

# Default static rotation angles
static_rot = {
    'rot_head': (0,0,0),
    'rot_arm_right': (0,0,0),
    'rot_arm_left': (0,0,0),
    'rot_leg_right': (0,0,0),
    'rot_leg_left': (0,0,0),
}

# Default walk offset configurations
walk_offset = {
    'head': (0,0,0),
    'body': (0,0,0),
    'right_arm': (0,0,0),
    'left_arm': (0,0,0),
    'right_leg': (0,0,0),
    'left_leg': (0,0,0),
}

# Default static offset configurations
static_offset = {
    'head': (0,0,0),
    'body': (0,0,0),
    'right_arm': (0,0,0),
    'left_arm': (0,0,0),
    'right_leg': (0,0,0),
    'left_leg': (0,0,0),
}

