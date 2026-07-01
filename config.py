# Shared configurations for differentiable_minecraft_renderer

full_part = ['head', 'body', 'left_arm', 'right_arm', 'left_leg', 'right_leg']

views = {
    "walk_front_both_layer_ortho": {
        "cam_front": (0.3, 0.1, 0.5), "zoom": 0.28, "look_at_y": 16, "walk": True,
        "core_display": full_part, "decor_display": full_part, "output_size": (512, 1024), "ortho": True
    },
    "walk_back_both_layer_ortho": {
        "cam_front": (-0.3, 0.1, -0.5), "zoom": 0.28, "look_at_y": 16, "walk": True,
        "core_display": full_part, "decor_display": full_part, "output_size": (512, 1024), "ortho": True
    },
    "walk_front_base_layer_ortho": {
        "cam_front": (0.3, 0.1, 0.5), "zoom": 0.28, "look_at_y": 16, "walk": True,
        "core_display": full_part, "decor_display": [], "output_size": (512, 1024), "ortho": True
    },
    "walk_back_base_layer_ortho": {
        "cam_front": (-0.3, 0.1, -0.5), "zoom": 0.28, "look_at_y": 16, "walk": True,
        "core_display": full_part, "decor_display": [], "output_size": (512, 1024), "ortho": True
    },
    "static_front": {
        "cam_front": (0.3, 0.1, 0.5), "zoom": 0.28, "look_at_y": 16, "walk": False,
        "core_display": full_part, "decor_display": full_part, "output_size": (512, 1024), "ortho": True
    },
    "static_back": {
        "cam_front": (-0.3, 0.1, -0.5), "zoom": 0.28, "look_at_y": 16, "walk": False,
        "core_display": full_part, "decor_display": full_part, "output_size": (512, 1024), "ortho": True
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

# Default walk offset configurations
walk_offset = {
    'head': (0,0,0),
    'body': (0,0,0),
    'right_arm': (0,0,-2),
    'left_arm': (0,0,2),
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

