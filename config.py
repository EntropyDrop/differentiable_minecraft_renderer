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

