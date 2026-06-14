#!/usr/bin/env python3
import os
import sys
import argparse
from PIL import Image

# Import numpy and torch if available, keep fallback
try:
    import numpy as np
except ImportError:
    np = None

try:
    import torch
except ImportError:
    torch = None

# Default dimensions for each orthographic walk view
DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 1024

def merge_2_pil(front_img: Image.Image, back_img: Image.Image, direction: str = "horizontal") -> Image.Image:
    w1, h1 = front_img.size
    w2, h2 = back_img.size
    mode = "RGBA" if (front_img.mode == "RGBA" or back_img.mode == "RGBA") else "RGB"
    
    if direction == "horizontal":
        out_w = w1 + w2
        out_h = max(h1, h2)
        merged = Image.new(mode, (out_w, out_h))
        merged.paste(front_img.convert(mode), (0, 0))
        merged.paste(back_img.convert(mode), (w1, 0))
    elif direction == "vertical":
        out_w = max(w1, w2)
        out_h = h1 + h2
        merged = Image.new(mode, (out_w, out_h))
        merged.paste(front_img.convert(mode), (0, 0))
        merged.paste(back_img.convert(mode), (0, h1))
    else:
        raise ValueError(f"Invalid direction: {direction}. Must be 'horizontal' or 'vertical'.")
    return merged

def split_2_pil(merged_img: Image.Image, direction: str = "auto") -> tuple[Image.Image, Image.Image]:
    w, h = merged_img.size
    if direction == "auto":
        if (w / h) > (DEFAULT_WIDTH / DEFAULT_HEIGHT):
            direction = "horizontal"
        else:
            direction = "vertical"
            
    if direction == "horizontal":
        split_w = w // 2
        front_img = merged_img.crop((0, 0, split_w, h))
        back_img = merged_img.crop((split_w, 0, w, h))
    elif direction == "vertical":
        split_h = h // 2
        front_img = merged_img.crop((0, 0, w, split_h))
        back_img = merged_img.crop((0, split_h, w, h))
    else:
        raise ValueError(f"Invalid direction: {direction}. Must be 'auto', 'horizontal', or 'vertical'.")
    return front_img, back_img

def merge_2_numpy(front_arr, back_arr, direction: str = "horizontal"):
    if np is None:
        raise RuntimeError("numpy is not installed or could not be imported.")
    ndim = front_arr.ndim
    if ndim != back_arr.ndim:
        raise ValueError(f"NumPy arrays must have the same number of dimensions. Got {ndim} and {back_arr.ndim}")
    if direction == "horizontal":
        axis = 2 if ndim == 4 else 1
        return np.concatenate([front_arr, back_arr], axis=axis)
    elif direction == "vertical":
        axis = 1 if ndim == 4 else 0
        return np.concatenate([front_arr, back_arr], axis=axis)
    else:
        raise ValueError(f"Invalid direction: {direction}. Must be 'horizontal' or 'vertical'.")

def split_2_numpy(merged_arr, direction: str = "auto"):
    if np is None:
        raise RuntimeError("numpy is not installed or could not be imported.")
    ndim = merged_arr.ndim
    if direction == "auto":
        if ndim == 4:
            _, h, w, _ = merged_arr.shape
        else:
            h, w = merged_arr.shape[:2]
        if (w / h) > (DEFAULT_WIDTH / DEFAULT_HEIGHT):
            direction = "horizontal"
        else:
            direction = "vertical"
    if direction == "horizontal":
        axis = 2 if ndim == 4 else 1
        return np.split(merged_arr, 2, axis=axis)
    elif direction == "vertical":
        axis = 1 if ndim == 4 else 0
        return np.split(merged_arr, 2, axis=axis)
    else:
        raise ValueError(f"Invalid direction: {direction}. Must be 'auto', 'horizontal', or 'vertical'.")

def merge_2_tensor(front_tensor, back_tensor, direction: str = "horizontal"):
    if torch is None:
        raise RuntimeError("torch is not installed or could not be imported.")
    if direction == "horizontal":
        return torch.cat([front_tensor, back_tensor], dim=-1)
    elif direction == "vertical":
        return torch.cat([front_tensor, back_tensor], dim=-2)
    else:
        raise ValueError(f"Invalid direction: {direction}. Must be 'horizontal' or 'vertical'.")

def split_2_tensor(merged_tensor, direction: str = "auto"):
    if torch is None:
        raise RuntimeError("torch is not installed or could not be imported.")
    h, w = merged_tensor.shape[-2], merged_tensor.shape[-1]
    if direction == "auto":
        if (w / h) > (DEFAULT_WIDTH / DEFAULT_HEIGHT):
            direction = "horizontal"
        else:
            direction = "vertical"
    if direction == "horizontal":
        return torch.chunk(merged_tensor, 2, dim=-1)
    elif direction == "vertical":
        return torch.chunk(merged_tensor, 2, dim=-2)
    else:
        raise ValueError(f"Invalid direction: {direction}. Must be 'auto', 'horizontal', or 'vertical'.")

def merge_pil(
    walk_front_base: Image.Image,
    walk_back_base: Image.Image,
    walk_front_both: Image.Image,
    walk_back_both: Image.Image
) -> Image.Image:
    """Merges four PIL Images into a 2x2 grid.
    Layout:
      [walk_front_base] [walk_back_base]
      [walk_front_both] [walk_back_both]
    """
    w1, h1 = walk_front_base.size
    w2, h2 = walk_back_base.size
    w3, h3 = walk_front_both.size
    w4, h4 = walk_back_both.size
    
    w = max(w1, w2, w3, w4)
    h = max(h1, h2, h3, h4)
    mode = "RGBA" if any(img.mode == "RGBA" for img in [walk_front_base, walk_back_base, walk_front_both, walk_back_both]) else "RGB"
    
    merged = Image.new(mode, (2 * w, 2 * h))
    merged.paste(walk_front_base.convert(mode), (0, 0))
    merged.paste(walk_back_base.convert(mode), (w, 0))
    merged.paste(walk_front_both.convert(mode), (0, h))
    merged.paste(walk_back_both.convert(mode), (w, h))
    return merged

def split_pil(merged_img: Image.Image) -> tuple[Image.Image, Image.Image, Image.Image, Image.Image]:
    total_w, total_h = merged_img.size
    w = total_w // 2
    h = total_h // 2
    
    walk_front_base = merged_img.crop((0, 0, w, h))
    walk_back_base = merged_img.crop((w, 0, total_w, h))
    walk_front_both = merged_img.crop((0, h, w, total_h))
    walk_back_both = merged_img.crop((w, h, total_w, total_h))
    return walk_front_base, walk_back_base, walk_front_both, walk_back_both

def merge_numpy(walk_front_base, walk_back_base, walk_front_both, walk_back_both):
    if np is None:
        raise RuntimeError("numpy is not installed or could not be imported.")
    ndim = walk_front_base.ndim
    for arr in [walk_back_base, walk_front_both, walk_back_both]:
        if arr.ndim != ndim:
            raise ValueError("All NumPy arrays must have the same number of dimensions.")
            
    if ndim == 4:
        top = np.concatenate([walk_front_base, walk_back_base], axis=2)
        bottom = np.concatenate([walk_front_both, walk_back_both], axis=2)
        return np.concatenate([top, bottom], axis=1)
    else:
        top = np.concatenate([walk_front_base, walk_back_base], axis=1)
        bottom = np.concatenate([walk_front_both, walk_back_both], axis=1)
        return np.concatenate([top, bottom], axis=0)

def split_numpy(merged_arr):
    if np is None:
        raise RuntimeError("numpy is not installed or could not be imported.")
    ndim = merged_arr.ndim
    if ndim == 4:
        h_axis, w_axis = 1, 2
    else:
        h_axis, w_axis = 0, 1
        
    top, bottom = np.split(merged_arr, 2, axis=h_axis)
    walk_front_base, walk_back_base = np.split(top, 2, axis=w_axis)
    walk_front_both, walk_back_both = np.split(bottom, 2, axis=w_axis)
    return walk_front_base, walk_back_base, walk_front_both, walk_back_both

def merge_tensor(walk_front_base, walk_back_base, walk_front_both, walk_back_both):
    if torch is None:
        raise RuntimeError("torch is not installed or could not be imported.")
    top = torch.cat([walk_front_base, walk_back_base], dim=-1)
    bottom = torch.cat([walk_front_both, walk_back_both], dim=-1)
    return torch.cat([top, bottom], dim=-2)

def split_tensor(merged_tensor):
    if torch is None:
        raise RuntimeError("torch is not installed or could not be imported.")
    top, bottom = torch.chunk(merged_tensor, 2, dim=-2)
    walk_front_base, walk_back_base = torch.chunk(top, 2, dim=-1)
    walk_front_both, walk_back_both = torch.chunk(bottom, 2, dim=-1)
    return walk_front_base, walk_back_base, walk_front_both, walk_back_both

def merge(*args, **kwargs):
    images = list(args)
    direction = kwargs.get("direction", "horizontal")
    
    # Gather from keywords if they are used instead of positional
    for kw in ["front", "back", "walk_front_base", "walk_back_base", "walk_front_both", "walk_back_both"]:
        if kw in kwargs:
            images.append(kwargs[kw])
            
    if len(images) == 2:
        img1, img2 = images
        if isinstance(img1, Image.Image):
            return merge_2_pil(img1, img2, direction)
        elif np is not None and isinstance(img1, np.ndarray):
            return merge_2_numpy(img1, img2, direction)
        elif torch is not None and torch.is_tensor(img1):
            return merge_2_tensor(img1, img2, direction)
        raise TypeError(f"Unsupported types for merge: {type(img1)}")
        
    elif len(images) == 4:
        img1, img2, img3, img4 = images
        if isinstance(img1, Image.Image):
            return merge_pil(img1, img2, img3, img4)
        elif np is not None and isinstance(img1, np.ndarray):
            return merge_numpy(img1, img2, img3, img4)
        elif torch is not None and torch.is_tensor(img1):
            return merge_tensor(img1, img2, img3, img4)
        raise TypeError(f"Unsupported types for merge: {type(img1)}")
        
    else:
        raise ValueError(f"Invalid number of images for merge: expected 2 or 4, got {len(images)}")

def split(merged, *args, **kwargs):
    direction = kwargs.get("direction", "auto")
    
    if isinstance(merged, Image.Image):
        total_w, total_h = merged.size
    elif np is not None and isinstance(merged, np.ndarray):
        ndim = merged.ndim
        if ndim == 4:
            total_h, total_w = merged.shape[1:3]
        else:
            total_h, total_w = merged.shape[:2]
    elif torch is not None and torch.is_tensor(merged):
        total_h, total_w = merged.shape[-2], merged.shape[-1]
    else:
        raise TypeError(f"Unsupported type for split: {type(merged)}")
        
    # Check if aspect ratio is 4-view (1:2 ratio but size exceeds DEFAULT_WIDTH)
    is_4_view = False
    aspect_ratio = total_w / total_h
    if total_w >= 1.5 * DEFAULT_WIDTH and aspect_ratio <= 0.6:
        is_4_view = True
        
    if is_4_view:
        if isinstance(merged, Image.Image):
            return split_pil(merged)
        elif np is not None and isinstance(merged, np.ndarray):
            return split_numpy(merged)
        else:
            return split_tensor(merged)
    else:
        if isinstance(merged, Image.Image):
            return split_2_pil(merged, direction)
        elif np is not None and isinstance(merged, np.ndarray):
            return split_2_numpy(merged, direction)
        else:
            return split_2_tensor(merged, direction)

def run_self_tests():
    print("Running self-tests for merge_views.py...")
    
    img1 = Image.new("RGBA", (DEFAULT_WIDTH, DEFAULT_HEIGHT), color=(255, 0, 0, 255))
    img2 = Image.new("RGBA", (DEFAULT_WIDTH, DEFAULT_HEIGHT), color=(0, 255, 0, 255))
    img3 = Image.new("RGBA", (DEFAULT_WIDTH, DEFAULT_HEIGHT), color=(0, 0, 255, 255))
    img4 = Image.new("RGBA", (DEFAULT_WIDTH, DEFAULT_HEIGHT), color=(255, 255, 0, 255))
    
    # 1. 4-Image Merging/Splitting
    print("  Testing 4-image merging and splitting...")
    merged4 = merge(img1, img2, img3, img4)
    assert merged4.size == (DEFAULT_WIDTH * 2, DEFAULT_HEIGHT * 2), f"Size mismatch: {merged4.size}"
    s1, s2, s3, s4 = split(merged4)
    assert s1.size == (DEFAULT_WIDTH, DEFAULT_HEIGHT)
    assert s2.size == (DEFAULT_WIDTH, DEFAULT_HEIGHT)
    assert s3.size == (DEFAULT_WIDTH, DEFAULT_HEIGHT)
    assert s4.size == (DEFAULT_WIDTH, DEFAULT_HEIGHT)
    assert np.array_equal(np.array(s1), np.array(img1))
    assert np.array_equal(np.array(s2), np.array(img2))
    assert np.array_equal(np.array(s3), np.array(img3))
    assert np.array_equal(np.array(s4), np.array(img4))
    print("    4-image tests passed!")
    
    # 2. 2-Image Merging/Splitting
    print("  Testing 2-image merging and splitting...")
    merged2_h = merge(img1, img2, direction="horizontal")
    assert merged2_h.size == (DEFAULT_WIDTH * 2, DEFAULT_HEIGHT)
    sh1, sh2 = split(merged2_h)
    assert sh1.size == (DEFAULT_WIDTH, DEFAULT_HEIGHT)
    assert sh2.size == (DEFAULT_WIDTH, DEFAULT_HEIGHT)
    assert np.array_equal(np.array(sh1), np.array(img1))
    assert np.array_equal(np.array(sh2), np.array(img2))
    
    merged2_v = merge(img1, img2, direction="vertical")
    assert merged2_v.size == (DEFAULT_WIDTH, DEFAULT_HEIGHT * 2)
    sv1, sv2 = split(merged2_v)
    assert sv1.size == (DEFAULT_WIDTH, DEFAULT_HEIGHT)
    assert sv2.size == (DEFAULT_WIDTH, DEFAULT_HEIGHT)
    assert np.array_equal(np.array(sv1), np.array(img1))
    assert np.array_equal(np.array(sv2), np.array(img2))
    print("    2-image tests passed!")
    
    print("All self-tests passed successfully!")

def main():
    parser = argparse.ArgumentParser(
        description="Merge and split orthographic walk perspectives in differentiable_minecraft_renderer."
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcommand to run")
    
    # Merge subcommand
    merge_parser = subparsers.add_parser("merge", help="Merge two or four images.")
    # Arguments for 2-image merge
    merge_parser.add_argument("--front", help="Path to front view image (for 2-image merge).")
    merge_parser.add_argument("--back", help="Path to back view image (for 2-image merge).")
    merge_parser.add_argument("--direction", default="horizontal", choices=["horizontal", "vertical"],
                              help="Direction for 2-image merge (horizontal or vertical).")
    # Arguments for 4-image merge
    merge_parser.add_argument("--walk_front_base", help="Path to walk_front_base_layer_ortho image.")
    merge_parser.add_argument("--walk_back_base", help="Path to walk_back_base_layer_ortho image.")
    merge_parser.add_argument("--walk_front_both", help="Path to walk_front_both_layer_ortho image.")
    merge_parser.add_argument("--walk_back_both", help="Path to walk_back_both_layer_ortho image.")
    
    merge_parser.add_argument("--output", required=True, help="Path to save the merged output image.")
    
    # Split subcommand
    split_parser = subparsers.add_parser("split", help="Split a merged image back into two or four views.")
    split_parser.add_argument("--input", required=True, help="Path to the merged input image.")
    # Arguments for 2-image split
    split_parser.add_argument("--front", help="Path to save front view.")
    split_parser.add_argument("--back", help="Path to save back view.")
    split_parser.add_argument("--direction", default="auto", choices=["auto", "horizontal", "vertical"],
                              help="Direction for 2-image split.")
    # Arguments for 4-image split
    split_parser.add_argument("--walk_front_base", help="Path to save walk_front_base_layer_ortho.")
    split_parser.add_argument("--walk_back_base", help="Path to save walk_back_base_layer_ortho.")
    split_parser.add_argument("--walk_front_both", help="Path to save walk_front_both_layer_ortho.")
    split_parser.add_argument("--walk_back_both", help="Path to save walk_back_both_layer_ortho.")
                              
    # Test subcommand
    subparsers.add_parser("test", help="Run self-verification tests.")
    
    args = parser.parse_args()
    
    if args.command == "merge":
        if args.walk_front_base or args.walk_back_base or args.walk_front_both or args.walk_back_both:
            for name, path in [
                ("walk_front_base", args.walk_front_base),
                ("walk_back_base", args.walk_back_base),
                ("walk_front_both", args.walk_front_both),
                ("walk_back_both", args.walk_back_both),
            ]:
                if not path or not os.path.exists(path):
                    print(f"Error: {name} image path missing or not found: {path}")
                    sys.exit(1)
            
            img1 = Image.open(args.walk_front_base)
            img2 = Image.open(args.walk_back_base)
            img3 = Image.open(args.walk_front_both)
            img4 = Image.open(args.walk_back_both)
            merged_img = merge(img1, img2, img3, img4)
        else:
            if not args.front or not args.back:
                print("Error: For 2-image merge, specify either 4-image arguments or both --front and --back.")
                sys.exit(1)
            if not os.path.exists(args.front):
                print(f"Error: Front image not found at: {args.front}")
                sys.exit(1)
            if not os.path.exists(args.back):
                print(f"Error: Back image not found at: {args.back}")
                sys.exit(1)
            img_front = Image.open(args.front)
            img_back = Image.open(args.back)
            merged_img = merge(img_front, img_back, direction=args.direction)
            
        out_dir = os.path.dirname(os.path.abspath(args.output))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            
        merged_img.save(args.output)
        print(f"Successfully merged views into: {args.output} (Size: {merged_img.size})")
        
    elif args.command == "split":
        if not os.path.exists(args.input):
            print(f"Error: Merged input image not found at: {args.input}")
            sys.exit(1)
            
        merged_img = Image.open(args.input)
        
        if args.walk_front_base or args.walk_back_base or args.walk_front_both or args.walk_back_both:
            for name in ["walk_front_base", "walk_back_base", "walk_front_both", "walk_back_both"]:
                if not getattr(args, name):
                    print(f"Error: For 4-image split, --{name} must be specified.")
                    sys.exit(1)
            s1, s2, s3, s4 = split(merged_img)
            
            for out_path in [args.walk_front_base, args.walk_back_base, args.walk_front_both, args.walk_back_both]:
                out_dir = os.path.dirname(os.path.abspath(out_path))
                if out_dir:
                    os.makedirs(out_dir, exist_ok=True)
                    
            s1.save(args.walk_front_base)
            s2.save(args.walk_back_base)
            s3.save(args.walk_front_both)
            s4.save(args.walk_back_both)
            print(f"Successfully split image into four views:")
            print(f"  walk_front_base: {args.walk_front_base}")
            print(f"  walk_back_base:  {args.walk_back_base}")
            print(f"  walk_front_both: {args.walk_front_both}")
            print(f"  walk_back_both:  {args.walk_back_both}")
        else:
            if not args.front or not args.back:
                print("Error: For 2-image split, specify either 4-image arguments or both --front and --back.")
                sys.exit(1)
            s1, s2 = split(merged_img, direction=args.direction)
            
            for out_path in [args.front, args.back]:
                out_dir = os.path.dirname(os.path.abspath(out_path))
                if out_dir:
                    os.makedirs(out_dir, exist_ok=True)
                    
            s1.save(args.front)
            s2.save(args.back)
            print(f"Successfully split image into two views:")
            print(f"  front: {args.front}")
            print(f"  back:  {args.back}")
            
    elif args.command == "test":
        run_self_tests()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
