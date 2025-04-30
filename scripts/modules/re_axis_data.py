import os, shutil, argparse, sys
import numpy as np
from PIL import Image
from pathlib import Path
from modules.stack_utils import load_image_stack, save_image_stack

def reorient_stack(input_dir, output_dir):
    """Reorients a stack of BMP images along three cardinal axes and saves the results."""
    volume, filenames = load_image_stack(input_dir)
    
    # Output directories
    axis_dirs = [
        os.path.join(output_dir, "axis_x"),
        os.path.join(output_dir, "axis_y"),
        os.path.join(output_dir, "axis_z")  # This will be the same as input_dir
    ]
    
    # Save original (axis_z)
    save_image_stack(axis_dirs[2], volume, filenames)
    
    # Reorient for axis_x (slices along depth direction)
    rotated_x = np.transpose(volume, (1, 0, 2))  # Shape: (height, depth, width)
    save_image_stack(axis_dirs[0], rotated_x, [f"slice_{i:03d}.bmp" for i in range(rotated_x.shape[0])])
    
    # Reorient for axis_y (slices along width direction)
    rotated_y = np.transpose(volume, (2, 1, 0))  # Shape: (width, height, depth)
    save_image_stack(axis_dirs[1], rotated_y, [f"slice_{i:03d}.bmp" for i in range(rotated_y.shape[0])])
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resamples a stack of images in all cardinal axes and saves it to the output folder.")
    parser.add_argument("input_folder", type=Path, help="Path to the input folder (must already exist).")
    parser.add_argument("output_folder", type=Path, help="Path to the output folder (will be cleared or created).")
    args = parser.parse_args()
    if not args.input_folder.exists():
        print(f"Error: input folder '{args.input_folder}' does not exist.", file=sys.stderr)
        sys.exit(1)
    if not args.input_folder.is_dir():
        print(f"Error: '{args.input_folder}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    reorient_stack(args.input_folder,args.output_folder)
