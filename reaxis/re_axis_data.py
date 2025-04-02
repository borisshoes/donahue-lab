import os
import shutil
import numpy as np
from PIL import Image

def load_images(input_dir):
    """Loads a stack of BMP images from a directory, assuming they are ordered lexicographically."""
    files = sorted([f for f in os.listdir(input_dir) if f.endswith(".bmp")])
    images = [np.array(Image.open(os.path.join(input_dir, f))) for f in files]
    return np.stack(images), files

def save_images(output_dir, images, filenames):
    """Saves a stack of images to the specified directory."""
    os.makedirs(output_dir, exist_ok=True)
    # Iterate over all the entries in the folder
    for entry in os.listdir(output_dir):
        entry_path = os.path.join(output_dir, entry)
        try:
            # Check if it's a file or a symbolic link and remove it
            if os.path.isfile(entry_path) or os.path.islink(entry_path):
                os.unlink(entry_path)
            # If it's a directory, remove it and all its contents
            elif os.path.isdir(entry_path):
                shutil.rmtree(entry_path)
        except Exception as e:
            print(f"Failed to delete {entry_path}. Reason: {e}")

    for i, img in enumerate(images):
        Image.fromarray(img).save(os.path.join(output_dir, filenames[i]))

def reorient_stack(input_dir, output_dir):
    """Reorients a stack of BMP images along three cardinal axes and saves the results."""
    volume, filenames = load_images(input_dir)
    
    # Output directories
    axis_dirs = [
        os.path.join(output_dir, "axis_x"),
        os.path.join(output_dir, "axis_y"),
        os.path.join(output_dir, "axis_z")  # This will be the same as input_dir
    ]
    
    # Save original (axis_z)
    save_images(axis_dirs[2], volume, filenames)
    
    # Reorient for axis_x (slices along depth direction)
    rotated_x = np.transpose(volume, (1, 0, 2))  # Shape: (height, depth, width)
    save_images(axis_dirs[0], rotated_x, [f"slice_{i:03d}.bmp" for i in range(rotated_x.shape[0])])
    
    # Reorient for axis_y (slices along width direction)
    rotated_y = np.transpose(volume, (2, 1, 0))  # Shape: (width, height, depth)
    save_images(axis_dirs[1], rotated_y, [f"slice_{i:03d}.bmp" for i in range(rotated_y.shape[0])])
    
if __name__ == "__main__":
    reorient_stack("C:/Users/Boris/Desktop/Classwork Homework/UMass/Donahue Lab/Python Scripts/mask/BHS6 Masks/3d_masks/final_mask","./reaxis/BHS6_ReAxis_3d_Mask")
