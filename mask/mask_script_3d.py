from . import mask_script_bmp
import os, shutil
import numpy as np
from PIL import Image
import cv2

def load_mask_stack(mask_dir):
    """Loads a stack of binary masks from a directory."""
    files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".bmp")])
    images = [np.array(Image.open(os.path.join(mask_dir, f))).astype(np.uint8) for f in files]
    return np.stack(images), files

def save_mask(output_path, mask):
    """Saves a single binary mask image."""
    Image.fromarray(mask).save(output_path)

def generate_and_combine_masks(input_dir, output_dir):
    """Generates masks for each axis and combines them into a final mask matching the Z-axis orientation."""
    axis_dirs = [
        os.path.join(input_dir, "axis_x"),
        os.path.join(input_dir, "axis_y"),
        os.path.join(input_dir, "axis_z")
    ]
    mask_dirs = [
        os.path.join(output_dir, "axis_x"),
        os.path.join(output_dir, "axis_y"),
        os.path.join(output_dir, "axis_z")
    ]
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

    final_mask_dir = os.path.join(output_dir, "final_mask")
    os.makedirs(final_mask_dir, exist_ok=True)
    
    # Generate masks for each axis
    for src, dst in zip(axis_dirs, mask_dirs):
        os.makedirs(dst, exist_ok=True)
        mask_script_bmp.make_masks(src, dst)
    
    # Load generated masks
    mask_x, _ = load_mask_stack(mask_dirs[0])  # Shape: (H, D, W)
    mask_y, _ = load_mask_stack(mask_dirs[1])  # Shape: (W, H, D)
    mask_z, filenames = load_mask_stack(mask_dirs[2])  # Shape: (D, H, W)
    
    # Reorient masks to match Z-axis
    mask_x_reoriented = np.transpose(mask_x, (1, 0, 2))  # (D, H, W)
    mask_y_reoriented = np.transpose(mask_y, (2, 1, 0))  # (D, H, W)
    
    # Compute final mask: at least two axes must be white
    #final_mask = ((mask_x_reoriented + mask_y_reoriented + mask_z) >= 2) * 255
    # Assuming mask_x_reoriented, mask_y_reoriented, and mask_z are binary masks (0 or 255)
    # Compute final mask brightness based on the number of white masks at each pixel
    # Create a binary mask for each mask
    mask_x_binary = (mask_x_reoriented > 0).astype(np.uint8)
    mask_y_binary = (mask_y_reoriented > 0).astype(np.uint8)
    mask_z_binary = (mask_z > 0).astype(np.uint8)

    # Sum the binary masks
    mask_sum = mask_x_binary + mask_y_binary + mask_z_binary

    # Initialize the final mask with zeros
    final_mask = np.zeros_like(mask_sum, dtype=np.uint8)

    # Set brightness levels based on the number of masks that are white
    final_mask[mask_sum == 1] = 85   # Brightness for 1 mask white
    final_mask[mask_sum == 2] = 170  # Brightness for 2 masks white
    final_mask[mask_sum == 3] = 255  # Brightness for all 3 masks white
    
    # Save final mask
    for i, img in enumerate(final_mask):
        save_mask(os.path.join(final_mask_dir, filenames[i]), img.astype(np.uint8))

    return final_mask

def compare_masks(input_dir, output_dir):
    compare_helper(os.path.join(input_dir, "axis_z"),os.path.join(output_dir, "final_mask"), "Final Mask")
    compare_helper(os.path.join(input_dir, "axis_x"),os.path.join(output_dir, "axis_x"), "X Mask")
    compare_helper(os.path.join(input_dir, "axis_y"),os.path.join(output_dir, "axis_y"), "Y Mask")
    compare_helper(os.path.join(input_dir, "axis_z"),os.path.join(output_dir, "axis_z"), "Z Mask")
    

def compare_helper(img_dir, mask_dir, title):
    imgs = []
    masks = []
    for filename in sorted(os.listdir(img_dir)):
        if filename.endswith(".bmp"):
            bmp_path = os.path.join(img_dir, filename)
            img = cv2.imread(bmp_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                imgs.append(img)
            else:
                print(f"Warning: Unable to read image: {bmp_path}")  # Warning output
    for filename in sorted(os.listdir(mask_dir)):
        if filename.endswith(".bmp"):
            bmp_path = os.path.join(mask_dir, filename)
            mask = cv2.imread(bmp_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                masks.append(mask)
            else:
                print(f"Warning: Unable to read mask: {bmp_path}")  # Warning output
    mask_script_bmp.compare_mask(imgs,masks,title)

if __name__ == "__main__":
    input_folder = "./reaxis/BHS6_ReAxis_2"
    output_folder = "C:/Users/Boris/Desktop/Classwork Homework/UMass/Donahue Lab/Python Scripts/mask/BHS6 Masks/3d_masks"
    generate_and_combine_masks(input_folder, output_folder)
    compare_masks(input_folder, output_folder)
