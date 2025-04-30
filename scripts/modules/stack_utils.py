import os, sys, shutil
import numpy as np
from pathlib import Path
from PIL import Image
from scipy.ndimage import distance_transform_edt

def load_image_stack(folder):
    files = sorted([f for f in os.listdir(folder) if f.endswith(".bmp")])
    stack = [np.array(Image.open(os.path.join(folder, f)).convert("L"), dtype=np.uint8) for f in files]
    return np.array(stack), files

def save_image_stack(output_folder, stack, filenames):
    prep_folder(output_folder)
    for img, name in zip(stack, filenames):
        img = Image.fromarray(img.astype(np.uint8))
        img.save(os.path.join(output_folder, name))

# Calculates the euclidean distance of all masked pixels from edge
def compute_distance_transform(mask_stack, edge_stack, dims = [1.0,1.0,1.0]):
    white_mask = (edge_stack == 255)
    distances = distance_transform_edt(~white_mask, sampling=tuple(dims))
    result = np.where(mask_stack == 255, distances, -1)
    return result

def prep_folder(folder_path: str) -> None:
    """
    Clears all files and subfolders from the given folder if it exists,
    or creates the folder (and any missing parents) if it doesn't.
    """
    folder = Path(folder_path)
    if folder.exists():
        # Remove each child (file or directory) in the folder
        for child in folder.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    else:
        # Create the folder (and parents) if it doesn’t exist
        folder.mkdir(parents=True, exist_ok=True)

def validate_folder(folder_path: str) -> None:
    if not folder_path.exists():
        print(f"Error: input folder '{folder_path}' does not exist.", file=sys.stderr)
        sys.exit(1)
    if not folder_path.is_dir():
        print(f"Error: '{folder_path}' is not a directory.", file=sys.stderr)
        sys.exit(1)