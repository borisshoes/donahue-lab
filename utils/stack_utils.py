import os, sys
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

def load_image_stack(folder):
    files = sorted([f for f in os.listdir(folder) if f.endswith(".bmp")])
    stack = [np.array(Image.open(os.path.join(folder, f)).convert("L"), dtype=np.uint8) for f in files]
    return np.array(stack), files

def save_image_stack(output_folder, stack, filenames):
    os.makedirs(output_folder, exist_ok=True)
    for img, name in zip(stack, filenames):
        img = Image.fromarray(img.astype(np.uint8))
        img.save(os.path.join(output_folder, name))

# Calculates the euclidean distance of all masked pixels from edge
def compute_distance_transform(mask_stack, edge_stack, dims = [1.0,1.0,1.0]):
    white_mask = (edge_stack == 255)
    distances = distance_transform_edt(~white_mask, sampling=tuple(dims))
    result = np.where(mask_stack == 255, distances, -1)
    return result

