import os, heapq, sys
import numpy as np
from PIL import Image
from numba import njit, prange
from collections import deque
import matplotlib.pyplot as plt
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
def compute_distance_transform(stack, edge_stack, dims = [1.0,1.0,1.0]):
    white_mask = (edge_stack == 255)
    distances = distance_transform_edt(~white_mask, sampling=tuple(dims))
    result = np.where(stack == 255, distances, -1)
    return result


# Shades mask based on distance
def normalize_distance_stack(distance_stack):
    max_distance = np.max(distance_stack[distance_stack >= 0])
    normalized_stack = np.full(distance_stack.shape, 0, dtype=np.uint8)
    
    size = distance_stack.shape[0]
    for z in range(distance_stack.shape[0]):
        if (100*(z)/size)//10 != (100*(z+1)/size)//10:
            print(f"{z}/{size}")
        for y in range(distance_stack.shape[1]):
            for x in range(distance_stack.shape[2]):
                if distance_stack[z, y, x] == -1:
                    normalized_stack[z, y, x] = 0  # Black for -1
                else:
                    normalized_stack[z, y, x] = 255 - int((distance_stack[z, y, x] / max_distance) * 255)  # Grayscale mapping
    
    return normalized_stack

# Finds edges of the mask
@njit(parallel=True)
def process_stack(stack):
    depth, height, width = stack.shape
    new_stack = np.zeros_like(stack, dtype=np.uint8)
    
    for z in prange(depth):
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if stack[z, y, x] == 255 and (z == 0 or z == depth-1):
                    new_stack[z, y, x] = 255  # Set white
                if stack[z, y, x] < 255:  # Only consider black pixels
                    # Check neighbors in 3D space for pure white (255) pixels
                    if (
                        stack[z, y-1, x] == 255 or stack[z, y+1, x] == 255 or
                        stack[z, y, x-1] == 255 or stack[z, y, x+1] == 255 or
                        (z > 0 and stack[z-1, y, x] == 255) or
                        (z < depth-1 and stack[z+1, y, x] == 255)
                    ):
                        new_stack[z, y, x] = 255  # Set white
    
    return new_stack

def calculate_distribution(distance_stack, data_folder, distance_bin = 1.0):
    """
    Calculate and plot the percentage of white pixels (value 255) in the raw bitmap stack
    for each unique distance value (excluding -1). Also computes the overall percentage of white pixels,
    considering only pixels with a distance >= 0, and plots both the total pixel count and white pixel count 
    (sharing the same axis labeled "Pixels"). 95% confidence intervals are added to the percentage bar chart.
    
    Parameters:
        distance_stack (np.ndarray): 3D array of distance values.
        data_folder (str): Path to the folder containing the raw bitmap image stack.
    """
    # Load the raw data stack from the provided folder
    raw_stack, _ = load_image_stack(data_folder)
    
    # Calculate overall white pixel percentage only for pixels with a valid distance (>= 0)
    valid_mask = (distance_stack >= 0)
    total_pixels_overall = np.sum(valid_mask)
    white_pixels_overall = np.sum(raw_stack[valid_mask] == 255)
    overall_percentage = (white_pixels_overall / total_pixels_overall) * 100
    print(f"Overall white pixel percentage (for pixels with distance >= 0): {overall_percentage:.2f}%")
    
    max_distance = np.max(distance_stack)

    # Calculate the number of bins.
    # If the maximum distance is not an exact multiple of distance_bin,
    # this still allocates the last bin to cover at least distance_bin.
    nbins = int(np.ceil(max_distance / distance_bin))
    nbins = max(nbins, 1)  # Ensure there's at least one bin.
    # Compute bin indices only for valid pixels.
    # For invalid pixels the value in bin_indices is not used.
    bin_indices = np.empty_like(distance_stack, dtype=np.int32)
    bin_indices[valid_mask] = np.floor(distance_stack[valid_mask] / distance_bin).astype(np.int32)
    bin_indices[valid_mask] = np.clip(bin_indices[valid_mask], 0, nbins - 1)

    white_mask = np.logical_and(raw_stack == 255, valid_mask)
    black_mask = np.logical_and(raw_stack == 0, valid_mask)

    # We apply the masks to bin_indices so that we only count pixels with the correct value.
    white_counts = np.bincount(bin_indices[white_mask].ravel(), minlength=nbins)
    black_counts = np.bincount(bin_indices[black_mask].ravel(), minlength=nbins)
    total_counts = white_counts + black_counts

    # Optionally, compute the left edges of the bins for reference.
    bin_edges = np.arange(0, nbins * distance_bin, distance_bin)
    
    # z-score for 95% confidence interval
    z = 1.96

    percentages = 100 * np.divide(white_counts, total_counts, out=np.zeros_like(white_counts, dtype=float), where=total_counts != 0)
    ci_errors = z * np.sqrt(np.divide(percentages * (100 - percentages), total_counts, out=np.full_like(percentages, 0.0, dtype=float), where=(total_counts != 0)))
    

    print(white_counts)
    print(total_counts)
    print(percentages)
    print(ci_errors)

    # Create a figure and primary axis for percentage
    fig, ax1 = plt.subplots(figsize=(10, 6))
    x = np.arange(nbins)
    
    # Bar chart for percentage of white pixels with 95% confidence interval error bars
    ax1.bar(x, percentages, yerr=ci_errors, color='C0', capsize=5, label='Bone Fraction (%)')
    ax1.set_xlabel("Euclidean Distance from Edge (mm)")
    ax1.set_ylabel("Bone Fraction (%)", color='C0')
    ax1.tick_params(axis='y', labelcolor='C0')
    ax1.set_ylim(bottom=0)  # Set the primary y-axis lower limit to 0
    plt.xticks(x, [f"{edge:.1f}-{edge + distance_bin:.1f}" for edge in bin_edges],rotation = 45)
    
    
    # Create a secondary axis for pixel counts and set its lower limit to 0
    ax2 = ax1.twinx()
    line_total, = ax2.plot(x, total_counts, color='C1', marker='o', linestyle='-', label='Total Pixels')
    line_white, = ax2.plot(x, white_counts, color='C2', marker='o', linestyle='-', label='Bone Pixels')
    ax2.set_ylabel("Pixels per Distance Bin")
    ax2.tick_params(axis='y')
    ax2.set_ylim(bottom=0)
    
    # Add a combined legend for the pixel counts
    ax2.legend(handles=[line_total, line_white], loc='upper right')
    
    plt.title(f"Bone Fraction by Pixel Distance (Avg Fraction: {overall_percentage:.2f}%)")
    fig.tight_layout()
    plt.show()


def main(input_folder, data_folder, distance_map_folder, edge_stack_folder):
    stack, filenames = load_image_stack(input_folder)
    edge_stack = process_stack(stack)

    save_image_stack(edge_stack_folder, edge_stack, filenames)
    distance_stack = compute_distance_transform(stack, edge_stack, [0.7421875,0.7421875,1.0])
    normalized_stack = normalize_distance_stack(distance_stack)
    save_image_stack(distance_map_folder, normalized_stack, filenames)
    calculate_distribution(distance_stack, data_folder, 0.75)
    


    
if __name__ == "__main__":
    #main("./BHS6 Bitmaps - Cleaned", "./mask/BHS6 Masks/3d_masks_auto/final_mask", "./distance_calc/distance_masks")
    main("./mask/BHS6 Masks/3d_masks_auto/final_mask", "./BHS6 Bitmaps - Binary Cleaned", "./distance_calc/distance_masks","./distance_calc/edge_masks")