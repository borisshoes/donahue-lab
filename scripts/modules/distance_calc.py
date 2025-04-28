import os, heapq, sys
import numpy as np
from numba import njit, prange
from collections import deque
import matplotlib.pyplot as plt
from modules import calc_voronoi_points
from modules.stack_utils import load_image_stack, save_image_stack, compute_distance_transform

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

def find_nodes(data_stack, mask_stack, dims = [1.0, 1.0, 1.0]):
    points_stack, distance_stack, points = calc_voronoi_points.find_distance_maxima(~data_stack, mask_stack, dims = dims)

    # prepare empty output mask (0/255)
    spheres = np.zeros_like(points_stack, dtype=np.uint8)

    # find all the center-voxels
    centers = np.argwhere(points_stack == 255)

    for z, y, x in centers:
        r = distance_stack[z, y, x]
        if r <= 0:
            continue  # no sphere to draw

        # compute how many voxels out to go in each axis
        rz = int(np.ceil(r / dims[0]))
        ry = int(np.ceil(r / dims[1]))
        rx = int(np.ceil(r / dims[2]))

        # clamp bounding‐box to volume
        z0, z1 = max(0, z - rz), min(spheres.shape[0], z + rz + 1)
        y0, y1 = max(0, y - ry), min(spheres.shape[1], y + ry + 1)
        x0, x1 = max(0, x - rx), min(spheres.shape[2], x + rx + 1)

        # generate coordinate offsets within that box
        dz = np.arange(z0, z1) - z
        dy = np.arange(y0, y1) - y
        dx = np.arange(x0, x1) - x
        zz, yy, xx = np.meshgrid(dz, dy, dx, indexing='ij')

        # compute squared physical distance
        dist2 = (zz * dims[0])**2 + (yy * dims[1])**2 + (xx * dims[2])**2

        # fill where inside the sphere
        mask = dist2 <= (r ** 2)
        subvol = spheres[z0:z1, y0:y1, x0:x1]
        subvol[mask] = 255
        spheres[z0:z1, y0:y1, x0:x1] = subvol

    return spheres


def calculate_distribution(distance_stack, data_stack, distance_bin = 1.0):
    """
    Calculate and plot the percentage of white pixels (value 255) in the raw bitmap stack
    for each unique distance value (excluding -1). Also computes the overall percentage of white pixels,
    considering only pixels with a distance >= 0, and plots both the total pixel count and white pixel count 
    (sharing the same axis labeled "Pixels"). 95% confidence intervals are added to the percentage bar chart.
    
    Parameters:
        distance_stack (np.ndarray): 3D array of distance values.
        data_folder (str): Path to the folder containing the raw bitmap image stack.
    """
    
    # Calculate overall white pixel percentage only for pixels with a valid distance (>= 0)
    valid_mask = (distance_stack >= 0)
    total_pixels_overall = np.sum(valid_mask)
    white_pixels_overall = np.sum(data_stack[valid_mask] == 255)
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

    white_mask = np.logical_and(data_stack == 255, valid_mask)
    black_mask = np.logical_and(data_stack == 0, valid_mask)

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
    mask_stack, filenames = load_image_stack(input_folder)
    data_stack, _ = load_image_stack(data_folder)

    # edge_stack = process_stack(mask_stack)
    # save_image_stack(edge_stack_folder, edge_stack, filenames)

    # distance_stack = compute_distance_transform(mask_stack, edge_stack, [0.7421875,0.7421875,1.0])
    # normalized_stack = normalize_distance_stack(distance_stack)
    # save_image_stack(distance_map_folder, normalized_stack, filenames)
    # calculate_distribution(distance_stack, data_stack, 0.75)
    
    node_stack = find_nodes(data_stack, mask_stack, [0.7421875,0.7421875,1.0])
    save_image_stack("./distance_calc/node_stack", node_stack, filenames)


if __name__ == "__main__":
    

    #main("./BHS6 Bitmaps - Cleaned", "./mask/BHS6 Masks/3d_masks_auto/final_mask", "./distance_calc/distance_masks")
    main("./mask/BHS6 Masks/3d_masks_auto/final_mask", "./BHS6 Bitmaps - Binary Cleaned", "./distance_calc/distance_masks","./distance_calc/edge_masks")