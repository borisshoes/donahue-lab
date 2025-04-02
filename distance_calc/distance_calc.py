import os
import numpy as np
from PIL import Image
from numba import njit, prange
from collections import deque
import matplotlib.pyplot as plt

def load_image_stack(folder):
    files = sorted([f for f in os.listdir(folder) if f.endswith(".bmp")])
    stack = [np.array(Image.open(os.path.join(folder, f)).convert("L"), dtype=np.uint8) for f in files]
    return np.array(stack), files

def save_image_stack(output_folder, stack, filenames):
    os.makedirs(output_folder, exist_ok=True)
    for img, name in zip(stack, filenames):
        img = Image.fromarray(img.astype(np.uint8))
        img.save(os.path.join(output_folder, name))


# Calculates the taxicab distance of all masked pixels from edge
def compute_distance_transform(stack, new_stack):
    depth, height, width = stack.shape
    distance_stack = np.full((depth, height, width), -1, dtype=np.int32)
    queue = deque()

    size = depth*height*width
    index = 0
    
    for z in range(depth):
        for y in range(height):
            for x in range(width):
                if (100*(index)/size)//10 != (100*(index+1)/size)//10:
                    print(f"{index}/{size}")
                index += 1

                if new_stack[z, y, x] == 255:
                    distance_stack[z, y, x] = 0
                    queue.append((z, y, x))
    
    directions = [
        (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1),  # 2D neighbors
        (-1, 0, 0), (1, 0, 0)  # Vertical neighbors
    ]
    
    while queue:
        z, y, x = queue.popleft()
        current_distance = distance_stack[z, y, x]
        
        for dz, dy, dx in directions:
            nz, ny, nx = z + dz, y + dy, x + dx
            if 0 <= nz < depth and 0 <= ny < height and 0 <= nx < width:
                if distance_stack[nz, ny, nx] == -1 and stack[nz, ny, nx] == 255:
                    distance_stack[nz, ny, nx] = current_distance + 1
                    queue.append((nz, ny, nx))
    
    return distance_stack


# Shades mask based on distance
def normalize_distance_stack(distance_stack):
    max_distance = np.max(distance_stack[distance_stack >= 0])
    normalized_stack = np.full(distance_stack.shape, 0, dtype=np.uint8)
    
    for z in range(distance_stack.shape[0]):
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

def calculate_distribution(distance_stack, data_folder):
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
    
    # Get unique distance values, filter out -1, and sort them
    unique_distances = np.unique(distance_stack)
    unique_sorted = np.sort(unique_distances[unique_distances != -1])
    
    percentages = []
    ci_errors = []
    total_pixels_list = []
    white_pixels_list = []
    
    # z-score for 95% confidence interval
    z = 1.96
    
    # For each distance value, calculate total and white pixels, and compute confidence interval margin
    for d in unique_sorted:
        mask = (distance_stack == d)
        total_pixels = np.sum(mask)
        white_pixels = np.sum(raw_stack[mask] == 255)
        if total_pixels > 0:
            p = white_pixels / total_pixels
            percentage = p * 100
            ci_error = z * 100 * np.sqrt(p * (1 - p) / total_pixels)
        else:
            percentage = 0
            ci_error = 0
        percentages.append(percentage)
        ci_errors.append(ci_error)
        total_pixels_list.append(total_pixels)
        white_pixels_list.append(white_pixels)
    
    # Create a figure and primary axis for percentage
    fig, ax1 = plt.subplots(figsize=(10, 6))
    x = range(len(unique_sorted))
    
    # Bar chart for percentage of white pixels with 95% confidence interval error bars
    ax1.bar(x, percentages, yerr=ci_errors, color='C0', capsize=5, label='Bone Fraction (%)')
    ax1.set_xlabel("Taxicab Distance from Edge (Pixels)")
    ax1.set_ylabel("Bone Fraction (%)", color='C0')
    ax1.tick_params(axis='y', labelcolor='C0')
    ax1.set_ylim(bottom=0)  # Set the primary y-axis lower limit to 0
    plt.xticks(x, unique_sorted)
    
    # Create a secondary axis for pixel counts and set its lower limit to 0
    ax2 = ax1.twinx()
    line_total, = ax2.plot(x, total_pixels_list, color='C1', marker='o', linestyle='-', label='Total Pixels')
    line_white, = ax2.plot(x, white_pixels_list, color='C2', marker='o', linestyle='-', label='Bone Pixels')
    ax2.set_ylabel("Pixels")
    ax2.tick_params(axis='y')
    ax2.set_ylim(bottom=0)
    
    # Add a combined legend for the pixel counts
    ax2.legend(handles=[line_total, line_white], loc='upper right')
    
    plt.title(f"Bone Fraction by Pixel Distance (Avg Fraction: {overall_percentage:.2f}%)")
    fig.tight_layout()
    plt.show()


def main(input_folder, data_folder, output_folder):
    stack, filenames = load_image_stack(input_folder)
    processed_stack = process_stack(stack)
    distance_stack = compute_distance_transform(stack, processed_stack)
    normalized_stack = normalize_distance_stack(distance_stack)
    save_image_stack(output_folder, normalized_stack, filenames)

    calculate_distribution(distance_stack, data_folder)
    
if __name__ == "__main__":
    #main("./BHS6 Bitmaps - Cleaned", "./mask/BHS6 Masks/3d_masks_auto/final_mask", "./distance_calc/distance_masks")
    main("./mask/BHS6 Masks/3d_masks_auto/final_mask", "./BHS6 Bitmaps - Binary Cleaned", "./distance_calc/distance_masks")