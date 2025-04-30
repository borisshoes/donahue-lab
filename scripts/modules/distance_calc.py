import os, heapq, sys, random, time
import numpy as np
from numba import njit, prange
from collections import deque
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map
import matplotlib.pyplot as plt
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor, as_completed
from modules import calc_voronoi_points
from modules.stack_utils import load_image_stack, save_image_stack, compute_distance_transform
from modules.a_star_pathfind import a_star_pathfind

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
    points_stack, distance_stack, points, distances = calc_voronoi_points.find_distance_maxima(~data_stack, mask_stack, dims = dims, merge_radius=3)

    # prepare output array (same dtype & shape as points_stack)
    node_stack = np.zeros_like(points_stack)
    white_value = points_stack.max()  # so if input mask is 0/255, we fill with 255; if 0/1, with 1

    dz, dy, dx = dims

    culled_points = []
    count = 0
    total = len(points)
    for z0, y0, x0 in points:
        count+=1
        if count % 50 == 0:
            print(f"{count}/{total}")

        r = distances[z0, y0, x0]
        if r <= 1.5:
            continue
        culled_points.append((z0,y0,x0))

        # compute how many voxels in each direction we need to cover that radius
        rz = int(np.ceil(r / dz))
        ry = int(np.ceil(r / dy))
        rx = int(np.ceil(r / dx))

        # clamp to image bounds
        z_min, z_max = max(z0 - rz, 0), min(z0 + rz + 1, node_stack.shape[0])
        y_min, y_max = max(y0 - ry, 0), min(y0 + ry + 1, node_stack.shape[1])
        x_min, x_max = max(x0 - rx, 0), min(x0 + rx + 1, node_stack.shape[2])

        # build a small grid of coordinates in the bounding box
        zz, yy, xx = np.ogrid[z_min:z_max, y_min:y_max, x_min:x_max]

        # compute physical distance from (z0,y0,x0)
        dist = np.sqrt(((zz - z0) * dz) ** 2 +
                       ((yy - y0) * dy) ** 2 +
                       ((xx - x0) * dx) ** 2)

        # wherever that distance is <= r, set the output to white_value
        mask = (dist <= r)
        node_stack[z_min:z_max, y_min:y_max, x_min:x_max][mask] = white_value

    print(f"Found {len(culled_points)} unculled points")
    return points_stack, distance_stack, node_stack, distances, culled_points


def compute_path_length(path, dims):
    """
    Sum euclidean segment lengths along `path`, scaling each axis by dims.
    """
    total = 0.0
    for (z0, y0, x0), (z1, y1, x1) in zip(path, path[1:]):
        dz = (z1 - z0) * dims[0]
        dy = (y1 - y0) * dims[1]
        dx = (x1 - x0) * dims[2]
        total += np.sqrt(dx*dx + dy*dy + dz*dz)
    return total

def calc_deflection_ratio(path, dims=[1.0, 1.0, 1.0]):
    """
    Path is a sequence of (z, y, x) points.
    Compute the ratio of the max perpendicular deflection
    from the straight line (start→end) over that line's length.
    """
    if len(path) < 2:
        return 0.0

    pts = np.asarray(path, dtype=float)  # shape (N,3) in (z,y,x)

    scales = np.asarray(dims, dtype=float)    # (z_scale, y_scale, x_scale)
    scaled = pts * scales                 # apply anisotropic scaling

    start = scaled[0]
    end   = scaled[-1]
    line_vec = end - start
    line_len = np.linalg.norm(line_vec)
    if line_len == 0.0:
        return 0.0

    # vector from start to each point
    diffs = scaled - start                    # shape (N,3)

    # perp distance = ||cross(diffs, line_vec)|| / |line_vec|
    cross = np.cross(diffs, line_vec)         # shape (N,3)
    perp_dists = np.linalg.norm(cross, axis=1) / line_len

    max_deflection = perp_dists.max()
    return max_deflection / line_len

def calc_path_stats(paths, distance_bin = 1.0, dims = [1.0, 1.0, 1.0]):
    """
    Bin `paths` by their scaled length, compute deflection ratio on each,
    and plot:
      - bar = mean(deflection_ratio*100) per bin (left y)
      - line = count of paths in each bin       (right y)

    Parameters
    ----------
    paths : List[List[tuple]]
        Each path is a list of (z,y,x) points.
    calc_deflection_ratio : Callable[[List[tuple]], float]
        Function returning a deflection ratio (0–1) for a single path.
    distance_bin : float
        Bin width along the length axis.
    dims : sequence of 3 floats
        Scale factors for z, y, x when computing true path length.
    """
    print("Calculating path statistics")
    # 1) compute lengths and ratios
    lengths = np.array([compute_path_length(p, dims) for p in paths])
    ratios  = np.array([calc_deflection_ratio(p) * 100.0 for p in paths])

    # 2) define bins
    max_len = lengths.max() if len(lengths)>0 else 0.0
    nbins = max(1, int(np.ceil(max_len / distance_bin)))
    bin_edges = np.arange(0, (nbins+1)*distance_bin, distance_bin)

    # 3) assign each path to a bin
    bin_idxs = np.floor_divide(lengths, distance_bin).astype(int)
    bin_idxs = np.clip(bin_idxs, 0, nbins-1)

    # 4) aggregate per bin
    mean_ratio = np.zeros(nbins)
    counts     = np.zeros(nbins, dtype=int)
    for b in range(nbins):
        sel = (bin_idxs == b)
        counts[b]     = sel.sum()
        mean_ratio[b] = ratios[sel].mean() if counts[b]>0 else 0.0
    
    # z-score for 95% confidence interval
    z = 1.96
    ci_errors = z * np.sqrt(np.divide(mean_ratio * (100 - mean_ratio), counts, out=np.full_like(mean_ratio, 0.0, dtype=float), where=(counts != 0)))

    # 5) plot
    x = np.arange(nbins)
    fig, ax1 = plt.subplots(figsize=(10,6))

    bars = ax1.bar(x, mean_ratio, label='Mean Deflection (%)', alpha=0.75, yerr=ci_errors)
    ax1.set_xlabel(f'Path Length (mm) [bin size = {distance_bin} mm]')
    ax1.set_ylabel('Mean Deflection Ratio (%)')
    ax1.set_ylim(0, mean_ratio.max()*1.1 if len(mean_ratio)>0 else 1)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{bin_edges[i]:.1f}–{bin_edges[i+1]:.1f}" for i in x],
                        rotation=45, ha='right')

    ax2 = ax1.twinx()
    line_counts, = ax2.plot(x, counts, marker='o', linestyle='-',
                             label='Paths per Bin')
    ax2.set_ylabel('Number of Paths')
    ax2.set_ylim(0, counts.max()*1.1 if len(counts)>0 else 1)

    # legend
    ax1.legend(loc='upper left')
    ax2.legend([line_counts], ['Paths per Bin'], loc='upper right')

    plt.title('Deflection Ratio vs. Path Length Distribution')
    fig.tight_layout()
    plt.show()

def calc_bone_distribution(distance_stack, data_stack, distance_bin = 1.0):
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
    print(f"{total_pixels_overall} {white_pixels_overall}")
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

def trace_segments(distances, nodes, dims = [1.0,1.0,1.0]):
    trace_stack = np.zeros_like(distances, dtype=np.uint8)

    paths = []
    pairs = list(combinations(nodes, 2))
    random.shuffle(pairs)

    count = 0
    num_pairs = len(pairs)
    for p1, p2 in pairs:
        paths.append(a_star_pathfind(distances,p1,p2,dims,alpha=0.3))
        count += 1
        if count % 10000 == 0:
            print(f"{count}/{num_pairs}")

    print("Completed pathfinding, now painting")
    for path in paths:
        for (z, y, x) in path:
            if (0 <= z < trace_stack.shape[0] and
                0 <= y < trace_stack.shape[1] and
                0 <= x < trace_stack.shape[2]):
                trace_stack[z, y, x] = 255

    return trace_stack, paths

def trace_segments_mt(distances, nodes, dims=[1.0,1.0,1.0],
                      alpha=0.3, max_workers=None):
    trace_stack = np.zeros_like(distances, dtype=np.uint8)
    pairs = list(combinations(nodes, 2))
    random.shuffle(pairs)
    num_pairs = len(pairs)

    paths = []
    # helper to call A* with our fixed params
    def _worker(p1, p2):
        return a_star_pathfind(distances, p1, p2, dims, alpha)

    print("Engaging pathfinding, this may take a while...")
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        # submit all tasks
        futures = {exe.submit(_worker, p1, p2): i
                   for i, (p1, p2) in enumerate(pairs, start=1)}

        for fut in as_completed(futures):
            idx = futures[fut]
            path = fut.result()
            paths.append(path)

    end   = time.perf_counter()
    print(f"Completed pathfinding in {end - start:.6f} seconds, now painting")
    # now paint them in
    for path in paths:
        for (z, y, x) in path:
            if (0 <= z < trace_stack.shape[0] and
                0 <= y < trace_stack.shape[1] and
                0 <= x < trace_stack.shape[2]):
                trace_stack[z, y, x] = 255

    return trace_stack, paths


def calculate_bone_fraction_curve(mask_folder, data_folder, distance_map_folder, edge_stack_folder):
    mask_stack, filenames = load_image_stack(mask_folder)
    data_stack, _ = load_image_stack(data_folder)

    edge_stack = process_stack(mask_stack)
    save_image_stack(edge_stack_folder, edge_stack, filenames)

    distance_stack = compute_distance_transform(mask_stack, edge_stack, [1.0,0.7421875,0.7421875])
    normalized_stack = normalize_distance_stack(distance_stack)
    save_image_stack(distance_map_folder, normalized_stack, filenames)
    calc_bone_distribution(distance_stack, data_stack, 0.75)

def generate_node_stack(mask_folder, data_folder, distance_map_folder, point_folder, node_folder, trace_folder):
    dims = [1.0,0.7421875,0.7421875]
    mask_stack, filenames = load_image_stack(mask_folder)
    data_stack, _ = load_image_stack(data_folder)
    point_stack, distance_stack, node_stack, distances, points = find_nodes(data_stack, mask_stack, dims)
    trace_stack, paths = trace_segments_mt(distances,points,dims)
    save_image_stack(point_folder, point_stack, filenames)
    save_image_stack(distance_map_folder, distance_stack, filenames)
    save_image_stack(node_folder, node_stack, filenames)
    save_image_stack(trace_folder, trace_stack, filenames)
    calc_path_stats(paths,distance_bin=5.0,dims=dims)