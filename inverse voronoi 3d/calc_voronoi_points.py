import os
import numpy as np
from PIL import Image
from collections import deque
import multiprocessing as mp
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt, minimum_filter, maximum_filter
import glob

def load_image_stack(folder):
    files = sorted([f for f in os.listdir(folder) if f.endswith(".bmp")])
    stack = [np.array(Image.open(os.path.join(folder, f)).convert("L"), dtype=np.uint8) for f in files]
    return np.array(stack), files

def save_image_stack(output_folder, stack, filenames):
    os.makedirs(output_folder, exist_ok=True)
    for img, name in zip(stack, filenames):
        img = Image.fromarray(img.astype(np.uint8))
        img.save(os.path.join(output_folder, name))

def bfs_find_distance(data_stack, z, y, x):
    depth, height, width = data_stack.shape
    queue = deque([(z, y, x, 0)])
    visited = set()
    visited.add((z, y, x))
    
    while queue:
        cz, cy, cx, dist = queue.popleft()
        if data_stack[cz, cy, cx] == 255:
            return dist
        
        for dz, dy, dx in [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]:
            nz, ny, nx = cz+dz, cy+dy, cx+dx
            if 0 <= nz < depth and 0 <= ny < height and 0 <= nx < width and (nz, ny, nx) not in visited:
                visited.add((nz, ny, nx))
                queue.append((nz, ny, nx, dist+1))
    return float('inf')  # Should never happen

def process_slice(args):
    z, data_stack, mask_stack = args
    height, width = data_stack.shape[1:]
    distance_map = np.zeros((height, width), dtype=np.uint8)
    print(f"Processing slice {z+1}/{data_stack.shape[0]}...")
    
    max_distance = 0
    voronoi_slice = np.zeros((height, width), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            # Only process black pixels (value 0) in data_stack where the mask is white (255)
            if data_stack[z, y, x] == 0 and mask_stack[z, y, x] == 255:
                distance = bfs_find_distance(data_stack, z, y, x)
                max_distance = max(max_distance, distance)
                voronoi_slice[y, x] = distance
    
    # Scale distances for better visibility in the distance map
    if max_distance > 0:
        for y in range(height):
            for x in range(width):
                if voronoi_slice[y, x] > 0:
                    distance_map[y, x] = int((voronoi_slice[y, x] / max_distance) * 255)
    
    return z, voronoi_slice, distance_map

def compute_voronoi(data_stack, mask_stack):
    depth = data_stack.shape[0]
    voronoi_stack = np.zeros_like(data_stack, dtype=np.uint8)
    distance_stack = np.zeros_like(data_stack, dtype=np.uint8)
    
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(process_slice, [(z, data_stack, mask_stack) for z in range(depth)])
    
    for z, voronoi_slice, distance_map in results:
        voronoi_stack[z] = voronoi_slice
        distance_stack[z] = distance_map
    
    # Find local maxima in the distance map to obtain a sparse set of seed points
    print("Finding local maxima...")
    merge_radius = 3  # Adjust this value as needed for merging nearby maxima
    maxima_list = []
    for z in range(1, depth-1):
        for y in range(1, data_stack.shape[1]-1):
            for x in range(1, data_stack.shape[2]-1):
                if voronoi_stack[z, y, x] > 0:
                    neighbors = [voronoi_stack[z+dz, y+dy, x+dx]
                                 for dz in [-1, 0, 1]
                                 for dy in [-1, 0, 1]
                                 for dx in [-1, 0, 1]
                                 if not (dz == dy == dx == 0)]
                    if voronoi_stack[z, y, x] > max(neighbors):
                        # Save the local maximum as (z, y, x, value)
                        maxima_list.append((z, y, x, voronoi_stack[z, y, x]))
    
    # Sort the list so that higher values come first (to prioritize stronger maxima)
    maxima_list.sort(key=lambda p: p[3], reverse=True)
    
    # Merge local maxima within the merge_radius using a weighted mean
    merged_maxima = []
    used = [False] * len(maxima_list)
    
    for i, point in enumerate(maxima_list):
        if used[i]:
            continue
        # Start a new cluster with this maximum
        cluster = [point]
        used[i] = True
        for j in range(i+1, len(maxima_list)):
            if used[j]:
                continue
            dz = point[0] - maxima_list[j][0]
            dy = point[1] - maxima_list[j][1]
            dx = point[2] - maxima_list[j][2]
            dist = np.sqrt(dz*dz + dy*dy + dx*dx)
            if dist <= merge_radius:
                cluster.append(maxima_list[j])
                used[j] = True
        # Compute the weighted mean position (weights are the local maximum values)
        total_weight = sum(p[3] for p in cluster)
        merged_z = int(round(sum(p[0] * p[3] for p in cluster) / total_weight))
        merged_y = int(round(sum(p[1] * p[3] for p in cluster) / total_weight))
        merged_x = int(round(sum(p[2] * p[3] for p in cluster) / total_weight))
        merged_maxima.append((merged_z, merged_y, merged_x))
    
    # Create the final local_maxima_stack from the merged maxima positions
    local_maxima_stack = np.zeros_like(voronoi_stack, dtype=np.uint8)
    for (z, y, x) in merged_maxima:
        local_maxima_stack[z, y, x] = 255
    
    num_maxima = len(merged_maxima)
    print(f"Found {num_maxima} merged maxima...")
    return local_maxima_stack, distance_stack

def generate_voronoi_graph(local_maxima_stack):
    """
    Given a binary 3D volume (with 255 marking seed points, i.e. the local maxima)
    generate the full 3D Voronoi graph (i.e. the boundaries between regions).
    
    The approach is as follows:
      1. Convert the seed points to a binary image (1 for seed, 0 for background).
      2. Invert the image so that seeds become 0 (the zero-valued points) and the background 1.
      3. Compute the Euclidean distance transform with return_indices to get, for every voxel,
         the coordinates of the nearest seed.
      4. Combine the indices to form a unique label for each region.
      5. Use a minimum and maximum filter on the label volume to detect where neighboring voxels
         come from different seed points (i.e. the region boundaries).
      6. Return the resulting boundary volume (with 255 for boundaries and 0 for everything else).
    """
    # Convert to binary: seeds (white) become 1 and background 0
    seeds = (local_maxima_stack > 0).astype(np.uint8)
    # Invert: seeds become 0 (so that distance_transform_edt will consider them as the target)
    inverted = 1 - seeds
    # Compute the distance transform, returning indices of the nearest zero (seed)
    distances, indices = distance_transform_edt(inverted, return_indices=True)
    D, H, W = local_maxima_stack.shape
    # Combine the coordinates into a unique label for each seed
    labels = indices[0] + D * (indices[1] + H * indices[2])
    # Use a 3x3x3 minimum and maximum filter to find regions where the labels differ
    min_label = minimum_filter(labels, size=3)
    max_label = maximum_filter(labels, size=3)
    boundary_mask = (min_label != max_label)
    # Convert to binary image: boundaries white (255), elsewhere black (0)
    voronoi_graph = boundary_mask.astype(np.uint8) * 255
    return voronoi_graph

if __name__ == "__main__":
    folder_path = "./BHS6 Bitmaps - Binary Cleaned"
    mask_folder_path = "./mask/BHS6 Masks/3d_masks_auto/final_mask"
    output_folder_path = "./inverse voronoi 3d/point_cloud"
    distance_map_output_folder = "./inverse voronoi 3d/distance_maps"
    voronoi_reconstruction_folder = "./inverse voronoi 3d/voronoi_reconstruction"

    data_stack, data_files = load_image_stack(folder_path)
    mask_stack, mask_files = load_image_stack(mask_folder_path)

    # Compute the distance transform and extract local maxima (seed points)
    voronoi_stack, distance_stack = compute_voronoi(data_stack, mask_stack)
    save_image_stack(output_folder_path, voronoi_stack, data_files)
    save_image_stack(distance_map_output_folder, distance_stack, data_files)

    # Use the local maxima (seed points) to generate the full Voronoi graph
    voronoi_graph_stack = generate_voronoi_graph(voronoi_stack)
    save_image_stack(voronoi_reconstruction_folder, voronoi_graph_stack, data_files)
