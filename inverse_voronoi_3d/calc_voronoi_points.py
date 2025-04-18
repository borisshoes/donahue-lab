import os
import numpy as np
from collections import deque
import multiprocessing as mp
import matplotlib.pyplot as plt
from utils.stack_utils import load_image_stack, save_image_stack, compute_distance_transform
from scipy.ndimage import distance_transform_edt, minimum_filter, maximum_filter

def write_points(coords_list, folder_path):
    # Verify that the folder exists
    if not os.path.isdir(folder_path):
        raise ValueError(f"The folder path '{folder_path}' is not valid or does not exist.")
    
    # Create the full file path
    file_path = os.path.join(folder_path, 'voronoi_points.txt')
    
    # Open the file and write the coordinates in x y z order
    with open(file_path, 'w') as file:
        for coord in coords_list:
            if len(coord) != 3:
                raise ValueError("Each coordinate tuple must contain exactly three elements (z, y, x).")
            z, y, x = coord
            # Write the coordinates in x y z order separated by a single space
            file.write(f"{x} {y} {z}\n")
    
    print(f"Coordinates have been written to {file_path}")


# Takes in two binary image stacks and first generates a distance map from every black pixel 
# to the nearest white pixel in data_stack, all black pixels in mask_stack have a distance value of -1
# Then finds the local maxima, with plateau merging of radius 'merge_radius' and returns a list of 3d points
# Returns a distance map, a stack of maxima, and a list of maxima
def find_distance_maxima(data_stack, mask_stack, merge_radius=3, dims = [1.0, 1.0, 1.0]):
    distances = compute_distance_transform(mask_stack, data_stack, dims)
    max_dist = distances.max()
    print(max_dist)
    depth, height, width = data_stack.shape
    distance_map = np.zeros_like(data_stack, dtype=np.uint8)

    for z in range(depth):
        for y in range(height):
            for x in range(width):
                value = distances[z, y, x]
                if value < 0:
                    value = 0
                if max_dist > 0:
                    value = int((value / max_dist) * 255.0)
                distance_map[z, y, x] = value

    print("Finding plateau-based maxima...")
    visited = np.zeros_like(distances, dtype=bool)
    maxima_points = []

    def find_plateau_and_centroid(z, y, x, value):
        queue = deque([(z, y, x)])
        plateau_points = []
        visited[z, y, x] = True
        while queue:
            cz, cy, cx = queue.popleft()
            plateau_points.append((cz, cy, cx))
            for dz in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dz == dy == dx == 0:
                            continue
                        nz, ny, nx = cz + dz, cy + dy, cx + dx
                        if (0 <= nz < depth and 0 <= ny < distances.shape[1] and 0 <= nx < distances.shape[2]
                                and not visited[nz, ny, nx] and distances[nz, ny, nx] == value):
                            visited[nz, ny, nx] = True
                            queue.append((nz, ny, nx))
        centroid = np.mean(plateau_points, axis=0).round().astype(int)
        return tuple(centroid)

    for z in range(1, depth - 1):
        for y in range(1, distances.shape[1] - 1):
            for x in range(1, distances.shape[2] - 1):
                if distances[z, y, x] > 0 and not visited[z, y, x]:
                    value = distances[z, y, x]
                    neighbors = [distances[z + dz, y + dy, x + dx]
                                 for dz in [-1, 0, 1] for dy in [-1, 0, 1] for dx in [-1, 0, 1]
                                 if not (dz == dy == dx == 0)]
                    if value >= max(neighbors):
                        centroid = find_plateau_and_centroid(z, y, x, value)
                        maxima_points.append(centroid)

    # Merge maxima points within merge_radius
    merged_points = []
    used = [False] * len(maxima_points)
    for i, p in enumerate(maxima_points):
        if used[i]:
            continue
        cluster = [p]
        used[i] = True
        for j in range(i + 1, len(maxima_points)):
            if not used[j]:
                dist = np.linalg.norm(np.array(p) - np.array(maxima_points[j]))
                if dist <= merge_radius:
                    cluster.append(maxima_points[j])
                    used[j] = True
        centroid = np.mean(cluster, axis=0).round().astype(int)
        merged_points.append(tuple(centroid))

    local_maxima_stack = np.zeros_like(distances, dtype=np.uint8)
    for (z, y, x) in merged_points:
        local_maxima_stack[z, y, x] = 255

    print(f"Found and merged {len(merged_points)} maxima.")
    return local_maxima_stack, distance_map, merged_points


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
    min_label = minimum_filter(labels, size=2)
    max_label = maximum_filter(labels, size=2)
    boundary_mask = (min_label != max_label)
    # Convert to binary image: boundaries white (255), elsewhere black (0)
    voronoi_graph = boundary_mask.astype(np.uint8) * 255
    return voronoi_graph

def generate_all(folder_path, mask_folder_path, output_folder_path, distance_map_output_folder, voronoi_reconstruction_folder, voronoi_point_list_folder):
    data_stack, data_files = load_image_stack(folder_path)
    mask_stack, mask_files = load_image_stack(mask_folder_path)

    # Compute the distance transform and extract local maxima (seed points)
    points_stack, distance_stack, points = find_distance_maxima(data_stack, mask_stack)
    save_image_stack(output_folder_path, points_stack, data_files)
    save_image_stack(distance_map_output_folder, distance_stack, data_files)
    write_points(points,voronoi_point_list_folder)

    # Use the local maxima (seed points) to generate the full Voronoi graph
    voronoi_graph_stack = generate_voronoi_graph(points_stack)
    save_image_stack(voronoi_reconstruction_folder, voronoi_graph_stack, data_files)


if __name__ == "__main__":
    #folder_path = "./BHS6 Bitmaps - Binary Cleaned"
    #mask_folder_path = "./mask/BHS6 Masks/3d_masks_auto/final_mask"
    folder_path = "./nrrd_to_bmp/output"
    mask_folder_path = "./mask/blank_mask"
    output_folder_path = "./inverse_voronoi_3d/point_cloud"
    distance_map_output_folder = "./inverse_voronoi_3d/distance_maps"
    voronoi_reconstruction_folder = "./inverse_voronoi_3d/voronoi_reconstruction"
    voronoi_point_list_path = "./inverse_voronoi_3d"

    generate_all(folder_path, mask_folder_path, output_folder_path, distance_map_output_folder, voronoi_reconstruction_folder,voronoi_point_list_path)
