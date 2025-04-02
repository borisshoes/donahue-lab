import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import math
import random
from skimage.filters import threshold_otsu

def binarize_ct_bmp(input_path, output_path):
    """
    Loads a BMP image (CT slice), applies Otsu's thresholding, and saves a binary (0 or 255) BMP.

    :param input_path:  Path to the input BMP (non-binary).
    :param output_path: Path to save the output BMP (binarized).
    """

    # 1. Load image in grayscale
    img = Image.open(input_path).convert("L")  # 'L' for 8-bit grayscale
    arr = np.array(img, dtype=np.uint8)

    # 2. Compute Otsu's threshold
    otsu_thresh = threshold_otsu(arr)

    # 3. Binarize: 0 if below threshold, 255 otherwise
    binary_arr = np.where(arr < otsu_thresh, 0, 255).astype(np.uint8)

    # 4. Save result
    out_img = Image.fromarray(binary_arr, mode="L")
    out_img.save(output_path, format="BMP")

    print(f"Binarized image saved to: {output_path}")

def load_binary_bmp(path):
    """
    Loads a binary BMP (pixel values 0 or 255) and returns:
    - A NumPy array of shape (height, width)
    - The width and height
    """
    img = Image.open(path).convert("L")  # ensure grayscale
    arr = np.array(img)
    # Ensure it's strictly 0 or 255
    arr[arr < 128] = 0
    arr[arr >= 128] = 255
    return arr, arr.shape[1], arr.shape[0]

def get_pixel_coords_by_color(arr):
    """
    Given a 2D array of 0 or 255, return two lists of (x, y) coordinates:
      - black_coords for pixels == 0
      - white_coords for pixels == 255
    Note: We use (x, y) = (col, row) convention.
    """
    black_coords = []
    white_coords = []
    height, width = arr.shape
    for y in range(height):
        for x in range(width):
            if arr[y, x] == 0:
                black_coords.append((x, y))
            else:
                white_coords.append((x, y))
    return black_coords, white_coords

def kmeans_representatives(coords, k):
    """
    Runs k-means on the given list of (x, y) coords, returns the cluster centers.
    """
    if len(coords) == 0 or k <= 0:
        return []
    # Convert to np array
    data = np.array(coords, dtype=float)
    # Fit k-means
    kmeans = KMeans(n_clusters=min(k, len(coords)), random_state=0).fit(data)
    centers = kmeans.cluster_centers_
    # Convert back to integer (round)
    centers = np.round(centers).astype(int)
    return [tuple(c) for c in centers]

def poisson_disc_sampling(coords, radius):
    """
    A simple accept-reject Poisson-disc-like sampling restricted to the given coords.
    - coords: list of (x, y) within the region (e.g. black or white)
    - radius: minimum distance required between any two accepted samples
    Returns a list of sample points.
    
    Note: This is a naive O(N^2) approach for demonstration. For large sets,
          a more advanced algorithm or spatial data structure is recommended.
    """
    if not coords:
        return []
    
    samples = []
    # Shuffle coords to get a random order
    random.shuffle(coords)
    
    for (x, y) in coords:
        # Check distance to existing samples
        too_close = False
        for (sx, sy) in samples:
            dist_sq = (x - sx)**2 + (y - sy)**2
            if dist_sq < radius**2:
                too_close = True
                break
        if not too_close:
            samples.append((x, y))
    return samples

def voronoi_reconstruct(width, height, black_points, white_points):
    """
    Given image dimensions and two sets of representative points,
    reconstruct a binary image using a 2-class Voronoi assignment:
      - For each pixel, find the nearest black point or white point.
      - Assign that pixel to black (0) or white (255).
    """
    # Convert lists to np arrays for faster distance checks if desired
    black_arr = np.array(black_points, dtype=float)
    white_arr = np.array(white_points, dtype=float)

    dark_color = 0
    light_color = 255
    point_color = 255
    
    out_arr = np.zeros((height, width), dtype=np.uint8)
    out_arr.fill(light_color)  # default to white, we'll overwrite with black if closer
    
    # For each pixel, compute nearest black vs white distance
    for y in range(height):
        for x in range(width):
            # If there are no black or white points, we must skip or default
            # (But normally we assume at least one of each)
            if len(black_arr) == 0:
                # If no black points, everything is white
                out_arr[y, x] = light_color
                continue
            if len(white_arr) == 0:
                # If no white points, everything is black
                out_arr[y, x] = dark_color
                continue
            
            # Dist to nearest black
            db = np.min(np.sum((black_arr - [x, y])**2, axis=1))
            # Dist to nearest white
            dw = np.min(np.sum((white_arr - [x, y])**2, axis=1))
            
            if db < dw:
                out_arr[y, x] = dark_color
            else:
                out_arr[y, x] = light_color

    return out_arr

def save_bmp(arr, path):
    """
    Save a 2D NumPy array (0 or 255) as a BMP file.
    """
    img = Image.fromarray(arr)
    img.convert("L").save(path, "BMP")

def display_images_side_by_side(image_paths):
    """
    Displays BMP images side by side for easy comparison.
    
    :param image_paths: List of file paths for the BMP images to display.
    """
    # Load images from the file paths.
    images = [Image.open(path) for path in image_paths]
    
    # Determine total width and maximum height needed for the combined image.
    widths, heights = zip(*(im.size for im in images))
    total_width = sum(widths)
    max_height = max(heights)
    
    # Create a new image with a white background.
    combined_image = Image.new("RGB", (total_width, max_height), color=(255, 255, 255))
    
    # Paste each image into the combined image.
    x_offset = 0
    for im in images:
        combined_image.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    
    # Display the combined image.
    combined_image.show()
    combined_image.save("results.png")

def main():
    # 1. Load input image (modify the path as needed)
    input_path = "input_binary.bmp"
    #binarize_ct_bmp("input.bmp",input_path)
    arr, width, height = load_binary_bmp(input_path)
    
    # 2. Separate pixel coords by color
    black_coords, white_coords = get_pixel_coords_by_color(arr)
    
    # ---------------------------
    # K-MEANS CLUSTERING
    # ---------------------------
    # Let's pick some cluster counts (adjust as needed)
    k_black = 100
    k_white = 100
    black_kmeans = kmeans_representatives(black_coords, k_black)
    white_kmeans = kmeans_representatives(white_coords, k_white)
    
    # Reconstruct with Voronoi
    kmeans_voronoi = voronoi_reconstruct(width, height, black_kmeans, white_kmeans)
    save_bmp(kmeans_voronoi, "kmeans_result.bmp")
    
    # ---------------------------
    # POISSON-DISC SAMPLING
    # ---------------------------
    # Let’s choose a radius (adjust based on image size)
    radius_black = 12
    radius_white = 12
    
    black_poisson = poisson_disc_sampling(black_coords, radius_black)
    white_poisson = poisson_disc_sampling(white_coords, radius_white)
    
    # Reconstruct with Voronoi
    poisson_voronoi = voronoi_reconstruct(width, height, black_poisson, white_poisson)
    save_bmp(poisson_voronoi, "poisson_result.bmp")
    
    print("Done! Created 'kmeans_result.bmp' and 'poisson_result.bmp'.")

    display_images_side_by_side(["input.bmp",input_path,"kmeans_result.bmp","poisson_result.bmp"])

if __name__ == "__main__":
    main()
