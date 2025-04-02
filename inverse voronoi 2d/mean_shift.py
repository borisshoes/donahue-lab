import numpy as np
from PIL import Image, ImageDraw
from sklearn.cluster import KMeans, MeanShift
import math, random, sys
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
import concurrent.futures

def load_binary_bmp(path):
    """
    Loads a binary BMP image (assumes pixels are 0 or 255) as grayscale.
    Returns a numpy array (dtype uint8) and image dimensions (width, height).
    """
    img = Image.open(path).convert("L")
    arr = np.array(img, dtype=np.uint8)
    # Ensure binary: any value below 128 -> 0, otherwise -> 255
    arr = np.where(arr < 128, 0, 255).astype(np.uint8)
    return arr, img.size[0], img.size[1]

def get_white_pixels(arr):
    """
    Returns a list of (x,y) coordinates for white pixels (value==255).
    """
    white_pixels = []
    height, width = arr.shape
    for y in range(height):
        for x in range(width):
            if arr[y, x] == 255:
                white_pixels.append((x, y))
    return white_pixels

def poisson_disc_sampling(coords, radius):
    """
    Naively performs Poisson-disc sampling (accept-reject) on a list of coordinates.
    Only returns a subset of points such that no two are closer than 'radius'.
    """
    if not coords:
        return []
    
    samples = []
    random.shuffle(coords)  # randomize order
    for (x, y) in coords:
        too_close = False
        for (sx, sy) in samples:
            if (x - sx)**2 + (y - sy)**2 < radius**2:
                too_close = True
                break
        if not too_close:
            samples.append((x, y))
    return samples

def save_points_image(points, width, height, path):
    """
    Creates and saves a BMP image of given dimensions (all black background)
    with white pixels at the provided 'points' coordinates.
    """
    img = Image.new("L", (width, height), 0)  # black background
    for (x, y) in points:
        if 0 <= x < width and 0 <= y < height:
            img.putpixel((x, y), 255)
    img.save(path, "BMP")
    print(f"Saved image: {path}")

def mean_shift_clustering(points, bandwidth=30):
    """
    Runs MeanShift clustering on the provided points.
    Returns the cluster centers as a list of (x,y) tuples.
    Bandwidth can be adjusted to control the smoothing.
    """
    if not points:
        return []
    data = np.array(points)
    ms = MeanShift(bandwidth=bandwidth)
    ms.fit(data)
    centers = ms.cluster_centers_
    centers = np.round(centers).astype(int)
    return [tuple(c) for c in centers]

def circle_fits(center, radius, original_arr):
    """
    Checks if a circle centered at 'center' with given 'radius' fits entirely
    within the white region of the original binary image (original_arr).
    Returns True if all pixels inside the circle are white (255), False otherwise.
    """
    cx, cy = center
    height, width = original_arr.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx*dx + dy*dy <= radius*radius:
                x = cx + dx
                y = cy + dy
                # Check bounds
                if x < 0 or x >= width or y < 0 or y >= height:
                    return False
                if original_arr[y, x] != 255:
                    return False
    return True

def find_max_circle(center, original_arr, max_radius=100):
    """
    Starting from radius=1, increases the radius until the circle no longer
    fits in the white region of the original binary image.
    Returns the maximum radius that fits.
    """
    r = 1
    while r <= max_radius and circle_fits(center, r, original_arr):
        r += 1
    return r - 1

def draw_filled_circles(centers, original_arr, output_path):
    """
    Creates a new BMP image by drawing the largest possible filled white circles
    around each cluster center without overfilling the original white region.
    """
    height, width = original_arr.shape
    # Start with a black image
    img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(img)
    
    for center in centers:
        max_r = find_max_circle(center, original_arr)
        if max_r > 0:
            x, y = center
            bbox = [x - max_r, y - max_r, x + max_r, y + max_r]
            draw.ellipse(bbox, fill=255)
    img.save(output_path, "BMP")
    print(f"Saved filled circles image: {output_path}")

    white_pixel_count = sum(1 for pixel in img.getdata() if pixel == 255)
    return white_pixel_count

def display_images_side_by_side(image_paths, output_path):
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
    combined_image.save(output_path)


def process_combination(poisson_radius, bandwidth, white_pixels, original_arr):
    """
    Function to process one combination of poisson_radius and bandwidth.
    Returns (poisson_radius, bandwidth, filled_pixels).
    """
    print(f"Running variation {poisson_radius}, {bandwidth}")
    sampled_points = poisson_disc_sampling(white_pixels, poisson_radius)
    cluster_centers = mean_shift_clustering(sampled_points, bandwidth)
    filled_pixels = draw_filled_circles(cluster_centers, original_arr, "temp_filled.bmp")
    
    return poisson_radius, bandwidth, filled_pixels

def multithread(input_path):
    original_arr, width, height = load_binary_bmp(input_path)
    white_pixels = get_white_pixels(original_arr)

    best_filled_count = 0
    best_params = None
    results = []

    poisson_radii = np.arange(1.5, 10.5, 0.5)  # 1 to 10 with 0.5 steps
    tasks = []

    # Create a list of tasks (radius, bandwidth pairs)
    for poisson_radius in poisson_radii:
        min_bandwidth = math.ceil(poisson_radius+0.5)
        for bandwidth in range(min_bandwidth, 21):  
            tasks.append((poisson_radius, bandwidth))

    # Use parallel processing to speed up computation
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_params = {
            executor.submit(process_combination, pr, bw, white_pixels, original_arr): (pr, bw)
            for pr, bw in tasks
        }

        for future in concurrent.futures.as_completed(future_to_params):
            pr, bw = future_to_params[future]
            try:
                poisson_radius, bandwidth, filled_pixels = future.result()
                results.append((poisson_radius, bandwidth, filled_pixels))

                if filled_pixels > best_filled_count:
                    best_filled_count = filled_pixels
                    best_params = (poisson_radius, bandwidth)
            except Exception as e:
                print(f"Error processing (Poisson Radius: {pr}, Bandwidth: {bw}): {e}")

    # Plot results
    poisson_vals, bandwidth_vals, filled_counts = zip(*results)
    
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(poisson_vals, bandwidth_vals, c=filled_counts, cmap='viridis', marker='s')
    plt.colorbar(scatter, label='Filled Pixels Count')
    plt.xlabel('Poisson Radius')
    plt.ylabel('Bandwidth')
    plt.title('Filled Pixels Count for Different Poisson Radius and Bandwidth')
    plt.show()
    
    print(f"Best parameters: Poisson Radius = {best_params[0]}, Bandwidth = {best_params[1]} with {best_filled_count} filled pixels.")


def single(input_path, radius, bandwidth):
    original_arr, width, height = load_binary_bmp(input_path)
    white_pixels = get_white_pixels(original_arr)


    sampled_points = poisson_disc_sampling(white_pixels, radius)
    cluster_centers = mean_shift_clustering(sampled_points, bandwidth)
    filled_pixels = draw_filled_circles(cluster_centers, original_arr, "temp_filled.bmp")
    
    save_points_image(sampled_points, width, height, "sampled.bmp")
    save_points_image(cluster_centers, width, height, "clustered.bmp")
    draw_filled_circles(cluster_centers, original_arr, "filled.bmp")
    display_images_side_by_side(["input.bmp", input_path, "sampled.bmp", "clustered.bmp", "filled.bmp"],f"mean_shift_reconstruction_{radius}_{bandwidth}.png")

def main():
    input_path = "input_binary.bmp"
    #multithread(input_path)
    single(input_path,2,5)
    
    
    
    
if __name__ == "__main__":
    main()