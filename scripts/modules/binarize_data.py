import os, cv2, argparse, sys
from pathlib import Path
import numpy as np
from modules.stack_utils import prep_folder, validate_folder

def convert_ct_images_to_binary(input_folder, output_folder, bump=0):
    prep_folder(output_folder)

    # 1) collect all the .bmp paths
    files = [f for f in os.listdir(input_folder) if f.lower().endswith('.bmp')]
    if not files:
        print("No .bmp files found in", input_folder)
        return

    # 2) load every image and stash both its flattened pixels and the image itself
    all_pixels = []
    images = []
    for fn in files:
        path = os.path.join(input_folder, fn)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: could not load {fn}, skipping.")
            continue
        images.append((fn, img))
        all_pixels.append(img.ravel())
    
    # 3) concatenate into one long row and compute Otsu on it
    all_pixels = np.concatenate(all_pixels).reshape(1, -1)
    otsu_T, _ = cv2.threshold(
        all_pixels, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    print(f"Global Otsu threshold: {otsu_T}")

    # 4) optionally bump it up so your mid-grays stay black
    T = min(255, int(otsu_T) + bump)
    print(f"Using T = {T} for all slices")

    # 5) apply the same T to every image and save
    for fn, img in images:
        _, binary = cv2.threshold(img, T, 255, cv2.THRESH_BINARY)
        outp = os.path.join(output_folder, fn)
        cv2.imwrite(outp, binary)
        print(f"Processed and saved {fn}")

# Example usage:
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Binarizes the images in the input folder and sends them to the output folder.")
    parser.add_argument("input_folder", type=Path, help="Path to the input folder (must already exist).")
    parser.add_argument("output_folder", type=Path, help="Path to the output folder (will be cleared or created).")
    args = parser.parse_args()
    validate_folder(args.input_folder)

    convert_ct_images_to_binary(args.input_folder, args.output_folder)
