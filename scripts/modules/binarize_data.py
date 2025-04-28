import os, cv2, argparse, sys
from pathlib import Path
from modules.stack_utils import prep_folder, validate_folder

def convert_ct_images_to_binary(input_folder, output_folder):
    prep_folder(output_folder)

    # Loop through each file in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.lower().endswith('.bmp'):
            input_path = os.path.join(input_folder, file_name)
            # Load the image in grayscale mode (essential for thresholding)
            image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Warning: Could not load {input_path}. Skipping.")
                continue

            # Apply Otsu's thresholding
            # The threshold value is automatically determined.
            _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Construct the output path and save the binary image as a .bmp file
            output_path = os.path.join(output_folder, file_name)
            cv2.imwrite(output_path, binary_image)
            print(f"Processed {file_name} and saved binary image to {output_path}")

# Example usage:
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Binarizes the images in the input folder and sends them to the output folder.")
    parser.add_argument("input_folder", type=Path, help="Path to the input folder (must already exist).")
    parser.add_argument("output_folder", type=Path, help="Path to the output folder (will be cleared or created).")
    args = parser.parse_args()
    validate_folder(args.input_folder)

    convert_ct_images_to_binary(args.input_folder, args.output_folder)
