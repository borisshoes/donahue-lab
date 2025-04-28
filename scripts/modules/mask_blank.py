import os, argparse
from pathlib import Path
from PIL import Image
from stack_utils import prep_folder, validate_folder

def process_images(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    prep_folder(output_folder)
    
    # Iterate over files in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.bmp'):
            input_path = os.path.join(input_folder, filename)
            with Image.open(input_path) as img:
                width, height = img.size
                # Create a white image with the same dimensions
                white_image = Image.new("RGB", (width, height), "white")
                output_path = os.path.join(output_folder, filename)
                white_image.save(output_path, format='BMP')
                print(f"Processed {filename}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Makes a stack of all-white images with the same dimensiosn as the input stack")
    parser.add_argument("input_folder", type=Path, help="Path to the input folder (must already exist).")
    parser.add_argument("output_folder", type=Path, help="Path to the output folder (will be cleared or created).")
    args = parser.parse_args()
    validate_folder(args.input_folder)
    process_images(args.input_folder, args.output_folder)