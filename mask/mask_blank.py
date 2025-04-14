import os
import argparse
from PIL import Image

def process_images(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
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
    input_folder = "./nrrd_to_bmp/output"
    output_folder = "./mask/blank_mask"
    process_images(input_folder, output_folder)