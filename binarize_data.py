import os
import cv2

def convert_ct_images_to_binary(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

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
    convert_ct_images_to_binary("./BHS6 Bitmaps - Cleaned","BHS6 Bitmaps - Binary Cleaned")
    #convert_ct_images_to_binary("./nrrd_to_bmp/output","./test_output")
