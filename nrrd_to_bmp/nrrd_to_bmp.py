import nrrd
import numpy as np
from PIL import Image
import os
import sys

import nrrd
import numpy as np
from PIL import Image
import os
import sys

def convert_nrrd_to_bmp(nrrd_filename, output_dir, slice_axis='first'):
    # Load the NRRD file
    data, header = nrrd.read(nrrd_filename)
    
    # Debug: print data shape and range
    print("Data shape:", data.shape)
    print("Data min:", data.min(), "Data max:", data.max())

    # Ensure the data is at least 2D
    if data.ndim < 2:
        print("Error: Input data must be at least 2D.")
        return

    # Always convert the data to float and then normalize linearly to 8-bit (0-255)
    data = data.astype(np.float32)
    data_min = data.min()
    data_max = data.max()
    if data_max - data_min == 0:
        print("Warning: Data has a constant value. Output will be zeroed.")
        data = np.zeros_like(data, dtype=np.uint8)
    else:
        data = ((data - data_min) / (data_max - data_min) * 255).astype(np.uint8)

    # Create the output directory if it doesn't exist.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # If the data is 3D, decide which axis to use for slicing.
    if data.ndim == 3:
        if slice_axis == 'first':
            num_slices = data.shape[0]
            slice_getter = lambda i: data[i]
        elif slice_axis == 'last':
            num_slices = data.shape[2]
            slice_getter = lambda i: data[:, :, i]
        else:
            print("Unsupported slice_axis. Use 'first' or 'last'.")
            return

        for i in range(num_slices):
            img = Image.fromarray(slice_getter(i))
            output_filename = os.path.join(output_dir, f"slice_{i:03d}.bmp")
            img.save(output_filename)
            print(f"Saved slice {i} as {output_filename}")

    # If the data is 2D, simply save it as one image.
    elif data.ndim == 2:
        img = Image.fromarray(data)
        output_filename = os.path.join(output_dir, "image.bmp")
        img.save(output_filename)
        print(f"Saved 2D image as {output_filename}")
    else:
        print("Unsupported data dimensions:", data.ndim)

if __name__ == "__main__":
    convert_nrrd_to_bmp("./Cropped NRRD.nrrd", "./nrrd_to_bmp/output", "last")
