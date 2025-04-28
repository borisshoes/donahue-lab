import nrrd, os, sys, argparse
import numpy as np
from pathlib import Path
from PIL import Image
from stack_utils import prep_folder

def convert_nrrd_to_bmp(nrrd_filename, output_dir, slice_axis='first'):
    # Load the NRRD file
    data, header = nrrd.read(nrrd_filename)

    prep_folder(output_dir)
    
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

def nrrd_file(path_str: str) -> Path:
    """
    argparse “type” function:  
    • Ensures the path has a .nrrd suffix  
    • (Optionally) ensures it exists on disk  
    Returns a pathlib.Path on success, or raises ArgumentTypeError.
    """
    p = Path(path_str)
    if p.suffix.lower() != ".nrrd":
        raise argparse.ArgumentTypeError(f"'{path_str}' is not a .nrrd file")
    if not p.exists():
        raise argparse.ArgumentTypeError(f"'{path_str}' does not exist")
    if not p.is_file():
        raise argparse.ArgumentTypeError(f"'{path_str}' is not a file")
    return p

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Converts an nrrd file to a bmp stack.")
    parser.add_argument("input_file", type=nrrd_file, help="A .nrrd file.")
    parser.add_argument("output_folder", type=Path, help="Path to the output folder (will be cleared or created).")
    args = parser.parse_args()

    convert_nrrd_to_bmp(args.input_file, args.output_folder, "last")
