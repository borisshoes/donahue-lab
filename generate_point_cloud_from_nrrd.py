import argparse, shutil, sys, os
import tkinter as tk
from tkinter import filedialog
from nrrd_to_bmp import nrrd_to_bmp
from mask import mask_blank
from inverse_voronoi_3d import calc_voronoi_points

def main():
    parser = argparse.ArgumentParser(
        description="Convert an nrrd image into a voronoi point cloud."
    )
    parser.add_argument("input_file", nargs='?', help="The file to generate the pointcloud from")
    args = parser.parse_args()

    if args.input_file:
        input_file = args.input_file
    else:
        root = tk.Tk()
        root.withdraw()  # Hide the root window
        input_file = filedialog.askopenfilename(
            title="Select NRRD File",
            filetypes=[("NRRD files", "*.nrrd")]
        )
        if input_file:
            print(f"Selected file: {input_file}")
        else:
            print("No file selected")
            sys.exit()

    folder_path = "./generated"
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        shutil.rmtree(folder_path)

    nrrd_to_bmp.convert_nrrd_to_bmp(input_file, "./generated/bmp_stack", "last")
    mask_blank.process_images("./generated/bmp_stack", "./generated/mask_stack")
    calc_voronoi_points.generate_all(
        "./generated/bmp_stack",
        "./generated/mask_stack",
        "./generated/voronoi_point_stack",
        "./generated/voronoi_map_stack",
        "./generated/voronoi_reconstruction_stack",
        "./generated")
    

if __name__ == '__main__':
    main()