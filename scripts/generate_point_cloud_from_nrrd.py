import argparse, shutil, sys, os
import tkinter as tk
from tkinter import filedialog
from modules import nrrd_to_bmp
from modules import mask_blank
from modules import calc_voronoi_points
from modules.stack_utils import prep_folder

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

    folder_path = "../generated"

    nrrd_to_bmp.convert_nrrd_to_bmp(input_file, folder_path+"/bmp_stack", "last")
    mask_blank.process_images(folder_path+"/bmp_stack", folder_path+"/mask_stack")
    calc_voronoi_points.generate_all(
        folder_path+"/bmp_stack",
        folder_path+"/mask_stack",
        folder_path)
    

if __name__ == '__main__':
    main()