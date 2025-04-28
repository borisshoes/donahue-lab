from modules import re_axis_data
from modules import mask_script_3d
from modules import mask_script_bmp

if __name__ == "__main__":
    bitmap_folder = "./BHS6 Bitmaps - Cleaned"
    #bitmap_folder = "./nrrd_to_bmp/output"
    reaxis_folder = "./reaxis/BHS6_ReAxis_auto"
    masks_folder = "./mask/BHS6 Masks/3d_masks_auto"
    re_axis_data.reorient_stack(bitmap_folder,reaxis_folder)
    mask_script_3d.generate_and_combine_masks(reaxis_folder, masks_folder)
    mask_script_3d.compare_masks(reaxis_folder, masks_folder)
    