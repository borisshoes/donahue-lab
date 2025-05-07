from modules import re_axis_data, mask_script_3d, mask_script_bmp, distance_calc, binarize_data

if __name__ == "__main__":
    bitmap_folder = "./BHS6 Bitmaps"
    output_folder = "./generated"
    reaxis_folder = output_folder+"/reaxis"
    masks_folder = output_folder+"/masks"
    dist_map_folder = output_folder+"/edge_distance_map"
    edge_folder = output_folder+"/shell_stack"
    binary_data_folder = output_folder+"/binary_bmps"
    dims = [1.0,0.7421875,0.7421875]
    re_axis_data.reorient_stack(bitmap_folder,reaxis_folder)
    mask_script_3d.generate_and_combine_masks(reaxis_folder, masks_folder)
    mask_script_3d.compare_masks(reaxis_folder, masks_folder)
    binarize_data.convert_ct_images_to_binary(bitmap_folder,binary_data_folder)
    distance_calc.calculate_bone_fraction_curve(masks_folder+"/final_mask", binary_data_folder, dist_map_folder, edge_folder, dims)

    