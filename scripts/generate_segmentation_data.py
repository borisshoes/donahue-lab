from modules import re_axis_data, mask_script_3d, mask_script_bmp, distance_calc, binarize_data

if __name__ == "__main__":
    bitmap_folder = "./BHS2 Bitmaps"
    output_folder = "./generated"
    reaxis_folder = output_folder+"/reaxis"
    masks_folder = output_folder+"/masks"
    binary_data_folder = output_folder+"/binary_bmps"
    dist_map_folder = output_folder+"/bone_distance_map"
    point_folder = output_folder+"/points"
    node_folder = output_folder+"/nodes"
    trace_folder = output_folder+"/traces"
    dims = [1.0,0.7395833,0.7395833]
    re_axis_data.reorient_stack(bitmap_folder,reaxis_folder)
    mask_script_3d.generate_and_combine_masks(reaxis_folder, masks_folder)
    mask_script_3d.compare_masks(reaxis_folder, masks_folder)
    binarize_data.convert_ct_images_to_binary(bitmap_folder,binary_data_folder)
    distance_calc.generate_node_stack(masks_folder+"/final_mask", binary_data_folder, dist_map_folder, point_folder, node_folder, trace_folder, dims)

    