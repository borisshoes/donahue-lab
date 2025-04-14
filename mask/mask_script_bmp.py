import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def refine_mask(data_folder, mask_folder, output_folder, save_examples = False, example_slice = 47):
    # Ensure mask folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Load BMP files
    image_stack = []
    mask_stack = []
    padding_size = 30
    for filename in sorted(os.listdir(data_folder)):
        if filename.endswith(".bmp"):
            bmp_path = os.path.join(data_folder, filename)
            bmp_image = cv2.imread(bmp_path, cv2.IMREAD_GRAYSCALE)
            
            # Add black padding around the image
            padded_image = cv2.copyMakeBorder(bmp_image, padding_size, padding_size, padding_size, padding_size, cv2.BORDER_CONSTANT, value=0)
            image_stack.append(padded_image)

    for filename in sorted(os.listdir(mask_folder)):
        if filename.endswith(".bmp"):
            bmp_path = os.path.join(mask_folder, filename)
            bmp_image = cv2.imread(bmp_path, cv2.IMREAD_GRAYSCALE)
            
            # Add black padding around the image
            padded_image = cv2.copyMakeBorder(bmp_image, padding_size, padding_size, padding_size, padding_size, cv2.BORDER_CONSTANT, value=0)
            mask_stack.append(padded_image)
    
    # Initialize mask and overlay stacks
    num_images = len(image_stack)
    index = 0
    step = 0

    # Process each image to generate masks and overlays
    for i,img in enumerate(image_stack):
        if (100*(index)/num_images)//10 != (100*(index+1)/num_images)//10:
            print(f"{index}/{num_images}")
        index += 1

        # Get the corresponding mask from mask_stack
        mask = mask_stack[i]

        # Threshold the image from image_stack to obtain a binary image.
        # Any pixel black pixel will stay black, all else is white.
        _, img_thresh = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
        if save_examples and index == example_slice:
            cv2.imwrite(f"refine_step-{step}.png", img_thresh)
            step += 1

        # Threshold the mask from mask_stack to obtain a binary image.
        # Any white pixel will stay white, all else is black.
        _, mask_thresh = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
        if save_examples and index == example_slice:
            cv2.imwrite(f"refine_step-{step}.png", mask_thresh)
            step += 1

        # Get the pixels in the data, but not the mask
        excluded = cv2.bitwise_and(img_thresh, cv2.bitwise_not(mask_thresh))
        if save_examples and index == example_slice:
            cv2.imwrite(f"refine_step-{step}.png", excluded)
            step += 1
        
        # Dilate the excluded pixels to make them more prominent
        dilated_excluded = cv2.dilate(excluded, np.ones((7, 7), np.uint8), iterations=1)
        if save_examples and index == example_slice:
            cv2.imwrite(f"refine_step-{step}.png", dilated_excluded)
            step += 1

        # Combine the original mask and the excluded pixels, ensuring that any white pixel in the image becomes white in the mask.
        combined_mask = cv2.bitwise_or(mask_thresh, dilated_excluded)
        if save_examples and index == example_slice:
            cv2.imwrite(f"refine_step-{step}.png", combined_mask)
            step += 1

        # Apply a morphological closing operation to smooth and confine the blob's shape.
        # An elliptical kernel helps in maintaining a smooth contour.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40, 40))
        smoothed_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        if save_examples and index == example_slice:
            cv2.imwrite(f"refine_step-{step}.png", smoothed_mask)
            step += 1

        # # Apply a small dilation to add white pixels along the edges.
        # kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # dilated_mask = cv2.dilate(smoothed_mask, kernel_dilate, iterations=1)
        # if save_examples and index == example_slice:
        #     cv2.imwrite(f"refine_step-{step}.png", dilated_mask)
        #     step += 1

        # Gaussian Blur to smooth edges
        blurred_mask = cv2.GaussianBlur(smoothed_mask, (75, 75), 0)
        if save_examples and index == example_slice:
            cv2.imwrite(f"refine_step-{step}.png", blurred_mask)
            step += 1

        # Linearly scale brightness so that the brightest pixel becomes 255.
        max_val = blurred_mask.max()
        if max_val > 0:
            scaled_mask = (blurred_mask.astype(np.float32) * (255.0 / max_val)).astype(np.uint8)
        else:
            scaled_mask = blurred_mask.copy()
        if save_examples and index == example_slice:
            cv2.imwrite(f"refine_step-{step}.png", scaled_mask)
            step += 1

        # Threshold after blur to retain binary form
        _, binary_mask = cv2.threshold(scaled_mask, 128, 255, cv2.THRESH_BINARY)
        if save_examples and index == example_slice:
            cv2.imwrite(f"refine_step-{step}.png", binary_mask)
            step += 1

        # Combine the binary mask with the image pixels ensuring we only add white pixels.
        added_mask = cv2.bitwise_or(binary_mask, img_thresh)
        if save_examples and index == example_slice:
            cv2.imwrite(f"refine_step-{step}.png", added_mask)
            step += 1

        # Apply a second morphological closing using a smaller elliptical kernel to further smooth the boundary.
        kernel_close2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
        closed_mask = cv2.morphologyEx(added_mask, cv2.MORPH_CLOSE, kernel_close2)
        if save_examples and index == example_slice:
            cv2.imwrite(f"refine_step-{step}.png", closed_mask)
            step += 1

        # Update the mask_stack with the processed (extended and smoothed) mask.
        mask_stack[i] = closed_mask
        
        # Crop final mask back to the original image size
        cropped_mask = mask_stack[i][padding_size:-padding_size, padding_size:-padding_size]

        # Save mask as BMP file
        mask_filename = os.path.join(output_folder, f"mask_{index:03d}.bmp")
        cv2.imwrite(mask_filename, cropped_mask)
        mask_stack[i] = cropped_mask
    return ([img[padding_size:-padding_size, padding_size:-padding_size] for img in image_stack], mask_stack)




def make_masks(input_folder, output_folder, save_examples = False, example_slice = 47):
    # Ensure mask folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Load BMP files
    image_stack = []
    mask_stack = []
    padding_size = 30
    for filename in sorted(os.listdir(input_folder)):
        if filename.endswith(".bmp"):
            bmp_path = os.path.join(input_folder, filename)
            bmp_image = cv2.imread(bmp_path, cv2.IMREAD_GRAYSCALE)
            
            # Add black padding around the image
            padded_image = cv2.copyMakeBorder(bmp_image, padding_size, padding_size, padding_size, padding_size, cv2.BORDER_CONSTANT, value=0)
            image_stack.append(padded_image)

    # Initialize mask and overlay stacks
    num_images = len(image_stack)
    index = 0
    step = 0

    # Process each image to generate masks and overlays
    for img in image_stack:
        if (100*(index)/num_images)//10 != (100*(index+1)/num_images)//10:
            print(f"{index}/{num_images}")
        index += 1

        # Binary Thresholding
        _, start_img = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
        if save_examples and index == example_slice:
            cv2.imwrite(f"step-{step}.png", start_img)
            step += 1

        # Dilation to ensure all white pixels are included in the mask
        dilated = cv2.dilate(start_img, np.ones((3, 3), np.uint8), iterations=3)

        if save_examples and index == example_slice:
            cv2.imwrite(f"step-{step}.png", dilated)
            step += 1

        # Morphological Gradient to get edges
        gradient = cv2.morphologyEx(dilated, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))

        if save_examples and index == example_slice:
            cv2.imwrite(f"step-{step}.png", gradient)
            step += 1

        # Gaussian Blur to smooth edges
        blurred_img = cv2.GaussianBlur(gradient, (5, 5), 0)

        if save_examples and index == example_slice:
            cv2.imwrite(f"step-{step}.png", blurred_img)
            step += 1

        # Morphological Closing to fill in gaps within the porous structure
        closed = cv2.morphologyEx(blurred_img, cv2.MORPH_CLOSE, np.ones((17, 17), np.uint8))

        if save_examples and index == example_slice:
            cv2.imwrite(f"step-{step}.png", closed)
            step += 1

        # Create Mask using contours
        mask = np.zeros_like(img)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mask, contours, -1, color=255, thickness=cv2.FILLED)

        if save_examples and index == example_slice:
            cv2.imwrite(f"step-{step}.png", mask)
            step += 1

        # Gaussian Blur to smooth edges
        smoothed_mask = cv2.GaussianBlur(mask, (75, 75), 0)

        if save_examples and index == example_slice:
            cv2.imwrite(f"step-{step}.png", smoothed_mask)
            step += 1

        # Threshold after blur to retain binary form
        _, blurred_img = cv2.threshold(smoothed_mask, 127, 255, cv2.THRESH_BINARY)

        if save_examples and index == example_slice:
            cv2.imwrite(f"step-{step}.png", blurred_img)
            step += 1

        # Erode to make mask boundaries hug the edge of the image data
        final_mask = cv2.erode(blurred_img, np.ones((6, 6), np.uint8)  , cv2.BORDER_REFLECT) 

        if save_examples and index == example_slice:
            cv2.imwrite(f"step-{step}.png", final_mask)
            step += 1

        # Crop final mask back to the original image size
        cropped_mask = final_mask[padding_size:-padding_size, padding_size:-padding_size]

        # Save mask as BMP file
        mask_filename = os.path.join(output_folder, f"mask_{index:03d}.bmp")
        cv2.imwrite(mask_filename, cropped_mask)
        mask_stack.append(cropped_mask)
    return ([img[padding_size:-padding_size, padding_size:-padding_size] for img in image_stack], mask_stack)

        

def compare_mask(images, masks, title = "Mask Comparison"):
    mask_stack = []
    overlay_stack = []
    for img, mask in zip(images, masks):
        # Otsu’s Thresholding for binary image comparison
        _, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, mask_bin = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)

        # Create overlay with white for overlap, gray for mask-only regions, and black elsewhere
        overlap = cv2.bitwise_and(img_bin, mask_bin)  # Areas where both DICOM image and mask are white
        mask_only = cv2.bitwise_and(mask_bin, cv2.bitwise_not(img_bin))  # Areas where only mask is white
        image_only = cv2.bitwise_and(img_bin, cv2.bitwise_not(mask_bin))  # Areas where only image is white

        # Assign pixel intensities for overlay
        overlay = np.zeros_like(img_bin)
        overlay[overlap > 0] = 255  # White for overlap
        overlay[mask_only > 0] = 127  # Gray for mask-only areas
        overlay[image_only > 0] = 80 # Dark Gray for image-only areas

        # Append the mask and overlay to stacks
        mask_stack.append(mask)
        overlay_stack.append(overlay)

    # Set up interactive viewer with matplotlib
    fig, (img_ax, mask_ax, overlay_ax) = plt.subplots(1, 3, figsize=(15, 5))
    plt.subplots_adjust(bottom=0.2)

    # Initialize display with the first slice
    img_ax.imshow(images[0], cmap="gray")
    img_ax.set_title("Image Slice")
    img_ax.axis("off")

    mask_ax.imshow(mask_stack[0], cmap="gray")
    mask_ax.set_title("Mask Slice")
    mask_ax.axis("off")

    overlay_ax.imshow(overlay_stack[0], cmap="gray")
    overlay_ax.set_title("Overlay (White=Overlap, Light Gray=Mask Only, Dark Gray=Image Only)")
    overlay_ax.axis("off")

    # Add a slider for scrolling through slices
    ax_slider = plt.axes([0.25, 0.05, 0.5, 0.03], facecolor="lightgoldenrodyellow")
    slider = Slider(ax_slider, "Slice", 0, len(images) - 1, valinit=0, valstep=1)
    update = lambda val: [ax.imshow(stack[int(slider.val)], cmap="gray") for ax, stack in zip((img_ax, mask_ax, overlay_ax), (images, mask_stack, overlay_stack))] or fig.canvas.draw_idle()
    slider.on_changed(update)

    plt.title(title)
    plt.show()

if __name__ == "__main__":
    folder_path = "C:/Users/Boris/Desktop/Classwork Homework/UMass/Donahue Lab/Python Scripts/BHS6 Bitmaps - Cleaned"
    mask_folder_path = "C:/Users/Boris/Desktop/Classwork Homework/UMass/Donahue Lab/Python Scripts/mask/BHS6 Masks/2d_mask"

    images, masks = make_masks(folder_path, mask_folder_path)
    compare_mask(images, masks)

    