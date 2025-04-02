import os
import cv2
import numpy as np
import pydicom
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Set folder path for DICOM files
folder_path = "C:\\Users\\Boris\\Desktop\\Classwork Homework\\UMass\\Donahue Lab\\Downloaded Data\\test_DICOM (ROI)"
#folder_path = "C:\\Users\\Boris\\Desktop\\Classwork Homework\\UMass\\Donahue Lab\\Downloaded Data\\ROI #5 DICOM"

# Load DICOM files
image_stack = []
padding_size = 30
for filename in sorted(os.listdir(folder_path)):
    if filename.endswith(".dcm"):
        dicom_path = os.path.join(folder_path, filename)
        dicom_image = pydicom.dcmread(dicom_path).pixel_array
        # Normalize to 8-bit range if necessary
        dicom_image = cv2.normalize(dicom_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Add black padding around the image
        padded_image = cv2.copyMakeBorder(dicom_image, padding_size, padding_size, padding_size, padding_size, cv2.BORDER_CONSTANT, value=0)
        image_stack.append(padded_image)

# Initialize mask and overlay stacks
mask_stack = []
overlay_stack = []

# Process each image to generate masks and overlays
for img in image_stack:
    # Step 1: Dilation to ensure all white pixels are included in the mask
    dilated = cv2.dilate(img, np.ones((3, 3), np.uint8), iterations=3)

    if len(mask_stack) == 86:
        cv2.imwrite(f"step-1.png", dilated)

    # Step 2: Otsu’s Thresholding
    _, otsu_thresh = cv2.threshold(dilated, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if len(mask_stack) == 86:
        cv2.imwrite(f"step-2.png", otsu_thresh)

    # Step 3: Morphological Gradient to get edges
    gradient = cv2.morphologyEx(otsu_thresh, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))

    if len(mask_stack) == 86:
        cv2.imwrite(f"step-3.png", gradient)

    # Step 4: Gaussian Blur to smooth edges
    blurred_img = cv2.GaussianBlur(gradient, (5, 5), 0)

    if len(mask_stack) == 86:
        cv2.imwrite(f"step-4.png", blurred_img)

    # Step 5: Morphological Closing to fill in gaps within the porous structure
    closed = cv2.morphologyEx(blurred_img, cv2.MORPH_CLOSE, np.ones((17, 17), np.uint8))

    if len(mask_stack) == 86:
        cv2.imwrite(f"step-5.png", closed)

    # Step 6: Create Mask using contours
    mask = np.zeros_like(img)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask, contours, -1, color=255, thickness=cv2.FILLED)

    if len(mask_stack) == 86:
        cv2.imwrite(f"step-6.png", mask)

    # Step 7: Gaussian Blur to smooth edges
    smoothed_mask = cv2.GaussianBlur(mask, (75, 75), 0)

    if len(mask_stack) == 86:
        cv2.imwrite(f"step-7.png", smoothed_mask)

    # Step 8: Threshold after blur to retain binary form
    _, blurred_img = cv2.threshold(smoothed_mask, 127, 255, cv2.THRESH_BINARY)

    if len(mask_stack) == 86:
        cv2.imwrite(f"step-8.png", blurred_img)

    # Step 9: Erode to make mask boundaries hug the edge of the image data
    final_mask = cv2.erode(blurred_img, np.ones((6, 6), np.uint8)  , cv2.BORDER_REFLECT) 

    if len(mask_stack) == 86:
        cv2.imwrite(f"step-9.png", final_mask)



    # Create overlay with white for overlap, gray for mask-only regions, and black elsewhere
    overlap = cv2.bitwise_and(img, final_mask)  # Areas where both DICOM image and mask are white
    mask_only = cv2.bitwise_and(final_mask, cv2.bitwise_not(img))  # Areas where only mask is white
    image_only = cv2.bitwise_and(img, cv2.bitwise_not(final_mask))  # Areas where only image is white

    # Assign pixel intensities for overlay
    overlay = np.zeros_like(img)
    overlay[overlap > 0] = 255  # White for overlap
    overlay[mask_only > 0] = 127  # Gray for mask-only areas
    overlay[image_only > 0] = 80 # Dark Gray for image-only areas

    # Append the mask and overlay to stacks
    mask_stack.append(final_mask)
    overlay_stack.append(overlay)

# Function to update display when slider changes
def update(val):
    slice_idx = int(slider.val)
    img_ax.imshow(image_stack[slice_idx], cmap="gray")
    mask_ax.imshow(mask_stack[slice_idx], cmap="gray")
    overlay_ax.imshow(overlay_stack[slice_idx], cmap="gray")
    fig.canvas.draw_idle()

# Set up interactive viewer with matplotlib
fig, (img_ax, mask_ax, overlay_ax) = plt.subplots(1, 3, figsize=(15, 5))
plt.subplots_adjust(bottom=0.2)

# Initialize display with the first slice
img_ax.imshow(image_stack[0], cmap="gray")
img_ax.set_title("DICOM Image Slice")
img_ax.axis("off")

mask_ax.imshow(mask_stack[0], cmap="gray")
mask_ax.set_title("Mask Slice")
mask_ax.axis("off")

overlay_ax.imshow(overlay_stack[0], cmap="gray")
overlay_ax.set_title("Overlay (White=Overlap, Light Gray=Mask Only, Dark Gray=Image Only)")
overlay_ax.axis("off")

# Add a slider for scrolling through slices
ax_slider = plt.axes([0.25, 0.05, 0.5, 0.03], facecolor="lightgoldenrodyellow")
slider = Slider(ax_slider, "Slice", 0, len(image_stack) - 1, valinit=0, valstep=1)
slider.on_changed(update)

plt.show()