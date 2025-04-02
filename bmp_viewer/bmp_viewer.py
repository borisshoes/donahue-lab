import os
import numpy as np
import imageio
from vedo import Volume, Plotter
import tkinter as tk
from tkinter import filedialog

# -----------------------
# 1. Load the BMP images via a folder selection dialog
# -----------------------
root = tk.Tk()
root.withdraw()  # Hide the main Tkinter window
folder = filedialog.askdirectory(title="Select folder containing BMP images")
if not folder:
    raise ValueError("No folder selected!")

# Get sorted list of .bmp files in the folder
files = sorted([os.path.join(folder, f) for f in os.listdir(folder)
                if f.lower().endswith('.bmp')])
if not files:
    raise ValueError("No .bmp files found in the selected folder.")

# Read each image and stack them into a 3D numpy array.
slices = [imageio.imread(f) for f in files]
data = np.stack(slices, axis=0)
n_slices = data.shape[0]

# -----------------------
# 2. Create the initial Volume
# -----------------------
# Create a volume from the full data stack.
vol = Volume(data)
vol.name = "volume"

# Set initial opacity value (1 is fully opaque)
opacity_value = 1.0

# Global variables to hold current slider values
current_min = 0
current_max = n_slices

# -----------------------
# 3. Define slider callback functions
# -----------------------

def update_volume():
    """
    Update the volume by zeroing out slices outside the [current_min, current_max] range.
    With the alpha transfer function mapping 0 intensity to 0 opacity, these slices are hidden.
    This keeps the overall volume's spatial location unchanged.
    """
    global vol, data, current_min, current_max, opacity_value, plt

    # Make a copy of the full data and set slices outside the desired range to 0.
    new_data = data.copy()
    if current_min > 0:
        new_data[:current_min, :, :] = 0
    if current_max < n_slices:
        new_data[current_max:, :, :] = 0

    # Create a new volume with the modified data.
    new_vol = Volume(new_data)
    new_vol.name = "volume"
    # Map 0 intensity to 0 opacity and the max intensity to the chosen opacity.
    new_vol.alpha([(0, 0), (new_data.max(), opacity_value)])
    plt.remove("volume")
    plt.add(new_vol)
    vol = new_vol
    plt.render()


def slider_min_callback(widget, event):
    """Callback for the minimum slice slider."""
    global current_min, current_max
    value = int(widget.GetRepresentation().GetValue())
    current_min = value
    if current_min >= current_max:
        current_min = current_max - 1
    update_volume()

def slider_max_callback(widget, event):
    """Callback for the maximum slice slider."""
    global current_min, current_max
    value = int(widget.GetRepresentation().GetValue())
    current_max = value
    if current_max <= current_min:
        current_max = current_min + 1
    update_volume()

def slider_opacity_callback(widget, event):
    """
    Callback for the opacity slider.
    The slider now outputs a value in the range 0-100. We convert that to a 0-1 float
    and update the transfer function so that 0 intensity remains transparent.
    """
    global opacity_value, vol, data, current_min, current_max, plt
    value = widget.GetRepresentation().GetValue()
    opacity_value = value / 100.0
    # Prepare a modified version of data for the current visible range.
    current_data = data.copy()
    current_data[:current_min, :, :] = 0
    current_data[current_max:, :, :] = 0
    vol.alpha([(0, 0), (current_data.max(), opacity_value)])
    plt.render()

# -----------------------
# 4. Create the Plotter and add sliders
# -----------------------
plt = Plotter(title="3D .bmp Stack Viewer", bg='white')

# Add the initial volume to the scene
plt.add(vol)

# Slider for minimum slice index
plt.add_slider(
    slider_min_callback,
    0,
    n_slices - 1,
    value=0,
    pos=[(0.1, 0.1), (0.4, 0.1)],
    title="Min Slice"
)

# Slider for maximum slice index
plt.add_slider(
    slider_max_callback,
    0,
    n_slices - 1,
    value=n_slices - 1,
    pos=[(0.6, 0.1), (0.9, 0.1)],
    title="Max Slice"
)

# Slider for overall opacity with range 0-100 (to represent percent)
slider_opacity = plt.add_slider(
    slider_opacity_callback,
    0,
    100,
    value=100,
    pos=[(0.1, 0.9), (0.4, 0.9)],
    title="Opacity"
)
# Set the slider's label format so it shows a percent sign.
slider_opacity.GetRepresentation().SetLabelFormat('%1.0f%%')

# -----------------------
# 5. Launch the interactive viewer
# -----------------------
plt.show(interactive=True)