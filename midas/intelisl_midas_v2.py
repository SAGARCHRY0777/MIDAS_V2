from PIL import Image
import numpy as np
import requests # Still needed if you want to download an example image initially
import torch
import os
import matplotlib.pyplot as plt # For saving images with colormaps

from transformers import DPTImageProcessor, DPTForDepthEstimation
# You'll need to have 'accelerate' installed for 'low_cpu_mem_usage=True' to work,
# especially if you're loading a large model and using specific device mapping strategies
# pip install 'accelerate>=0.26.0'

# --- Configuration ---
MODEL_NAME = "Intel/dpt-large"
MODEL_SAVE_PATH = "D:\\spa\\SFA3D\\transformers_midas\\model"
RESULTS_SAVE_PATH = "D:\\spa\\SFA3D\\transformers_midas\\test_results"
KITTI_DATA_PATH = "D:\\spa\\SFA3D\\dataset\\kitti\\training\\image_2"
NUM_SAMPLES_TO_PROCESS = 75  # Set the number of samples you want to process

# Ensure directories exist
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(RESULTS_SAVE_PATH, exist_ok=True)

# --- Determine Device (GPU or CPU) ---
# This is crucial for GPU utilization.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"CUDA memory cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")

# --- Load or Download and Save Model and Processor ---
# The processor and model are usually saved in the same directory.
# Hugging Face `from_pretrained` can load from a local path if it exists.

# Check if the model and processor are already downloaded
if os.path.exists(MODEL_SAVE_PATH) and os.path.isdir(MODEL_SAVE_PATH) and \
   os.path.exists(os.path.join(MODEL_SAVE_PATH, "config.json")): # Check for a common config file
    print(f"Loading image processor and model from local path: {MODEL_SAVE_PATH}")
    image_processor = DPTImageProcessor.from_pretrained(MODEL_SAVE_PATH)
    # When loading from a local path, `low_cpu_mem_usage=True` might not be as critical
    # as it is for initial downloads, but it generally doesn't hurt.
    model = DPTForDepthEstimation.from_pretrained(MODEL_SAVE_PATH, low_cpu_mem_usage=True)
else:
    print(f"Downloading image processor and model from Hugging Face Hub and saving to: {MODEL_SAVE_PATH}")
    image_processor = DPTImageProcessor.from_pretrained(MODEL_NAME)
    # `low_cpu_mem_usage=True` is particularly useful here to prevent out-of-memory
    # issues during the initial model loading, especially on systems with limited RAM.
    model = DPTForDepthEstimation.from_pretrained(MODEL_NAME, low_cpu_mem_usage=True)
    # Save them locally
    image_processor.save_pretrained(MODEL_SAVE_PATH)
    model.save_pretrained(MODEL_SAVE_PATH)
    print("Model and processor saved locally.")

# --- Move model to GPU (if 'cuda' device was selected) ---
# This is where the model's parameters are transferred from CPU RAM to GPU VRAM.
# This must happen BEFORE any inference.
model.to(device)
model.eval() # Set the model to evaluation mode (disables dropout, batch norm updates, etc.)

# --- Process KITTI images ---
image_files = sorted([f for f in os.listdir(KITTI_DATA_PATH) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

if not image_files:
    print(f"No image files found in {KITTI_DATA_PATH}. Please check the path and file extensions.")
    exit()

for i, image_file in enumerate(image_files):
    if i >= NUM_SAMPLES_TO_PROCESS:
        break

    image_path = os.path.join(KITTI_DATA_PATH, image_file)
    print(f"Processing image {i+1}/{min(NUM_SAMPLES_TO_PROCESS, len(image_files))}: {image_file}")

    try:
        image = Image.open(image_path).convert("RGB") # Ensure image is RGB
    except Exception as e:
        print(f"Warning: Could not read or process image {image_file}. Error: {e}. Skipping.")
        continue

    # Prepare image for the model
    # The `to(device)` here ensures the input tensor is also on the GPU.
    inputs = image_processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad(): # Disable gradient calculations for inference (saves memory and speeds up)
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # Interpolate to original size
    # Note: image.size[::-1] gives (height, width) which is what interpolate expects for `size`
    # The interpolation also happens on the GPU if `predicted_depth` is on GPU.
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )

    # Visualize and save the prediction
    # Move the tensor to CPU before converting to NumPy, as NumPy doesn't work with GPU tensors.
    output = prediction.squeeze().cpu().numpy()

    # Normalize to 0-255 and convert to uint8 for saving as an image
    # Note: For depth maps, simply scaling to 0-255 is a common visualization.
    # If you need raw float depth values, skip the formatting to uint8.
    formatted = (output * 255 / np.max(output)).astype("uint8")
    depth_image_to_save = Image.fromarray(formatted)

    # Construct the output filename
    base_name = os.path.splitext(image_file)[0]
    output_filename_png = os.path.join(RESULTS_SAVE_PATH, f"depth_{base_name}.png")

    # Save the depth map using matplotlib for colormapping (optional, but good for visualization)
    # Using 'magma' colormap for depth visualization
    plt.imsave(output_filename_png, output, cmap='magma')
    # If you just want to save the grayscale formatted image without colormap:
    # depth_image_to_save.save(output_filename_png)

    print(f"Saved processed depth map to: {output_filename_png}")

print(f"\nProcessing complete. {min(NUM_SAMPLES_TO_PROCESS, len(image_files))} images processed and saved.")