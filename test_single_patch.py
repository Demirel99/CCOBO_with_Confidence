# iterate_on_single_patch_with_iterative_inference.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from scipy.ndimage import gaussian_filter

# --- Import from your project's config and model files ---
try:
    from config import (
        DEVICE, MODEL_INPUT_SIZE, BEST_MODEL_PATH, GT_PSF_SIGMA
        # Add IMG_MEAN_CPU, IMG_STD_CPU here if they are in your config.py
    )
    print("Successfully imported configuration from config.py")
except ImportError:
    print("ERROR: Could not import from config.py. Please ensure it exists in the same directory and defines:")
    print("  DEVICE, MODEL_INPUT_SIZE, BEST_MODEL_PATH, GT_PSF_SIGMA")
    print("Using fallback default values (may not be correct for your model).")
    # Fallback defaults (adjust if necessary, but ideally fix config.py)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_INPUT_SIZE = 224
    BEST_MODEL_PATH = "path/to/your/best_model.pth" # THIS WILL LIKELY FAIL - FIX config.py
    GT_PSF_SIGMA = 4.0
    # exit() # Optionally exit if config is crucial

try:
    from model import VGG19FPNASPP
    print("Successfully imported VGG19FPNASPP from model.py")
except ImportError:
    print("ERROR: Could not import VGG19FPNASPP from model.py.")
    print("Please ensure model.py is in the same directory and defines the VGG19FPNASPP class.")
    exit() # Critical, cannot proceed without the model definition

# ImageNet Mean/Std for Normalization/Unnormalization
# If these are defined in your config.py, you can remove these lines
# and import them from config above.
IMG_MEAN_CPU = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMG_STD_CPU = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# --- Helper Functions (from test_full_image_patched_inference.py) ---
def create_input_psf_from_points(points_list, shape, sigma):
    h, w = shape
    delta_map = np.zeros((h, w), dtype=np.float32)
    if not points_list:
        return delta_map
    for x, y in points_list:
        x_coord = np.clip(int(round(x)), 0, w - 1)
        y_coord = np.clip(int(round(y)), 0, h - 1)
        delta_map[y_coord, x_coord] += 1.0
    input_psf = gaussian_filter(delta_map, sigma=sigma, order=0, mode='constant', cval=0.0)
    max_val = np.max(input_psf)
    if max_val > 1e-7:
        input_psf /= max_val
    return input_psf

# --- Main Script Logic ---

# Path to the image
image_file_path = r"C:\Users\Mehmet_Postdoc\Desktop\datasets_for_experiments\ShanghaiTech_Crowd_Counting_Dataset\part_A_final\test_data\images\IMG_115.jpg" #Example Path

# Load the image
image = cv2.imread(image_file_path)
if image is None:
    print(f"Error: Could not load image from {image_file_path}")
    exit()

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print("Original image shape:", image_rgb.shape)

initial_patch_height = 224
initial_patch_width = 224
img_height, img_width, _ = image_rgb.shape
patches_orig_size = []

for r in range(0, img_height - initial_patch_height + 1, initial_patch_height):
    for c in range(0, img_width - initial_patch_width + 1, initial_patch_width):
        patch = image_rgb[r:r + initial_patch_height, c:c + initial_patch_width]
        patches_orig_size.append(patch)

print(f"Number of {initial_patch_width}x{initial_patch_height} patches extracted: {len(patches_orig_size)}")

if not patches_orig_size:
    print("No patches were extracted. Exiting.")
    exit()

patches_resized = [cv2.resize(patch, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)) for patch in patches_orig_size]
print(f"Resized patches to {MODEL_INPUT_SIZE}x{MODEL_INPUT_SIZE}.")

patch_index_to_test = 1
if patch_index_to_test >= len(patches_resized):
    print(f"Error: Patch index {patch_index_to_test} is out of bounds. Max index is {len(patches_resized)-1}.")
    print(f"Using patch index 0 instead.")
    patch_index_to_test = 0
    if not patches_resized:
        print("No patches available. Exiting.")
        exit()

selected_patch_np = patches_resized[patch_index_to_test]

if not os.path.exists(BEST_MODEL_PATH):
    print(f"CRITICAL ERROR: Model weights not found at '{BEST_MODEL_PATH}' (from config.py).")
    print("Please ensure BEST_MODEL_PATH in config.py is correct.")

    if BEST_MODEL_PATH == "path/to/your/best_model.pth": # Specific check for default placeholder
        print("Using a placeholder model path. Inference will likely fail if not updated.")
        print("Attempting to proceed without loading weights (model will be initialized randomly).")
        model_loaded_successfully = False
    else:
        exit()
else:
    model_loaded_successfully = True


model = VGG19FPNASPP().to(DEVICE)

if model_loaded_successfully:
    try:
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
        print(f"Model loaded successfully from {BEST_MODEL_PATH} to {DEVICE}.")
    except Exception as e:
        print(f"Error loading model state_dict from {BEST_MODEL_PATH}: {e}")
        print("Ensure the model definition in model.py matches the weights file.")
        print("Proceeding with an uninitialized model for structure testing if possible.")
        # exit() # Critical for actual inference
else:
     print("Proceeding with randomly initialized model (weights not loaded).")

model.eval()

patch_tensor_np_float = selected_patch_np.astype(np.float32) / 255.0
patch_tensor_chw = torch.from_numpy(patch_tensor_np_float).permute(2, 0, 1) # This is on CPU

# Perform normalization on CPU, as IMG_MEAN_CPU and IMG_STD_CPU are CPU tensors
patch_tensor_norm = (patch_tensor_chw - IMG_MEAN_CPU) / IMG_STD_CPU

# Now, add batch dimension and move the normalized tensor to the target DEVICE
image_patch_tensor_batch = patch_tensor_norm.unsqueeze(0).to(DEVICE)

# Prepare patch for display (unnormalize) - ensure tensors are on CPU for unnormalization and display
display_patch_tensor_cpu = image_patch_tensor_batch.squeeze(0).cpu() * IMG_STD_CPU + IMG_MEAN_CPU
display_patch_np = np.clip(display_patch_tensor_cpu.permute(1, 2, 0).numpy(), 0, 1)


# --- Iterative Inference Loop ---
print(f"\nStarting iterative inference on patch {patch_index_to_test}...")
predicted_points_on_patch = [] # Stores (x,y) for PSF generation in next iteration
all_predictions_with_scores = [] # Stores ((x,y), score) for all predictions
num_iterations = 700 # Number of points to attempt to predict

for iter_num in range(num_iterations):
    current_input_psf_np = create_input_psf_from_points(
        predicted_points_on_patch, # Uses all points found so far
        shape=(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
        sigma=GT_PSF_SIGMA
    )
    current_input_psf_tensor = torch.from_numpy(current_input_psf_np).float().unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        predicted_output_psf_tensor, predicted_confidence_logits = model(image_patch_tensor_batch, current_input_psf_tensor)

    confidence_score = torch.sigmoid(predicted_confidence_logits).item()
    output_psf_np = predicted_output_psf_tensor.squeeze().cpu().numpy()
    max_yx = np.unravel_index(np.argmax(output_psf_np), output_psf_np.shape)
    pred_y, pred_x = max_yx[0], max_yx[1]

    # Append to list for next iteration's PSF
    predicted_points_on_patch.append((pred_x, pred_y))
    # Store this prediction with its score for final analysis/display
    all_predictions_with_scores.append(((pred_x, pred_y), confidence_score))

    print(f"  Iteration {iter_num + 1}/{num_iterations}: New point ({pred_x}, {pred_y}), Confidence: {confidence_score:.4f}")


print("\nIterative inference finished.")
print(f"Total raw predictions made on patch {patch_index_to_test}: {len(all_predictions_with_scores)}")

# --- Filter points for final display based on confidence ---
CONFIDENCE_THRESHOLD_FOR_DISPLAY = 0.5  # Adjust this threshold as needed
points_to_display_on_patch = []
for point_coords, score in all_predictions_with_scores:
    if score >= CONFIDENCE_THRESHOLD_FOR_DISPLAY:
        points_to_display_on_patch.append(point_coords)

print(f"Number of points on patch {patch_index_to_test} with confidence >= {CONFIDENCE_THRESHOLD_FOR_DISPLAY:.2f}: {len(points_to_display_on_patch)}")
if points_to_display_on_patch:
    # print(f"Filtered points (x, y) for display:", points_to_display_on_patch)
    pass
else:
    print(f"No points met the confidence threshold of {CONFIDENCE_THRESHOLD_FOR_DISPLAY:.2f} for display.")


# --- Final Plotting ---
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title("Original Full Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(selected_patch_np) # Show the resized patch that was used for inference
patch_title = f"Selected Patch {patch_index_to_test} ({MODEL_INPUT_SIZE}x{MODEL_INPUT_SIZE})\n"
if points_to_display_on_patch:
    final_points_np = np.array(points_to_display_on_patch)
    plt.scatter(final_points_np[:, 0], final_points_np[:, 1], s=30, c='yellow', marker='x', label=f'Conf >= {CONFIDENCE_THRESHOLD_FOR_DISPLAY:.2f}')
    patch_title += f"{len(points_to_display_on_patch)} points shown"
    plt.legend(loc='best')
else:
    patch_title += f"No points with Conf >= {CONFIDENCE_THRESHOLD_FOR_DISPLAY:.2f}"
    # Optional: add text directly on image if no points
    plt.text(MODEL_INPUT_SIZE / 2, MODEL_INPUT_SIZE / 2,
             f"No points with\nconfidence >= {CONFIDENCE_THRESHOLD_FOR_DISPLAY:.2f}",
             horizontalalignment='center', verticalalignment='center',
             fontsize=9, color='red', bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))

plt.title(patch_title)
plt.axis('off')
plt.tight_layout()
plt.show()