# iterate_on_single_patch_with_iterative_inference.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from scipy.ndimage import gaussian_filter
from scipy.io import loadmat


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
IMG_MEAN_CPU = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMG_STD_CPU = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# --- Helper Functions ---
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

def apply_nms_to_points(detections_with_scores, radius):
    """
    Applies Non-Maximum Suppression to a list of detected points.
    Args:
        detections_with_scores: A list of tuples, where each tuple is ((x, y), score).
        radius: The NMS radius. Points within this distance of a higher-scored point will be suppressed.
    Returns:
        A list of [x, y] coordinates (as lists) of the points kept after NMS.
    """
    if not detections_with_scores:
        return []

    # Extract points and scores
    points_list = [det[0] for det in detections_with_scores] # list of (x,y) tuples/lists
    scores_list = [det[1] for det in detections_with_scores] # list of scores

    if not points_list:
        return []

    points_np = np.array(points_list, dtype=np.float32) # Shape (N, 2)
    scores_np = np.array(scores_list, dtype=np.float32) # Shape (N,)

    # Sort by scores in descending order
    # 'order' will contain original indices of points, sorted by score
    order = scores_np.argsort()[::-1]
    
    kept_original_indices = []
    # Boolean array indicating if a point (at its original index) is suppressed
    suppressed_flags = np.zeros(len(points_np), dtype=bool)

    for i_order_idx in range(len(order)):
        original_idx_i = order[i_order_idx] # Original index of the i-th point in sorted list

        if suppressed_flags[original_idx_i]:
            continue
        
        kept_original_indices.append(original_idx_i) # Keep this point
        current_point_coords = points_np[original_idx_i]
        
        for j_order_idx in range(i_order_idx + 1, len(order)):
            original_idx_j = order[j_order_idx] # Original index of the j-th point in sorted list

            if suppressed_flags[original_idx_j]:
                continue
            
            other_point_coords = points_np[original_idx_j]
            
            # Calculate squared Euclidean distance (faster than sqrt)
            dist_sq = np.sum((current_point_coords - other_point_coords)**2)
            
            if dist_sq <= radius**2:
                suppressed_flags[original_idx_j] = True # Suppress this point
                
    # Retrieve the coordinates of the kept points using their original indices
    final_kept_points_coords = points_np[kept_original_indices].tolist()
    return final_kept_points_coords

def load_gt_points(gt_path):
    if not os.path.exists(gt_path): return np.array([])
    try:
        mat_data = loadmat(gt_path)
        if 'image_info' in mat_data: return mat_data['image_info'][0, 0][0, 0][0].astype(np.float32)
        if 'annPoints' in mat_data: return mat_data['annPoints'].astype(np.float32)
        for _, value in mat_data.items():
            if isinstance(value, np.ndarray) and value.ndim == 2 and value.shape[1] == 2: return value.astype(np.float32)
        return np.array([])
    except Exception: return np.array([])

pred_points=[]
true_points=[]
# --- Main Script Logic ---
for i in range(316):
    print(i+1)
    im_num=i+1
    # Path to the image
    image_file_path = r"C:\Users\Mehmet_Postdoc\Desktop\datasets_for_experiments\ShanghaiTech_Crowd_Counting_Dataset\part_B_final\test_data\images\IMG_%d.jpg"%im_num #Example Path
    gt_file_path = r"C:\Users\Mehmet_Postdoc\Desktop\datasets_for_experiments\ShanghaiTech_Crowd_Counting_Dataset\part_B_final\test_data\ground_truth\GT_IMG_%d.mat"%im_num #Example Path
    # Load ground truth points if available
    gt_points = load_gt_points(gt_file_path)
    # Load the image
    image = cv2.imread(image_file_path)
    if image is None:
        print(f"Error: Could not load image from {image_file_path}")
        exit()

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_height, img_width, _ = image_rgb.shape
    # print(f"Original image shape: ({img_height}, {img_width}, {image_rgb.shape[2]})")

    if img_height == 0 or img_width == 0:
        print(f"Error: Image loaded from {image_file_path} has zero height or width.")
        exit()


    # --- Patch Extraction Parameters ---
    PATCH_EXTRACT_HEIGHT = 224
    PATCH_EXTRACT_WIDTH = 224
    STRIDE_H = PATCH_EXTRACT_HEIGHT
    STRIDE_W = PATCH_EXTRACT_WIDTH
    # print(f"Patch extraction settings: Size=({PATCH_EXTRACT_HEIGHT}x{PATCH_EXTRACT_WIDTH}), Stride=({STRIDE_H}x{STRIDE_W})")
    # print(f"Model input size: {MODEL_INPUT_SIZE}x{MODEL_INPUT_SIZE}")

    # --- Calculate Patch Coordinates ---
    r_positions = []
    if img_height > 0:
        if img_height <= PATCH_EXTRACT_HEIGHT: r_positions = [0]
        else:
            r_positions = list(range(0, img_height - PATCH_EXTRACT_HEIGHT + 1, STRIDE_H))
            if not r_positions or r_positions[-1] < img_height - PATCH_EXTRACT_HEIGHT:
                r_positions.append(img_height - PATCH_EXTRACT_HEIGHT) # Ensure last part of image is covered
        r_positions = sorted(list(set(r_positions)))

    c_positions = []
    if img_width > 0:
        if img_width <= PATCH_EXTRACT_WIDTH: c_positions = [0]
        else:
            c_positions = list(range(0, img_width - PATCH_EXTRACT_WIDTH + 1, STRIDE_W))
            if not c_positions or c_positions[-1] < img_width - PATCH_EXTRACT_WIDTH:
                c_positions.append(img_width - PATCH_EXTRACT_WIDTH) # Ensure last part of image is covered
        c_positions = sorted(list(set(c_positions)))

    if not r_positions or not c_positions:
        print("Error: Could not determine patch coordinates. Image might be too small or patch/stride settings problematic.")
        if img_height > 0 and img_width > 0 :
            print("Attempting a single patch from (0,0).")
            r_positions = [0]; c_positions = [0]
        else: exit()

    # --- Extract and Prepare Patches ---
    patches_to_process = []
    for r_start in r_positions:
        for c_start in c_positions:
            patch_orig_np = image_rgb[r_start : r_start + PATCH_EXTRACT_HEIGHT, c_start : c_start + PATCH_EXTRACT_WIDTH]
            patch_resized_np = cv2.resize(patch_orig_np, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))
            patch_tensor_np_float = patch_resized_np.astype(np.float32) / 255.0
            patch_tensor_chw = torch.from_numpy(patch_tensor_np_float).permute(2, 0, 1)
            patch_tensor_norm = (patch_tensor_chw - IMG_MEAN_CPU) / IMG_STD_CPU
            patch_tensor_batch = patch_tensor_norm.unsqueeze(0).to(DEVICE)
            patches_to_process.append({
                "tensor": patch_tensor_batch, "r_orig": r_start, "c_orig": c_start,
            })
    # print(f"Number of patches to process: {len(patches_to_process)}")
    if not patches_to_process: print("No patches were extracted. Exiting."); exit()

    # # --- Visualize Extracted Patches (Optional) ---
    # print("Visualizing extracted patches...")
    # image_with_patches_viz = image_rgb.copy()
    # for p_data in patches_to_process:
    #     r, c = p_data["r_orig"], p_data["c_orig"]
    #     cv2.rectangle(image_with_patches_viz, (c, r), (c + PATCH_EXTRACT_WIDTH, r + PATCH_EXTRACT_HEIGHT), (0, 255, 0), 2)
    # plt.figure(figsize=(10, 10 * img_height/img_width if img_width > 0 else 10))
    # plt.imshow(image_with_patches_viz)
    # plt.title(f"Extracted Patches for Processing ({len(patches_to_process)} patches)")
    # plt.axis('off'); plt.show()

    # --- Load Model ---
    if not os.path.exists(BEST_MODEL_PATH):
        print(f"CRITICAL ERROR: Model weights not found at '{BEST_MODEL_PATH}'.")
        model_loaded_successfully = False
        if BEST_MODEL_PATH == "path/to/your/best_model.pth":
            print("Using placeholder path. Proceeding with random weights.")
        else: exit()
    else: model_loaded_successfully = True

    model = VGG19FPNASPP().to(DEVICE)
    if model_loaded_successfully:
        try:
            model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
            # print(f"Model loaded successfully from {BEST_MODEL_PATH} to {DEVICE}.")
        except Exception as e:
            print(f"Error loading model: {e}. Proceeding with uninitialized model.")
    else: print("Proceeding with randomly initialized model.")
    model.eval()

    # --- Iterative Inference Over All Patches ---
    all_detections_full_image = [] # Stores ((x_full, y_full), score)
    num_iterations_per_patch = 100
    early_stop_confidence = 0.01
    min_iters_before_early_stop = 10

    # print(f"\nStarting iterative inference on {len(patches_to_process)} patches...")
    for patch_idx, patch_data in enumerate(patches_to_process):
        # print(f"  Processing patch {patch_idx + 1}/{len(patches_to_process)} at original (r,c): ({patch_data['r_orig']}, {patch_data['c_orig']})")
        image_patch_tensor_batch = patch_data["tensor"]
        r_offset, c_offset = patch_data["r_orig"], patch_data["c_orig"]
        predicted_points_on_patch_local = []
        
        for iter_num in range(num_iterations_per_patch):
            current_input_psf_np = create_input_psf_from_points(
                predicted_points_on_patch_local,
                (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), GT_PSF_SIGMA)
            current_input_psf_tensor = torch.from_numpy(current_input_psf_np).float().unsqueeze(0).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                pred_out_psf, pred_conf_logits = model(image_patch_tensor_batch, current_input_psf_tensor)

            confidence_score = torch.sigmoid(pred_conf_logits).item()
            output_psf_np = pred_out_psf.squeeze().cpu().numpy()
            max_yx = np.unravel_index(np.argmax(output_psf_np), output_psf_np.shape)
            pred_y_local, pred_x_local = max_yx[0], max_yx[1]
            predicted_points_on_patch_local.append((pred_x_local, pred_y_local))

            scale_x = PATCH_EXTRACT_WIDTH / MODEL_INPUT_SIZE
            scale_y = PATCH_EXTRACT_HEIGHT / MODEL_INPUT_SIZE
            pred_x_full = np.clip(int(round((pred_x_local + 0.5) * scale_x + c_offset)), 0, img_width - 1)
            pred_y_full = np.clip(int(round((pred_y_local + 0.5) * scale_y + r_offset)), 0, img_height - 1)
            all_detections_full_image.append(((pred_x_full, pred_y_full), confidence_score))
            
            if iter_num > min_iters_before_early_stop and confidence_score < early_stop_confidence:
                print(f"    Patch {patch_idx+1} Iter {iter_num + 1}: Early stopping (Conf: {confidence_score:.4f}).")
                break
        # print(f"    Patch {patch_idx+1} finished after {iter_num+1} iterations.")

    # print("\nIterative inference for all patches finished.")
    # print(f"Total raw predictions made on full image: {len(all_detections_full_image)}")

    # --- Filter points by confidence ---
    CONFIDENCE_THRESHOLD_FOR_DISPLAY = 0.5
    NMS_RADIUS = 7 # NMS radius in pixels

    confident_detections_with_scores = []
    for point_coords_full, score in all_detections_full_image:
        if score >= CONFIDENCE_THRESHOLD_FOR_DISPLAY:
            confident_detections_with_scores.append((point_coords_full, score))

    num_confident_points = len(confident_detections_with_scores)
    # print(f"Number of points on full image with confidence >= {CONFIDENCE_THRESHOLD_FOR_DISPLAY:.2f}: {num_confident_points}")

    # --- Apply NMS ---
    if num_confident_points > 0:
        # print(f"Applying NMS with radius {NMS_RADIUS}px...")
        final_nms_points_coords = apply_nms_to_points(confident_detections_with_scores, NMS_RADIUS)
        num_points_after_nms = len(final_nms_points_coords)
        print(f"Number of points after NMS: {num_points_after_nms}")
    else:
        final_nms_points_coords = []
        num_points_after_nms = 0
        print("No confident points to apply NMS.")

    # # --- Final Plotting of Detections on Full Image ---
    # plt.figure(figsize=(12, 12 * img_height/img_width if img_width > 0 else 12))
    # plt.imshow(image_rgb)
    # plot_title = (f"Full Image Detections ({num_points_after_nms} points)\n"
    #               f"Conf >= {CONFIDENCE_THRESHOLD_FOR_DISPLAY:.2f}, NMS Radius = {NMS_RADIUS}px")

    # if final_nms_points_coords:
    #     # final_nms_points_coords is a list of lists, e.g., [[x1, y1], [x2, y2], ...]
    #     # Convert to NumPy array for easier slicing for scatter plot
    #     final_points_np_array = np.array(final_nms_points_coords)
    #     if final_points_np_array.size > 0: # Ensure array is not empty
    #         plt.scatter(final_points_np_array[:, 0], final_points_np_array[:, 1], s=15, c='yellow', marker='x', label=f'Detections after NMS')
    #     # plt.legend(loc='best') # Legend can be noisy
    # else:
    #     plot_title += f"\nNo points after Conf >= {CONFIDENCE_THRESHOLD_FOR_DISPLAY:.2f} & NMS (R={NMS_RADIUS})"
    #     plt.text(img_width / 2, img_height / 2,
    #              f"No points found meeting criteria\n(Conf >= {CONFIDENCE_THRESHOLD_FOR_DISPLAY:.2f}, NMS Radius={NMS_RADIUS}px)",
    #              horizontalalignment='center', verticalalignment='center',
    #              fontsize=12, color='red', bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))

    # plt.title(plot_title)
    # plt.axis('off')
    # plt.tight_layout()
    # plt.show()

    # print("Script finished.")
    print(f"{len(gt_points)} ground truth points")

    Abs_diff = np.abs(num_points_after_nms - len(gt_points))
    print(f"Absolute difference between predicted and true points: {Abs_diff}")

    pred_points.append(num_points_after_nms)
    true_points.append(len(gt_points))

print("Predicted points:", pred_points)
print("True points:", true_points)
diff=np.abs(np.array(pred_points)-np.array(true_points))
print(np.mean(diff))