# test_full_image_patched_inference.py
"""
Test script to run iterative inference on a full image by dividing it into
patches. For each patch, points are predicted iteratively until the model's
confidence for the next point drops below a threshold, or a maximum number
of iterations is reached.
Clears its specific output directory before running.
Prints the total number of GT points in the original image (for comparison).
Skips processing patches that have no ground truth annotations.
Applies Non-Maximum Suppression (NMS) to the final predicted points.
Calculates and reports Mean Absolute Error (MAE).
"""
import os
import cv2
import torch
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import glob
import shutil
import math # For NMS distance calculation

# --- Configuration ---
from config import (
    DEVICE, MODEL_INPUT_SIZE, IMAGE_DIR_TEST, GT_DIR_TEST,
    BEST_MODEL_PATH, OUTPUT_DIR, GT_PSF_SIGMA
)
from model import VGG19FPNASPP

# ImageNet Mean/Std for Normalization/Unnormalization
IMG_MEAN_CPU = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMG_STD_CPU = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# --- Inference Control Parameters ---
CONFIDENCE_THRESHOLD = 0.2  # Stop if confidence for next point is below this
MAX_ITERATIONS_PER_PATCH = 100 # Max predictions per patch, even if confidence remains high
NMS_RADIUS = 7 # Radius for Non-Maximum Suppression (in pixels)

# --- Helper Functions ---

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

def create_input_psf_from_points(points_list, shape, sigma):
    h, w = shape
    delta_map = np.zeros((h, w), dtype=np.float32)
    if not points_list: # If points_list is empty
        return delta_map # Return an all-zero map
    for x, y in points_list:
        x_coord, y_coord = np.clip(int(round(x)), 0, w - 1), np.clip(int(round(y)), 0, h - 1)
        delta_map[y_coord, x_coord] += 1.0
    input_psf = gaussian_filter(delta_map, sigma=sigma, order=0, mode='constant', cval=0.0)
    max_val = np.max(input_psf)
    if max_val > 1e-7: input_psf /= max_val
    return input_psf

def get_patch_coordinates(img_h, img_w, patch_size):
    """
    Generates coordinates (x_start, y_start, x_end, y_end) for patches.
    Assumes img_h, img_w are dimensions of the (potentially padded) image
    and are >= patch_size.
    """
    coords = []
    y_range = list(range(0, img_h - patch_size, patch_size))
    if img_h >= patch_size : 
        y_range.append(img_h - patch_size)
    y_starts = sorted(list(set(y_range)))
    if not y_starts and img_h >= patch_size : y_starts = [0]

    x_range = list(range(0, img_w - patch_size, patch_size))
    if img_w >= patch_size:
        x_range.append(img_w - patch_size)
    x_starts = sorted(list(set(x_range)))
    if not x_starts and img_w >= patch_size : x_starts = [0]

    for sy in y_starts:
        for sx in x_starts:
            coords.append((sx, sy, sx + patch_size, sy + patch_size))
    return coords

def perform_iterative_inference_on_patch(
    model,
    image_patch_tensor_batch, 
    psf_sigma=GT_PSF_SIGMA,
    model_input_size=MODEL_INPUT_SIZE,
    device=DEVICE,
    confidence_threshold=CONFIDENCE_THRESHOLD, 
    max_iterations=MAX_ITERATIONS_PER_PATCH   
):
    predicted_points_local = []
    for iter_count in range(max_iterations):
        current_input_psf_np = create_input_psf_from_points(
            predicted_points_local, 
            shape=(model_input_size, model_input_size),
            sigma=psf_sigma
        )
        current_input_psf_tensor = torch.from_numpy(current_input_psf_np).float().unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            predicted_output_psf_tensor, predicted_confidence_logits = model(image_patch_tensor_batch, current_input_psf_tensor)
        confidence_score = torch.sigmoid(predicted_confidence_logits).item() 
        if confidence_score < confidence_threshold:
            break 
        output_psf_np = predicted_output_psf_tensor.squeeze().cpu().numpy() 
        max_yx = np.unravel_index(np.argmax(output_psf_np), output_psf_np.shape)
        pred_y, pred_x = max_yx[0], max_yx[1]
        predicted_points_local.append((pred_x, pred_y))
    return predicted_points_local

def apply_nms(points, radius_threshold):
    """
    Applies Non-Maximum Suppression to a list of (x, y) points.
    Points are processed in the order they are given.
    """
    if not points:
        return []
    
    points_to_process = list(points) 
    kept_points = []
    
    while points_to_process:
        current_point = points_to_process.pop(0) 
        kept_points.append(current_point)
        
        remaining_points_after_nms = []
        for point_to_check in points_to_process:
            dist = math.sqrt((current_point[0] - point_to_check[0])**2 + (current_point[1] - point_to_check[1])**2)
            if dist > radius_threshold:
                remaining_points_after_nms.append(point_to_check)
        points_to_process = remaining_points_after_nms
        
    return kept_points

# --- Main Function ---
def main():
    print("--- Full Image Patched Iterative Inference Script (Confidence-Driven, NMS, MAE) ---")
    patch_level_visuals_enabled = False 

    script_output_dir = os.path.join(OUTPUT_DIR, "full_image_patched_outputs_confidence_driven_nms_mae") # New folder
    if os.path.exists(script_output_dir):
        print(f"Clearing existing output directory: {script_output_dir}")
        try: shutil.rmtree(script_output_dir)
        except OSError as e:
            print(f"Error clearing directory: {e}. Please close files and retry.")
            return
    os.makedirs(script_output_dir, exist_ok=True)
    print(f"Outputs will be saved in: {script_output_dir}")
    print(f"Confidence Threshold: {CONFIDENCE_THRESHOLD}, Max Iterations per Patch: {MAX_ITERATIONS_PER_PATCH}, NMS Radius: {NMS_RADIUS}")

    if not os.path.exists(BEST_MODEL_PATH):
        print(f"Error: Model not found at {BEST_MODEL_PATH}"); return
    model = VGG19FPNASPP().to(DEVICE)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("Model loaded.")

    test_image_paths = sorted(glob.glob(os.path.join(IMAGE_DIR_TEST, '*.jpg')))
    if not test_image_paths: print(f"No test images in {IMAGE_DIR_TEST}"); return
    
    num_images_to_test = min(50, len(test_image_paths)) # Process a subset for quick testing
    # num_images_to_test = len(test_image_paths) # Uncomment to process all test images
    selected_image_paths = test_image_paths[:num_images_to_test]
    print(f"Will process {len(selected_image_paths)} images for testing.")

    total_absolute_error = 0.0
    image_count_for_mae = 0

    for image_path in selected_image_paths:
        img_filename = os.path.basename(image_path)
        img_filestem = os.path.splitext(img_filename)[0]
        gt_path = os.path.join(GT_DIR_TEST, "GT_" + img_filestem + ".mat")
        print(f"\nProcessing image: {img_filename}")

        img_orig_bgr = cv2.imread(image_path)
        if img_orig_bgr is None:
            print(f"Failed to load image {image_path}. Skipping.")
            continue
        img_orig_rgb = cv2.cvtColor(img_orig_bgr, cv2.COLOR_BGR2RGB)
        gt_orig_global = load_gt_points(gt_path) 
        
        num_gt_points_original_image = len(gt_orig_global)
        print(f"Total number of GT points in the original image: {num_gt_points_original_image}")

        h_orig, w_orig = img_orig_rgb.shape[:2]
        pad_h_bottom = max(0, MODEL_INPUT_SIZE - h_orig)
        pad_w_right = max(0, MODEL_INPUT_SIZE - w_orig)
        
        img_to_process = cv2.copyMakeBorder(img_orig_rgb, 0, pad_h_bottom, 0, pad_w_right,
                                            cv2.BORDER_CONSTANT, value=[0,0,0]) 
        h_proc, w_proc = img_to_process.shape[:2]

        patch_coords_list = get_patch_coordinates(h_proc, w_proc, MODEL_INPUT_SIZE)
        if not patch_coords_list:
            print(f"  No patches generated for image {img_filename} (h_proc: {h_proc}, w_proc: {w_proc}, patch_size: {MODEL_INPUT_SIZE}). Skipping.")
            continue
        # print(f"  Generated {len(patch_coords_list)} patches for the image.")

        all_predicted_points_global_raw = [] 
        processed_patch_count = 0

        for i, (x_start, y_start, x_end, y_end) in enumerate(patch_coords_list):
            num_gt_in_patch = 0
            if gt_orig_global.size > 0:
                for gx_orig, gy_orig in gt_orig_global:
                    if x_start <= gx_orig < x_end and y_start <= gy_orig < y_end:
                        num_gt_in_patch += 1
            
            if num_gt_in_patch == 0:
                continue 
            
            processed_patch_count += 1
            img_patch_np = img_to_process[y_start:y_end, x_start:x_end]
            patch_tensor_np = img_patch_np.astype(np.float32) / 255.0
            patch_tensor_chw = torch.from_numpy(patch_tensor_np).permute(2, 0, 1)
            patch_tensor_norm = (patch_tensor_chw - IMG_MEAN_CPU) / IMG_STD_CPU
            patch_tensor_batch = patch_tensor_norm.unsqueeze(0).to(DEVICE) 

            predicted_points_local = perform_iterative_inference_on_patch(
                model, patch_tensor_batch, psf_sigma=GT_PSF_SIGMA,
                model_input_size=MODEL_INPUT_SIZE, device=DEVICE,
                confidence_threshold=CONFIDENCE_THRESHOLD, 
                max_iterations=MAX_ITERATIONS_PER_PATCH    
            )

            for plx, ply in predicted_points_local:
                pgx, pgy = plx + x_start, ply + y_start
                if pgx < w_orig and pgy < h_orig:
                     all_predicted_points_global_raw.append((pgx, pgy))

            if patch_level_visuals_enabled and predicted_points_local:
                gt_local_to_patch_coords = [] 
                if gt_orig_global.size > 0: 
                    for gx_orig, gy_orig in gt_orig_global:
                        if x_start <= gx_orig < x_end and y_start <= gy_orig < y_end:
                            lx, ly = gx_orig - x_start, gy_orig - y_start
                            gt_local_to_patch_coords.append((lx, ly))
                gt_local_to_patch_np = np.array(gt_local_to_patch_coords)
                plt.figure(figsize=(6,6))
                display_patch_tensor = patch_tensor_batch.squeeze(0).cpu() * IMG_STD_CPU + IMG_MEAN_CPU
                display_patch_np = np.clip(display_patch_tensor.permute(1, 2, 0).numpy(), 0, 1)
                plt.imshow(display_patch_np)
                if gt_local_to_patch_np.size > 0: 
                    plt.scatter(gt_local_to_patch_np[:,0], gt_local_to_patch_np[:,1], s=30, facecolors='none', edgecolors='lime', lw=1, label=f'Local GT ({len(gt_local_to_patch_np)})')
                preds_local_np = np.array(predicted_points_local)
                plt.scatter(preds_local_np[:,0], preds_local_np[:,1], s=20, c='red', marker='x', label=f'Local Preds ({len(preds_local_np)})')
                plt.title(f"Patch {i+1} (GT: {num_gt_in_patch}) Results") 
                plt.axis('off'); plt.legend()
                patch_plot_path = os.path.join(script_output_dir, f"{img_filestem}_patch_{i+1:03d}.png")
                plt.savefig(patch_plot_path); plt.close()
        
        print(f"  Processed {processed_patch_count} (non-empty GT) out of {len(patch_coords_list)} total patches for image {img_filename}.")
        
        num_predicted_before_nms = len(all_predicted_points_global_raw)
        all_predicted_points_global_nms = apply_nms(all_predicted_points_global_raw, NMS_RADIUS)
        num_predicted_after_nms = len(all_predicted_points_global_nms)
        
        print(f"  Predicted points for {img_filename}: {num_predicted_before_nms} (before NMS), {num_predicted_after_nms} (after NMS with radius {NMS_RADIUS})")

        # Calculate and accumulate absolute error for MAE
        current_abs_error = abs(num_predicted_after_nms - num_gt_points_original_image)
        total_absolute_error += current_abs_error
        image_count_for_mae += 1
        print(f"  Absolute Error for {img_filename}: {current_abs_error}")

        plt.figure(figsize=(12, 12 * h_orig / w_orig if w_orig > 0 else 12))
        plt.imshow(img_orig_rgb)
        if gt_orig_global.size > 0:
            plt.scatter(gt_orig_global[:, 0], gt_orig_global[:, 1], s=30, facecolors='none', edgecolors='lime', linewidths=1.5, label=f'Ground Truth ({num_gt_points_original_image})')
        
        if all_predicted_points_global_nms: 
            preds_global_nms_np = np.array(all_predicted_points_global_nms)
            plt.scatter(preds_global_nms_np[:, 0], preds_global_nms_np[:, 1], s=20, c='red', marker='x', label=f'Predicted (NMS, {num_predicted_after_nms})')
        
        plt.title(f"Full Image Predictions for {img_filename}\nGT: {num_gt_points_original_image}, Pred (NMS): {num_predicted_after_nms}, AE: {current_abs_error}")
        plt.axis('off'); plt.legend()
        final_summary_path = os.path.join(script_output_dir, f"{img_filestem}_full_summary_conf_nms.png")
        plt.savefig(final_summary_path); plt.close()
        # print(f"  Full image processing complete. Total points predicted (after NMS): {num_predicted_after_nms}") # Redundant with above
        print(f"  Final summary plot saved to: {final_summary_path}")

    print("\n--- All selected images processed. ---")

    if image_count_for_mae > 0:
        mae = total_absolute_error / image_count_for_mae
        print(f"\nOverall Mean Absolute Error (MAE) for {image_count_for_mae} images: {mae:.2f}")
    else:
        print("\nNo images were processed for MAE calculation.")

if __name__ == "__main__":
    main()