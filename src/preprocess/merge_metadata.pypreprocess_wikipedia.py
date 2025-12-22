import os
import cv2
import csv
import numpy as np
from skimage import io, img_as_float
from skimage.restoration import denoise_wavelet
from skimage.filters import sobel
from skimage.feature import local_binary_pattern
from scipy.stats import skew, kurtosis, entropy


DATASET_WIKIPEDIA = "data/WikiPedia"
OUTPUT_DIR = "processed_data/WikiPedia"
os.makedirs(OUTPUT_DIR, exist_ok=True)
CSV_PATH = os.path.join(OUTPUT_DIR, "metadata_features_wikipedia.csv")


def load_and_preprocess(img_path, size=(512, 512)):
    img = io.imread(img_path, as_gray=True)
    img = img_as_float(img)
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

# Noise residual
def extract_noise_residual(img):
    denoised = denoise_wavelet(img, channel_axis=None, rescale_sigma=True)
    return img - denoised

# Patch extraction
def extract_patches(img, patch_size=128, stride=128):
    patches = []
    h, w = img.shape
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            patches.append(img[i:i+patch_size, j:j+patch_size])
    return patches

# Metadata computation
def compute_metadata_features(img, residual, file_path, scanner_id, resolution="unknown"):
    h, w = img.shape
    aspect_ratio = w / h
    file_size_kb = os.path.getsize(file_path) / 1024

    # --- Features from RESIDUAL (NOISE) ---
    pixels = residual.flatten()
    mean_intensity = np.mean(pixels)
    std_intensity = np.std(pixels)
    skewness = skew(pixels)
    kurt = kurtosis(pixels)
    ent = entropy(np.histogram(pixels, bins=256)[0] + 1e-6)

    # --- LBP on RESIDUAL ---
    # Scale residual to 0-255 integer range to prevent UserWarning
    residual_int = np.uint8(255 * (residual - np.min(residual)) / (np.max(residual) - np.min(residual) + 1e-6))
    lbp = local_binary_pattern(residual_int, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp, bins=np.arange(0, 11), range=(0, 10), density=True)
    
    features = {
        "file_name": os.path.basename(file_path),
        "main_class": "Wikipedia",
        "resolution": resolution,
        "class_label": scanner_id,
        "width": w,
        "height": h,
        "aspect_ratio": aspect_ratio,
        "file_size_kb": file_size_kb,
        "mean_intensity": mean_intensity,
        "std_intensity": std_intensity,
        "skewness": skewness,
        "kurtosis": kurt,
        "entropy": ent
    }
    
    # Add LBP features to the dictionary
    for i, val in enumerate(lbp_hist):
        features[f"lbp_hist_{i}"] = val

    # --- Edge density from ORIGINAL image ---
    edges = sobel(img)
    features['edge_density'] = np.mean(edges > 0.1)

    return features

# Main preprocessing function
def preprocess_wikipedia_dataset(wikipedia_dir, out_dir, csv_path):
    # Add LBP features to the CSV header
    fieldnames = [
        "file_name", "main_class", "resolution", "class_label",
        "width", "height", "aspect_ratio", "file_size_kb",
        "mean_intensity", "std_intensity", "skewness", "kurtosis",
        "entropy", "edge_density"
    ] + [f"lbp_hist_{i}" for i in range(10)]

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Walk through all folders and subfolders
        for root, dirs, files in os.walk(wikipedia_dir):
            # Skip root if it's the main dataset folder (to infer scanner_id)
            if root == wikipedia_dir:
                continue

            # Infer scanner_id from first level folder
            scanner_id = os.path.basename(os.path.dirname(root))
            subfolder_name = os.path.basename(root)

            save_path = os.path.join(out_dir, scanner_id, subfolder_name)
            os.makedirs(save_path, exist_ok=True)

            files_processed = 0
            for file in files:
                if not file.lower().endswith(('.png', '.tif', '.jpg', '.jpeg')):
                    continue

                img_path = os.path.join(root, file)
                try:
                    img = load_and_preprocess(img_path)
                except Exception as e:
                    print(f" Failed to read {img_path}: {e}")
                    continue

                residual = extract_noise_residual(img)
                # The script saves patches, but we are focusing on the features for the CSV
                # patches = extract_patches(residual)
                # for idx, patch in enumerate(patches):
                #     out_name = f"{os.path.splitext(file)[0]}_{idx}.npy"
                #     np.save(os.path.join(save_path, out_name), patch)
                
                # Pass the residual to the feature computation
                features = compute_metadata_features(img, residual, img_path, scanner_id)
                writer.writerow(features)
                files_processed += 1

            if files_processed > 0:
                print(f" Processed {files_processed} files in folder: {root}")

if __name__ == "__main__":
    preprocess_wikipedia_dataset(DATASET_WIKIPEDIA, OUTPUT_DIR, CSV_PATH)
    print(" Wikipedia preprocessing + metadata feature extraction complete.")
