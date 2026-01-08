import os
import cv2
import csv
import numpy as np
from skimage import io, img_as_float
from skimage.restoration import denoise_wavelet
from skimage.filters import sobel
from scipy.stats import skew, kurtosis, entropy

# Define Paths
DATA_ROOT = "datasets"
OFFICIAL_DIR = os.path.join(DATA_ROOT, "official")
WIKIPEDIA_DIR = os.path.join(DATA_ROOT, "Wikipedia")

OUTPUT_DIR = "processed_data"
PATCHES_DIR = os.path.join(OUTPUT_DIR, "patches")
CSV_PATH = os.path.join(OUTPUT_DIR, "combined_features.csv")

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)


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
            patches.append(img[i:i + patch_size, j:j + patch_size])

    return patches


# Metadata computation
def compute_metadata_features(img, file_path, scanner_id, dataset_source, resolution="unknown"):
    h, w = img.shape
    aspect_ratio = w / h
    file_size_kb = os.path.getsize(file_path) / 1024

    pixels = img.flatten()

    mean_intensity = np.mean(pixels)
    std_intensity = np.std(pixels)
    skewness = skew(pixels)
    kurt_val = kurtosis(pixels)
    ent = entropy(np.histogram(pixels, bins=256, range=(0, 1))[0] + 1e-6)

    edges = sobel(img)
    edge_density = np.mean(edges > 0.1)

    return {
        "file_name": os.path.basename(file_path),
        "dataset_source": dataset_source,
        "main_class": dataset_source.capitalize(),  # 'Official' or 'Wikipedia'
        "resolution": resolution,
        "class_label": scanner_id,
        "width": w,
        "height": h,
        "aspect_ratio": aspect_ratio,
        "file_size_kb": file_size_kb,
        "mean_intensity": mean_intensity,
        "std_intensity": std_intensity,
        "skewness": skewness,
        "kurtosis": kurt_val,
        "entropy": ent,
        "edge_density": edge_density
    }


def process_directory(source_dir, dataset_source, writer):
    print(f"Starting processing for: {dataset_source} from {source_dir}")

    for root, dirs, files in os.walk(source_dir):

        # Skip root if it's the main dataset folder
        if root == source_dir:
            continue

        # Extract scanner_id and subfolder
        rel_path = os.path.relpath(root, source_dir)
        path_parts = rel_path.split(os.sep)

        if len(path_parts) < 2:
            continue

        scanner_id = path_parts[0]
        subfolder_name = path_parts[-1]  # resolution or category

        resolution = "unknown"
        if subfolder_name.isdigit():
            resolution = subfolder_name

        save_path_base = os.path.join(PATCHES_DIR, dataset_source, scanner_id, subfolder_name)
        os.makedirs(save_path_base, exist_ok=True)

        files_processed = 0

        for file in files:

            if not file.lower().endswith(('.png', '.tif', '.jpg', '.jpeg', '.bmp')):
                continue

            img_path = os.path.join(root, file)

            try:
                img = load_and_preprocess(img_path)
            except Exception as e:
                print(f" Failed to read {img_path}: {e}")
                continue

            # Extract patches
            residual = extract_noise_residual(img)
            patches = extract_patches(residual)

            for idx, patch in enumerate(patches):
                out_name = f"{os.path.splitext(file)[0]}_{idx}.npy"
                np.save(os.path.join(save_path_base, out_name), patch)

            # Compute metadata
            features = compute_metadata_features(img, img_path, scanner_id, dataset_source, resolution)
            writer.writerow(features)

            files_processed += 1

        if files_processed > 0:
            print(f" Processed {files_processed} files in: {scanner_id}/{subfolder_name}")


def main():
    fieldnames = [
        "file_name", "dataset_source", "main_class", "resolution", "class_label",
        "width", "height", "aspect_ratio", "file_size_kb",
        "mean_intensity", "std_intensity", "skewness", "kurtosis",
        "entropy", "edge_density"
    ]

    with open(CSV_PATH, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        if os.path.exists(OFFICIAL_DIR):
            process_directory(OFFICIAL_DIR, "official", writer)
        else:
            print(f"Warning: Official dataset directory not found at {OFFICIAL_DIR}")

        if os.path.exists(WIKIPEDIA_DIR):
            process_directory(WIKIPEDIA_DIR, "wikipedia", writer)
        else:
            print(f"Warning: Wikipedia dataset directory not found at {WIKIPEDIA_DIR}")

    print(f"\nCombined feature extraction complete. Metadata saved to: {CSV_PATH}")


if __name__ == "__main__":
    main()