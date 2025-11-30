import os
import csv
import numpy as np
from skimage.filters import sobel
from scipy.stats import skew, kurtosis, entropy

DATASET_OFFICIAL = r"D:\project\Tracefinder\data"
OUTPUT_DIR = r"D:\project\Tracefinder\processed_data"
CSV_PATH = r"D:\project\Tracefinder\processed_data\metadata_features.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_and_preprocess_npy(npy_path):
    img = np.load(npy_path)
    img = img.astype(np.float32)
    if img.max() > 1:
        img = img / 255.0
    return img

def compute_metadata_features(img, file_path, scanner_id):
    h, w = img.shape
    aspect_ratio = w / h
    file_size_kb = os.path.getsize(file_path) / 1024

    pixels = img.flatten()
    mean_intensity = np.mean(pixels)
    std_intensity = np.std(pixels)
    skewness = skew(pixels)
    kurt = kurtosis(pixels)

    ent = entropy(np.histogram(pixels, bins=256, range=(0, 1))[0] + 1e-6)

    edges = sobel(img)
    edge_density = np.mean(edges > 0.1)

    return {
        "file_name": os.path.basename(file_path),
        "main_class": "Official",
        "resolution": "npy",
        "class_label": scanner_id,
        "width": w,
        "height": h,
        "aspect_ratio": aspect_ratio,
        "file_size_kb": file_size_kb,
        "mean_intensity": mean_intensity,
        "std_intensity": std_intensity,
        "skewness": skewness,
        "kurtosis": kurt,
        "entropy": ent,
        "edge_density": edge_density
    }

def preprocess_official_dataset(official_dir, csv_path):
    fieldnames = [
        "file_name", "main_class", "resolution", "class_label",
        "width", "height", "aspect_ratio", "file_size_kb",
        "mean_intensity", "std_intensity", "skewness", "kurtosis",
        "entropy", "edge_density"
    ]

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for root, dirs, files in os.walk(official_dir):
            if root == official_dir:
                continue

            scanner_id = os.path.basename(root)

            for file in files:
                if not file.endswith(".npy"):
                    continue

                npy_path = os.path.join(root, file)

                try:
                    img = load_and_preprocess_npy(npy_path)
                except:
                    continue

                features = compute_metadata_features(img, npy_path, scanner_id)
                writer.writerow(features)

    print("Preprocessing completed")

if __name__ == "__main__":
    preprocess_official_dataset(DATASET_OFFICIAL, CSV_PATH)
