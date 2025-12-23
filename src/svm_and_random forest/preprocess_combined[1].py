import os
import csv
import numpy as np
from skimage import io, img_as_float
from skimage.restoration import denoise_wavelet
from skimage.filters import sobel
from scipy.stats import skew, kurtosis, entropy

# PATHS

DATA_ROOT = r"D:\Project\TraceFinder\data"
OFFICIAL_DIR = os.path.join(DATA_ROOT, "Official")
WIKIPEDIA_DIR = os.path.join(DATA_ROOT, "Wikipedia")

OUTPUT_DIR = r"D:\Project\TraceFinder\processed_data"
CSV_PATH = os.path.join(OUTPUT_DIR, "combined_features.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# FUNCTIONS

def load_image(path):
    img = io.imread(path, as_gray=True)
    return img_as_float(img)

def noise_residual(img):
    denoised = denoise_wavelet(img, channel_axis=None, rescale_sigma=True)
    return img - denoised

def fft_features(img):
    f = np.fft.fft2(img)
    fshift = np.abs(np.fft.fftshift(f))
    return np.mean(fshift), np.std(fshift)

def compute_features(img, file_path, scanner_id, dataset_source, resolution):
    h, w = img.shape
    pixels = img.flatten()

    hist = np.histogram(pixels, bins=256, range=(0, 1))[0] + 1e-6
    residual = noise_residual(img)
    r_pixels = residual.flatten()

    fft_mean, fft_std = fft_features(residual)

    return {
        "file_name": os.path.basename(file_path),
        "dataset_source": dataset_source,
        "main_class": dataset_source.capitalize(),
        "resolution": resolution,
        "class_label": scanner_id,

        "width": w,
        "height": h,
        "aspect_ratio": w / h,
        "file_size_kb": os.path.getsize(file_path) / 1024,

        "mean_intensity": np.mean(pixels),
        "std_intensity": np.std(pixels),
        "skewness": skew(pixels),
        "kurtosis": kurtosis(pixels),
        "entropy": entropy(hist),

        "residual_mean": np.mean(r_pixels),
        "residual_std": np.std(r_pixels),

        "fft_mean": fft_mean,
        "fft_std": fft_std,

        "edge_density": np.mean(sobel(img) > 0.1)
    }

def process_directory(source_dir, dataset_source, writer):
    for root, _, files in os.walk(source_dir):
        if root == source_dir:
            continue

        rel = os.path.relpath(root, source_dir).split(os.sep)
        if len(rel) < 2:
            continue

        scanner_id = rel[0]
        resolution = rel[-1] if rel[-1].isdigit() else "unknown"

        for file in files:
            if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
                continue

            path = os.path.join(root, file)
            try:
                img = load_image(path)
            except:
                continue

            features = compute_features(
                img, path, scanner_id, dataset_source, resolution
            )
            writer.writerow(features)

def main():
    fields = [
        "file_name", "dataset_source", "main_class", "resolution", "class_label",
        "width", "height", "aspect_ratio", "file_size_kb",
        "mean_intensity", "std_intensity", "skewness", "kurtosis", "entropy",
        "residual_mean", "residual_std",
        "fft_mean", "fft_std",
        "edge_density"
    ]

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        if os.path.exists(OFFICIAL_DIR):
            process_directory(OFFICIAL_DIR, "official", writer)

        if os.path.exists(WIKIPEDIA_DIR):
            process_directory(WIKIPEDIA_DIR, "wikipedia", writer)

    print("Feature extraction complete")

if __name__ == "__main__":
    main()
