import os
import cv2
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

DATASET_DIR = "Data_Sets"
OUTPUT_CSV = "metadata_features.csv"

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")

rows = []

for main_class in os.listdir(DATASET_DIR):
    main_class_path = os.path.join(DATASET_DIR, main_class)
    if not os.path.isdir(main_class_path):
        continue

    # walk through ALL subfolders recursively
    for root, dirs, files in os.walk(main_class_path):
        for file in files:
            if not file.lower().endswith(IMAGE_EXTS):
                continue

            img_path = os.path.join(root, file)

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            h, w = img.shape

            rows.append({
                "file_name": file,
                "main_class": main_class,
                "width": w,
                "height": h,
                "mean_intensity": float(np.mean(img)),
                "std_intensity": float(np.std(img)),
                "skewness": float(skew(img.flatten())),
                "kurtosis": float(kurtosis(img.flatten()))
            })

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False)

print("CSV CREATED:", OUTPUT_CSV)
print("Total samples:", len(df))
print("Class distribution:")
print(df["main_class"].value_counts())