# src/hybrid_cnn/feature_extraction.py
import os, pickle, numpy as np
from tqdm import tqdm
from utils import corr2d, extract_enhanced_features

RES_PATH = "results/hybrid_cnn/official_wiki_residuals.pkl"
OUT_DIR  = "results/hybrid_cnn"

os.makedirs(OUT_DIR, exist_ok=True)

print("Loading residuals...")
with open(RES_PATH, "rb") as f:
    residuals = pickle.load(f)

features = []
labels = []

print("Extracting PRNU + enhanced features...")
for dataset in ["official", "Wikipedia"]:
    print(f"Processing {dataset} dataset...")
    for scanner in tqdm(residuals[dataset].keys(), desc=f"Scanners in {dataset}"):
        res_list = residuals[dataset][scanner]
        for res in res_list:
            feat = extract_enhanced_features(res)
            features.append(feat)
            labels.append(scanner)

features = np.array(features, dtype=np.float32)
labels   = np.array(labels)

print("Feature shape:", features.shape)

with open(f"{OUT_DIR}/features.pkl", "wb") as f:
    pickle.dump({"features": features, "labels": labels}, f)

print("✅ Feature extraction completed")
