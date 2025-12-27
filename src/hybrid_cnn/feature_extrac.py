import os
import pickle
import numpy as np
from tqdm import tqdm

from utils import batch_corr_gpu, extract_enhanced_features



#  PATH CONFIGURATION



BASE_MODEL_DIR = r"D:\Project\TraceFinder\models\hybrid_cnn"   


RES_PATH = os.path.join(BASE_MODEL_DIR, "official_wiki_residuals.pkl")      ### ← CHECK
FLATFIELD_RESIDUALS_PATH = os.path.join(BASE_MODEL_DIR, "flatfield_residuals.pkl")  ### ← CHECK




# Scanner fingerprints (PRNU)
FP_OUT_PATH = os.path.join(BASE_MODEL_DIR, "scanner_fingerprints.pkl")      ### ← CHECK
ORDER_NPY = os.path.join(BASE_MODEL_DIR, "fp_keys.npy")                     ### ← CHECK

#  Feature files
FEATURES_OUT = os.path.join(BASE_MODEL_DIR, "features.pkl")                 ### ← CHECK
ENHANCED_OUT = os.path.join(BASE_MODEL_DIR, "enhanced_features.pkl")        ### ← CHECK



# STEP 1: Compute Scanner Fingerprints


if not os.path.exists(FLATFIELD_RESIDUALS_PATH):
    raise FileNotFoundError(
        f"Flatfield residuals not found at: {FLATFIELD_RESIDUALS_PATH}\n"
        "Run processing.py first."
    )

print("Loading flatfield residuals...")
with open(FLATFIELD_RESIDUALS_PATH, "rb") as f:
    flatfield_residuals = pickle.load(f)

scanner_fingerprints = {}

print("Computing scanner fingerprints from Flatfield images...")
for scanner, residuals in flatfield_residuals.items():
    if not residuals:
        continue

    # Average residuals to form fingerprint
    stack = np.stack(residuals, axis=0)
    fingerprint = np.mean(stack, axis=0)
    scanner_fingerprints[scanner] = fingerprint

# Save fingerprints
os.makedirs(os.path.dirname(FP_OUT_PATH), exist_ok=True)

with open(FP_OUT_PATH, "wb") as f:
    pickle.dump(scanner_fingerprints, f)

# Save deterministic scanner order
fp_keys = sorted(scanner_fingerprints.keys())
np.save(ORDER_NPY, np.array(fp_keys))

print(f"Saved {len(scanner_fingerprints)} scanner fingerprints")
print(f"Fingerprint keys saved to {ORDER_NPY}")



# STEP 2: Load Residuals (Official + Wikipedia)


if not os.path.exists(RES_PATH):
    raise FileNotFoundError(
        f"Residual file not found at: {RES_PATH}\n"
        "Run processing.py first."
    )

print("Loading official + Wikipedia residuals...")
with open(RES_PATH, "rb") as f:
    residuals_dict = pickle.load(f)



# STEP 3: PRNU Correlation Features


features = []
labels = []

for dataset_name in ["Official", "Wikipedia"]:   # 🔴 Must match preprocessing keys
    if dataset_name not in residuals_dict:
        print(f"Dataset {dataset_name} not found in residuals.")
        continue

    print(f"\nComputing PRNU features for {dataset_name} dataset...")

    for scanner, dpi_dict in tqdm(residuals_dict[dataset_name].items()):
        if not isinstance(dpi_dict, dict):
            continue

        for dpi, res_list in dpi_dict.items():
            if not res_list:
                continue

            # Batch correlation (GPU if available, CPU fallback inside utils)
            corrs = batch_corr_gpu(res_list, scanner_fingerprints, fp_keys)

            features.extend(corrs.tolist())
            labels.extend([scanner] * len(res_list))

# Save PRNU features
with open(FEATURES_OUT, "wb") as f:
    pickle.dump({"features": features, "labels": labels}, f)

print(f"\nSaved PRNU features to {FEATURES_OUT}")
print(f"Feature shape: {len(features)} x {len(features[0])}")



# STEP 4: Enhanced Features (FFT + LBP + Texture)


enhanced_features = []
enhanced_labels = []

for dataset_name in ["Official", "Wikipedia"]:   
    if dataset_name not in residuals_dict:
        continue

    print(f"\nExtracting enhanced features for {dataset_name} dataset...")

    for scanner, dpi_dict in tqdm(residuals_dict[dataset_name].items()):
        if not isinstance(dpi_dict, dict):
            continue

        for dpi, res_list in dpi_dict.items():
            for res in res_list:
                feat = extract_enhanced_features(res)
                enhanced_features.append(feat)
                enhanced_labels.append(scanner)

# Save enhanced features
with open(ENHANCED_OUT, "wb") as f:
    pickle.dump(
        {"features": enhanced_features, "labels": enhanced_labels},
        f
    )

print(f"\nSaved enhanced features to {ENHANCED_OUT}")
print(f"Enhanced feature shape: {len(enhanced_features)} x {len(enhanced_features[0])}")


print("\n FEATURE EXTRACTION COMPLETED SUCCESSFULLY")
