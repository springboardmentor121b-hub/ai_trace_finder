import os
import pickle
import numpy as np
from tqdm import tqdm

from utils import batch_corr_gpu, extract_enhanced_features, corr2d

# =====================================================
# PATHS
# =====================================================

BASE_DIR = r"C:\Infosys_Internship"
RESULTS_DIR = os.path.join(BASE_DIR, "results", "hybrid_cnn")

FLATFIELD_RESIDUALS_PATH = os.path.join(RESULTS_DIR, "flatfield_residuals.pkl")
RES_PATH = os.path.join(RESULTS_DIR, "official_wiki_residuals.pkl")
FEATURES_OUT = os.path.join(RESULTS_DIR, "features.pkl")
ENHANCED_OUT = os.path.join(RESULTS_DIR, "enhanced_features.pkl")

# =====================================================
# LOAD FINGERPRINTS
# =====================================================

with open(FLATFIELD_RESIDUALS_PATH, "rb") as f:
    flatfield_residuals = pickle.load(f)

scanner_fingerprints = {}
for scanner, residuals in flatfield_residuals.items():
    if residuals:
        scanner_fingerprints[scanner] = np.mean(
            np.stack(residuals, axis=0), axis=0
        )

fp_keys = sorted(scanner_fingerprints.keys())
print(f"Loaded {len(fp_keys)} scanner fingerprints")

# =====================================================
# LOAD RESIDUALS
# =====================================================

with open(RES_PATH, "rb") as f:
    residuals_dict = pickle.load(f)

print("Datasets:", residuals_dict.keys())

# =====================================================
# PRNU FEATURE EXTRACTION (ROBUST)
# =====================================================

features, labels = [], []

for dataset_name in ["official", "Wikipedia"]:
    if dataset_name not in residuals_dict:
        continue

    print(f"Extracting PRNU features for {dataset_name} ...")

    for scanner, data in tqdm(residuals_dict[dataset_name].items()):

        # 🔥 CASE 1: data is LIST of residuals
        if isinstance(data, list):
            res_lists = [data]

        # 🔥 CASE 2: data is DICT (dpi → residuals)
        elif isinstance(data, dict):
            res_lists = data.values()

        else:
            continue

        for res_list in res_lists:
            if not res_list:
                continue

            # Try GPU correlation
            try:
                res_array = np.stack(res_list, axis=0).astype(np.float32)
                corrs = batch_corr_gpu(res_array, scanner_fingerprints, fp_keys)
            except Exception:
                corrs = None

            # 🔁 CPU FALLBACK
            if corrs is None or len(corrs) == 0:
                for res in res_list:
                    row = [
                        corr2d(res, scanner_fingerprints[fp])
                        for fp in fp_keys
                    ]
                    features.append(row)
                    labels.append(scanner)
            else:
                corrs = np.asarray(corrs)
                for i in range(corrs.shape[0]):
                    features.append(corrs[i].tolist())
                    labels.append(scanner)

# =====================================================
# SAVE PRNU FEATURES
# =====================================================

if len(features) == 0:
    raise RuntimeError("PRNU feature extraction failed — features empty")

with open(FEATURES_OUT, "wb") as f:
    pickle.dump({"features": features, "labels": labels}, f)

print(f"✅ PRNU features extracted: {len(features)} samples")

# =====================================================
# ENHANCED FEATURES
# =====================================================

enhanced_features, enhanced_labels = [], []

for dataset_name in ["official", "Wikipedia"]:
    if dataset_name not in residuals_dict:
        continue

    for scanner, data in tqdm(residuals_dict[dataset_name].items()):
        if isinstance(data, list):
            res_lists = [data]
        elif isinstance(data, dict):
            res_lists = data.values()
        else:
            continue

        for res_list in res_lists:
            for res in res_list:
                enhanced_features.append(extract_enhanced_features(res))
                enhanced_labels.append(scanner)

with open(ENHANCED_OUT, "wb") as f:
    pickle.dump(
        {"features": enhanced_features, "labels": enhanced_labels}, f
    )

print(f"✅ Enhanced features extracted: {len(enhanced_features)} samples")
print("\n🎉 FEATURE EXTRACTION COMPLETED SUCCESSFULLY")
