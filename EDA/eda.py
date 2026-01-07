import os
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "processed_data/combined_features.csv"
OUT_DIR = "EDA"

os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------
# LOAD CSV
# -------------------------
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV not found at: {CSV_PATH}")

print("Loading CSV for EDA...")
df = pd.read_csv(CSV_PATH)

print("Total samples:", len(df))
print("Columns:", df.columns.tolist())

# -------------------------
# CLASS DISTRIBUTION
# -------------------------
plt.figure(figsize=(6,4))
df["main_class"].value_counts().plot(kind="bar")
plt.title("Class Distribution")
plt.xlabel("Printer Class")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "class_distribution.png"))
plt.close()

# -------------------------
# MEAN INTENSITY
# -------------------------
plt.figure(figsize=(6,4))
df["mean_intensity"].hist(bins=30)
plt.title("Mean Intensity Distribution")
plt.xlabel("Mean Intensity")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "intensity.png"))
plt.close()

# -------------------------
# RESOLUTION
# -------------------------
plt.figure(figsize=(6,4))
df["resolution"].hist(bins=20)
plt.title("Resolution Distribution")
plt.xlabel("Resolution")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "resolution_distribution.png"))
plt.close()

print("EDA completed successfully")
print("EDA outputs saved in EDA/ folder")