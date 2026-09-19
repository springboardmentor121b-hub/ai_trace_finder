import streamlit as st
import pandas as pd
from pathlib import Path
import sys

# Ensure configs module can be imported
sys.path.append(str(Path(__file__).resolve().parents[1]))
try:
    from configs.config import METADATA_CSV
except ImportError:
    METADATA_CSV = Path(__file__).resolve().parents[1] / "data" / "metadata_features.csv"

st.title("🧪 Metadata Feature Extraction – TraceFinder")

st.markdown("""
Metadata features play a **critical role** in scanner-source identification.  
These statistical features help the model learn the subtle differences between various scanners.
""")

st.markdown("---")

# Load metadata CSV dynamically
csv_path = METADATA_CSV

if not csv_path.exists():
    st.error(f"metadata_features.csv not found at {csv_path}. Please generate or add your dataset file.")
    st.stop()

df = pd.read_csv(csv_path)

# -----------------------------
# FEATURE DESCRIPTION SECTION
# -----------------------------

st.subheader("📘 What Features Were Extracted?")

st.markdown("""
TraceFinder extracts **10 core metadata features** from each image:

| Feature | Meaning |
|--------|---------|
| **Width & Height** | Physical pixel dimensions of the image |
| **Aspect Ratio** | Width / Height |
| **File Size (KB)** | Memory footprint of the file |
| **Mean Intensity** | Average pixel brightness |
| **Standard Deviation** | Variation in pixel intensity |
| **Skewness** | Measures asymmetry of pixel distribution |
| **Kurtosis** | Measures concentration & sharpness |
| **Entropy** | Randomness/texture complexity |
| **Edge Density** | % of pixels marked as edges |

These features help the ML model identify **scanner-specific fingerprints** that exist even in metadata.
""")

st.markdown("---")

# -----------------------------
# PREVIEW DATA
# -----------------------------

st.subheader("🔍 Sample Metadata Table")
st.dataframe(df.head(15))

st.markdown("---")

# -----------------------------
# FEATURE DISTRIBUTION VISUALIZATION
# -----------------------------

st.subheader("📊 Explore a Feature Distribution")

feature_cols = [
    c for c in df.columns 
    if c not in ["file_name", "main_class", "resolution", "class_label"]
]

selected = st.selectbox("Select a feature to visualize:", feature_cols)

st.line_chart(df[selected], height=250)

st.markdown("""
### 📎 Why Visualize Features?

Visualizing features helps you understand:

- How metadata varies across images  
- Whether the feature can help differentiate scanners  
- Possible patterns or outliers  
- Which features contribute most to classification  
""")

st.markdown("---")

# -----------------------------
# CLASS DISTRIBUTION
# -----------------------------

st.subheader("🏷️ Scanner Class Distribution")
st.bar_chart(df["class_label"].value_counts())

st.markdown("""
This chart shows how many images belong to each scanner class.
Balanced class distribution usually gives better model performance.
""")
