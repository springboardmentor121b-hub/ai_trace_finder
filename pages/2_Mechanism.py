import streamlit as st
from pathlib import Path
import sys

# Ensure configs module can be imported
sys.path.append(str(Path(__file__).resolve().parents[1]))
try:
    from configs.config import MECHANISM_PATH
except ImportError:
    MECHANISM_PATH = Path(__file__).resolve().parents[1] / "doc" / "assets" / "mechanism.png"

st.title("⚙️ Internal Mechanism – TraceFinder")

if MECHANISM_PATH.exists():
    st.image(str(MECHANISM_PATH), caption="TraceFinder Internal Mechanism", use_column_width=True)
else:
    st.error("mechanism.png not found. Please place it in doc/assets/.")

st.markdown("""
### 🔬 What This Mechanism Shows

This diagram shows the **technical workflow** inside TraceFinder:

- **Pre-processing**  
  Convert to grayscale, resize, normalize.

- **Wavelet Denoising**  
  Removes high-frequency content to reveal scanner noise.

- **Noise Residual Extraction**  
  Original – denoised = Scanner noise pattern.

- **Patch Extraction (128×128)**  
  Helps capture consistent noise fingerprints.

- **Feature Vector Construction**  
  Metadata + statistical noise features.

- **ML Model (Random Forest / SVM)**  
  Trained to classify which scanner produced the image.

- **Prediction Engine**  
  Outputs the scanner identity.
""")
