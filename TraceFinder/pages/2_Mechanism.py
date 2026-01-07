import streamlit as st
from pathlib import Path

st.title("⚙️ Internal Mechanism – TraceFinder")

mech_path = Path("assets/mechanism.png")

if mech_path.exists():
    st.image(str(mech_path), caption="TraceFinder Internal Mechanism", use_column_width=True)
else:
    st.error("mechanism.png not found. Please place it in the main folder.")

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
