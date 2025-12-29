import streamlit as st
from PIL import Image
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Live Prediction | TraceFinder",
    layout="wide"
)

# ---------------- TITLE ----------------
st.title("🔮 Live Prediction")
st.subheader("Upload scanned document and identify source scanner")

st.markdown("---")

# ---------------- MODEL SELECTION ----------------
st.header("🧠 Select Prediction Model")

model_choice = st.selectbox(
    "Choose a model for prediction",
    ["CNN", "Random Forest", "SVM"]
)

st.markdown("---")

# ---------------- FILE UPLOADER ----------------
st.header("📤 Upload Scanned Image")

uploaded_file = st.file_uploader(
    "Browse and upload a scanned image (JPG / PNG / JPEG / TIFF)",
    type=["jpg", "jpeg", "png", "tif", "tiff"]
)

# ---------------- PREDICTION LOGIC ----------------
if uploaded_file is not None:
    try:
        # Open image (TIFF supported)
        image = Image.open(uploaded_file)

        # Convert TIFF / grayscale / CMYK to RGB
        image = image.convert("RGB")

        # Display image
        st.image(
            image,
            caption="Uploaded Scanned Image",
            width=400
        )

        st.markdown("### 🔍 Prediction Result")

        if st.button("🚀 Predict Source"):
            # ---------------- DUMMY LOGIC (replace later) ----------------
            # This is placeholder until real models are connected

            if model_choice == "CNN":
                predicted_source = "Flatbed Scanner"
                confidence = 0.91
                machine = "CNN Model"

            elif model_choice == "Random Forest":
                predicted_source = "Mobile Scanner App"
                confidence = 0.86
                machine = "Random Forest Model"

            else:
                predicted_source = "Office Scanner"
                confidence = 0.83
                machine = "SVM Model"

            # ---------------- SHOW RESULTS ----------------
            st.success(f"📄 **Predicted Source:** {predicted_source}")
            st.info(f"🧠 **Model Used:** {machine}")
            st.warning(f"📊 **Confidence:** {confidence * 100:.2f}%")

    except Exception as e:
        st.error("❌ Unable to read this image. Please upload a valid scanned image (TIFF/JPG/PNG).")

else:
    st.info("👆 Please upload a scanned image to start live prediction.")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("© 2025 TraceFinder | Live Prediction Module")