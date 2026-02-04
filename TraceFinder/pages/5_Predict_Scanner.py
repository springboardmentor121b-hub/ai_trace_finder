import streamlit as st
from prediction import predict_image
import numpy as np

st.set_page_config(layout="wide")

st.title("🔍 Predict Scanner from Image")

st.markdown("Select a model, upload a scanned image, and identify its source scanner.")

# ---------------- MODEL SELECTION ---------------- #
model_choice = st.selectbox(
    "🧠 Choose Model",
    ["Random Forest", "SVM", "CNN"]
)

# ---------------- IMAGE UPLOAD ---------------- #
uploaded_file = st.file_uploader(
    "📤 Upload scanned image",
    type=["jpg", "jpeg", "png", "tif", "tiff"]
)

if uploaded_file:
    file_bytes = uploaded_file.getvalue()

    with st.spinner("Analyzing image..."):
        pred, probs = predict_image(file_bytes, model_choice)

    st.success(f"✅ Predicted Scanner: **{pred}**")

    st.markdown("### 📊 Class Probabilities")
    st.bar_chart(probs)
