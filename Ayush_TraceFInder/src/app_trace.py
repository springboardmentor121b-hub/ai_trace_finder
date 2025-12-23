# app_tracefinder.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import cv2
import os
from io import BytesIO
from skimage import io as skio, img_as_float
from skimage.restoration import denoise_wavelet
from skimage.filters import sobel
from scipy.stats import skew, kurtosis, entropy
from PIL import Image

st.set_page_config(page_title="TraceFinder — Scanner/Camera ID", layout="wide")
st.title("TraceFinder — Scanner / Camera Source Identifier")
st.markdown("Upload one or more images (.tif, .jpg, .png). The app uses the baseline model (Random Forest) to predict the source device.")

# -------------------------
# Helpers (same feature functions used in preprocessing)
# -------------------------
def load_img_grayscale_bytes(file_bytes, size=(512,512)):
    # read bytes -> skimage float grayscale -> resize via cv2
    arr = skio.imread(BytesIO(file_bytes.read()), as_gray=True)
    arr = img_as_float(arr)
    h, w = arr.shape
    # cv2.resize expects float32
    resized = cv2.resize(arr.astype('float32'), size, interpolation=cv2.INTER_AREA)
    return resized

def compute_metadata_features(img_array, file_size_bytes=0, scanner_id="unknown", resolution="unknown"):
    # img_array: float image with range [0,1], shape (H,W)
    h, w = img_array.shape
    aspect_ratio = w / h if h != 0 else 0
    file_size_kb = file_size_bytes / 1024.0

    pixels = img_array.flatten()
    mean_intensity = float(np.mean(pixels))
    std_intensity = float(np.std(pixels))
    skewness = float(skew(pixels))
    kurt = float(kurtosis(pixels))
    ent = float(entropy(np.histogram(pixels, bins=256, range=(0,1))[0] + 1e-6))

    edges = sobel(img_array)
    edge_density = float(np.mean(edges > 0.1))

    return {
        "file_name": scanner_id,  # placeholder, will be overwritten by uploader filename
        "main_class": "Official",
        "resolution": resolution,
        "class_label": scanner_id,
        "width": int(w),
        "height": int(h),
        "aspect_ratio": aspect_ratio,
        "file_size_kb": file_size_kb,
        "mean_intensity": mean_intensity,
        "std_intensity": std_intensity,
        "skewness": skewness,
        "kurtosis": kurt,
        "entropy": ent,
        "edge_density": edge_density
    }

def compute_noise_residual(img_array):
    # optional: compute denoised residual (used only if you want to show residual)
    try:
        denoised = denoise_wavelet(img_array, channel_axis=None, rescale_sigma=True)
        residual = img_array - denoised
        return residual
    except Exception:
        return None

# -------------------------
# Load models (baseline)
# -------------------------
@st.cache_resource
def load_models():
    model_path = os.path.join("models", "baseline", "random_forest.joblib")
    scaler_path = os.path.join("models", "baseline", "scaler.joblib")
    encoder_path = os.path.join("models", "baseline", "label_encoder.joblib")

    if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(encoder_path)):
        st.error("Baseline model files not found. Train baseline models first.")
        return None, None, None

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    le = joblib.load(encoder_path)
    return model, scaler, le

model, scaler, label_encoder = load_models()

# -------------------------
# UI: uploader
# -------------------------
uploaded_files = st.file_uploader("Upload images (multiple allowed)", type=["tif","tiff","jpg","jpeg","png"], accept_multiple_files=True)

if uploaded_files and model is not None and scaler is not None:
    results = []
    cols = st.columns(2)
    left = cols[0]
    right = cols[1]

    # Process each file and show a compact card
    for idx, up in enumerate(uploaded_files):
        file_name = up.name
        # read file bytes twice: one for display, one for processing
        bytes_io = BytesIO(up.read())
        bytes_io_for_display = BytesIO(bytes_io.getvalue())
        bytes_io_for_proc = BytesIO(bytes_io.getvalue())

        # Display preview
        with left:
            st.markdown(f"**File:** {file_name}")
            try:
                pil_img = Image.open(bytes_io_for_display)
                st.image(pil_img, caption=file_name, use_column_width=True)
            except Exception as e:
                st.warning(f"Could not preview image: {e}")

        # Preprocess & features
        try:
            img_arr = load_img_grayscale_bytes(bytes_io_for_proc)
            features = compute_metadata_features(img_arr, file_size_bytes=len(bytes_io.getvalue()))
            features["file_name"] = file_name
            # Prepare for model (same columns order expected by scaler)
            # We'll trust scaler and training used the same columns (drop identifiers)
            feature_order = ["width","height","aspect_ratio","file_size_kb","mean_intensity","std_intensity","skewness","kurtosis","entropy","edge_density"]
            X = np.array([[features[k] for k in feature_order]])
            X_scaled = scaler.transform(X)
            pred_idx = model.predict(X_scaled)[0]
            pred = label_encoder.inverse_transform([pred_idx])[0]
            probs = model.predict_proba(X_scaled)[0]
            class_labels = label_encoder.classes_
            class_prob_pairs = sorted(zip(class_labels, probs), key=lambda x: x[1], reverse=True)
            top3 = class_prob_pairs[:3]

            # show results
            with right:
                st.markdown("**Prediction**")
                st.success(f"Predicted: **{pred}**")
                st.markdown("**Top-3 (class — probability)**")
                for cl, p in top3:
                    st.write(f"- {cl} — {p:.3f}")

                # metadata features show / hide
                if st.checkbox(f"Show metadata features for {file_name}", key=f"meta_{idx}"):
                    df_meta = pd.DataFrame([features])
                    st.dataframe(df_meta.T, width=450)

                # optional: show noise residual (small)
                if st.checkbox(f"Show noise residual (small) for {file_name}", key=f"res_{idx}"):
                    residual = compute_noise_residual(img_arr)
                    if residual is None:
                        st.info("Residual not available (PyWavelets missing or error).")
                    else:
                        # scale residual for display
                        rmin, rmax = residual.min(), residual.max()
                        viz = (residual - rmin) / (rmax - rmin + 1e-12)
                        st.image(viz, clamp=True, channels="GRAY", caption="Noise residual (visualized)")

            # collect row
            row = features.copy()
            row["predicted"] = pred
            # store full prob vector as columns
            for i,(cl,p) in enumerate(class_prob_pairs):
                row[f"prob__{cl}"] = float(p)
            results.append(row)

        except Exception as e:
            st.error(f"Failed to process {file_name}: {e}")

    # After processing all, show a summary DataFrame and download option
    if results:
        df_results = pd.DataFrame(results)
        st.markdown("---")
        st.header("Batch Results")
        st.dataframe(df_results)

        csv = df_results.to_csv(index=False).encode('utf-8')
        st.download_button("Download results CSV", data=csv, file_name="tracefinder_results.csv", mime="text/csv")

else:
    if not uploaded_files:
        st.info("Upload one or more images to get predictions.")
    elif model is None:
        st.error("Model not loaded. Please train models (train_baseline.py) first.")
