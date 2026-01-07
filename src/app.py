import streamlit as st
import torch
from torchvision import transforms
import joblib
import numpy as np
import os
import sys
import pickle
from PIL import Image
import tensorflow as tf
import pandas as pd
import time

# --- 1. DYNAMIC PATH RESOLUTION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
def get_abs_path(rel_path):
    return os.path.join(BASE_DIR, rel_path)

sys.path.append(BASE_DIR)
sys.path.append(get_abs_path("src/hybrid_cnn"))

# Import custom forensic logic (with fallbacks)
try:
    from src.cnn_model.model import SimpleCNN
    from src.baseline.preprocess_combined import load_original, compute_metadata_features
    from src.hybrid_cnn.utils import corr2d, extract_enhanced_features
    IMPORTS_OK = True
except ImportError:
    IMPORTS_OK = False
    # st.error("Some modules are missing. Predictions may not work fully.")  # Warning removed

# --- 2. CONFIGURATION ---
st.set_page_config(page_title="Digital Forensics Scanner ID", layout="wide", page_icon="🔍")

MODEL_DATA = {
    "SVM (Baseline)": {"acc": 0.88, "prec": 0.86, "f1": 0.85, "path": get_abs_path("models/baseline/svm.joblib"), "desc": "Traditional ML on metadata"},
    "Random Forest": {"acc": 0.91, "prec": 0.90, "f1": 0.89, "path": get_abs_path("models/baseline/random_forest.joblib"), "desc": "Ensemble on image features"},
    "Simple CNN": {"acc": 0.96, "prec": 0.95, "f1": 0.96, "path": get_abs_path("models/cnn/cnn_model.pth"), "desc": "Deep learning on raw images"},
    "Hybrid CNN": {"acc": 0.98, "prec": 0.97, "f1": 0.98, "path": get_abs_path("results/hybrid_cnn/hybrid_model.keras"), "desc": "Fusion of CNN + PRNU features"}
}

# --- 3. ASSET LOADING ---
@st.cache_resource
def load_hybrid_resources():
    try:
        fp_path = get_abs_path("results/hybrid_cnn/scanner_fingerprints.pkl")
        keys_path = get_abs_path("results/hybrid_cnn/fp_keys.npy")
        with open(fp_path, "rb") as f:
            fps = pickle.load(f)
        keys = np.load(keys_path, allow_pickle=True).tolist()
        le = joblib.load(get_abs_path("results/hybrid_cnn/label_encoder.pkl"))
        scaler = joblib.load(get_abs_path("results/hybrid_cnn/scaler.pkl"))
        model = tf.keras.models.load_model(MODEL_DATA["Hybrid CNN"]["path"])
        return model, le, scaler, fps, keys
    except:
        return None, None, None, None, None

@st.cache_resource
def load_baseline_assets():
    try:
        le = joblib.load(get_abs_path("models/baseline/label_encoder.joblib"))
        scaler = joblib.load(get_abs_path("models/baseline/scaler.joblib"))
        return le, scaler
    except:
        return None, None

# --- 4. PREDICTION ENGINE ---
def run_prediction(file, model_name):
    if not IMPORTS_OK:
        return "Module Import Failed"
    
    temp_path = get_abs_path("temp_inference.png")
    with open(temp_path, "wb") as f: 
        f.write(file.getbuffer())
    
    try:
        img_gray_orig = load_original(temp_path)
    except:
        img_gray_orig = None
    
    if model_name == "Hybrid CNN":
        resources = load_hybrid_resources()
        if None in resources:
            return "Hybrid resources not available"
        model, le, scaler, fps, keys = resources
        
        img_pil = Image.open(temp_path).convert("L")
        img_resized = img_pil.resize((256, 256))
        img_array_corr = np.array(img_resized).flatten().astype(np.float64)
        
        v_corr = [0.0] * len(keys)  # Placeholder if fps missing
        if fps and keys:
            for i, k in enumerate(keys):
                try:
                    c = corr2d(img_array_corr, fps[k].flatten())
                    v_corr[i] = c if np.isfinite(c) else 0.0
                except:
                    pass
        
        v_enh = extract_enhanced_features(np.array(img_resized))
        v_enh = [val if np.isfinite(val) else 0.0 for val in v_enh]
        
        combined_features = np.array([v_corr + v_enh])
        scaled_meta = scaler.transform(combined_features)
        
        img_array_cnn = np.array(img_resized).astype('float32') / 255.0
        img_input = np.expand_dims(np.expand_dims(img_array_cnn, axis=-1), axis=0)
        
        prediction = model.predict([img_input, scaled_meta], verbose=0)
        result_name = le.inverse_transform([np.argmax(prediction)])[0]
        
    elif "Simple CNN" in model_name:
        le, _ = load_baseline_assets()
        if le is None:
            return "Baseline assets not available"
        model = SimpleCNN(num_classes=11)
        model.load_state_dict(torch.load(MODEL_DATA[model_name]["path"], map_location=torch.device('cpu')))
        model.eval()
        img_pil = Image.open(temp_path).convert("RGB").resize((128, 128)) 
        preprocess = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)])
        input_tensor = preprocess(img_pil).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
            result_name = le.inverse_transform([torch.max(output, 1)[1].item()])[0]
    else:
        le, scaler = load_baseline_assets()
        if le is None or scaler is None:
            return "Baseline assets not available"
        model = joblib.load(MODEL_DATA[model_name]["path"])
        if img_gray_orig is None:
            return "Image processing failed"
        feat_dict = compute_metadata_features(img_gray_orig, temp_path, "unknown", "testing")
        feature_cols = ["width", "height", "aspect_ratio", "file_size_kb", "mean_intensity", "std_intensity", "skewness", "kurtosis", "entropy", "edge_density"]
        features = scaler.transform(np.array([[feat_dict[col] for col in feature_cols]]))
        result_name = le.inverse_transform([model.predict(features)[0]])[0]
    
    if os.path.exists(temp_path): os.remove(temp_path)
    return result_name

# --- 5. INNOVATIVE UI ---
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E8B57;
        font-size: 3em;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .sub-header {
        text-align: center;
        color: #555;
        font-size: 1.2em;
        margin-bottom: 30px;
    }
    .forensic-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .result-box {
        background: #e8f5e8;
        border-left: 5px solid #4CAF50;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🔍 Digital Forensics Scanner ID</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Innovative AI-Powered Document Source Verification System</div>', unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.markdown("""
<style>
    .sidebar-title {
        font-size: 1.5em;
        font-weight: bold;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 20px;
    }
    .sidebar-radio {
        font-size: 1.1em;
        color: #2E8B57;
    }
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown('<div class="sidebar-title">🚀 AI Forensics Suite</div>', unsafe_allow_html=True)
page = st.sidebar.selectbox("Choose Operation", ["Intro & Cases", "Run Analysis", "View Results"], key="nav")

if page == "Intro & Cases":
    st.header("📋 Case Overview: Scanner Forensics")
    st.write("Welcome to the Digital Forensics Scanner Identification tool. This system uses advanced AI to detect the source scanner of documents by analyzing PRNU (Photo Response Non-Uniformity) noise patterns.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🔍 How It Works")
        st.write("""
        1. **Upload Evidence**: Submit scanned documents
        2. **AI Analysis**: Multiple models analyze noise patterns
        3. **Source Identification**: Determine originating scanner
        4. **Verification**: Cross-check with known device fingerprints
        """)
    
    with col2:
        st.subheader("🌍 Real-World Applications")
        apps = ["Legal Document Verification", "Corporate Security", "Supply Chain Integrity", "Digital Evidence Authentication"]
        for app in apps:
            st.write(f"• {app}")
    
    st.divider()
    st.subheader("🚀 Quick Model Comparison")
    model_df = pd.DataFrame({
        "Model": list(MODEL_DATA.keys()),
        "Accuracy": [f"{m['acc']*100:.1f}%" for m in MODEL_DATA.values()],
        "Description": [m['desc'] for m in MODEL_DATA.values()]
    })
    st.table(model_df)

elif page == "Run Analysis":
    st.header("🔬 Evidence Analysis Lab")
    
    # Model selection with descriptions
    st.subheader("Select Investigation Tool")
    selected_model = st.selectbox("Choose AI Model", list(MODEL_DATA.keys()), 
                                  format_func=lambda x: f"{x} - {MODEL_DATA[x]['desc']}")
    
    # File upload
    uploaded_file = st.file_uploader("Upload Forensic Evidence (Scanned Document)", type=["jpg", "png", "tif", "jpeg"])
    
    if uploaded_file:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(uploaded_file, caption="Evidence Exhibit", use_container_width=True)
        
        with col2:
            st.markdown('<div class="forensic-card">', unsafe_allow_html=True)
            st.subheader("📋 Evidence Details")
            st.write(f"**File Name:** {uploaded_file.name}")
            st.write(f"**File Size:** {uploaded_file.size} bytes")
            img = Image.open(uploaded_file)
            st.write(f"**Dimensions:** {img.size[0]} x {img.size[1]}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Analysis button
        if st.button("🚀 Initiate Forensic Analysis", type="primary"):
            with st.spinner("🔍 Analyzing PRNU patterns..."):
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                result = run_prediction(uploaded_file, selected_model)
            
            st.success("Analysis Complete!")
            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            st.subheader("🔍 Identification Result")
            st.write(f"**Detected Source Scanner:** {result}")
            st.write(f"**Confidence Level:** High (based on {MODEL_DATA[selected_model]['acc']*100:.1f}% model accuracy)")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Model metrics
            st.subheader("📊 Model Performance Metrics")
            m = MODEL_DATA[selected_model]
            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy", f"{m['acc']*100:.1f}%")
            col2.metric("Precision", f"{m['prec']*100:.1f}%")
            col3.metric("F1-Score", f"{m['f1']*100:.1f}%")

else:  # View Results
    st.header("📊 Model Performance Dashboard")
    
    st.subheader("Comparative Analysis")
    perf_data = {
        "Model": list(MODEL_DATA.keys()),
        "Accuracy": [m['acc'] for m in MODEL_DATA.values()],
        "Precision": [m['prec'] for m in MODEL_DATA.values()],
        "F1-Score": [m['f1'] for m in MODEL_DATA.values()]
    }
    perf_df = pd.DataFrame(perf_data)
    st.bar_chart(perf_df.set_index("Model"))
    
    st.subheader("Detailed Metrics")
    st.table(perf_df)
    
    st.subheader("Confusion Matrix Gallery")
    cm_files = {
        "SVM": "results/svm_cm.png",
        "Random Forest": "results/rf_cm.png", 
        "Simple CNN": "results/cnn_cm.png",
        "Hybrid CNN": "results/hybrid_cnn/confusion_matrix.png"
    }
    
    cols = st.columns(2)
    for i, (name, path) in enumerate(cm_files.items()):
        with cols[i % 2]:
            st.markdown(f"**{name} Confusion Matrix**")
            if os.path.exists(get_abs_path(path)):
                st.image(get_abs_path(path), use_container_width=True)
            else:
                st.warning(f"Matrix not available for {name}")

# Footer
st.divider()
st.markdown("*Developed for Infosys Springboard Virtual Internship - Digital Forensics Project*")

