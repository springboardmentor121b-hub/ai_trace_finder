# app_streamlit.py
# Streamlit landing page for TraceFinder with sticker headings + PDF + description + prediction demo
# Run from ai_trace_finder folder (with venv activated):
#   python -m streamlit run app_streamlit.py

import streamlit as st
from pathlib import Path
import base64
import os
import sys

import cv2
import numpy as np
import joblib
import pandas as pd
from skimage.filters import sobel
from scipy.stats import skew, kurtosis, entropy

# ===== CNN IMPORTS =====
import torch
import torch.nn.functional as F

st.set_page_config(page_title="TraceFinder — Landing", layout="wide", page_icon="🟣")

# ------ CONFIG -------
BASE_DIR = Path(__file__).parent.resolve()

# Models folder (ai_trace_finder/models)
MODEL_DIR = BASE_DIR / "models"

# ===== Baseline paths (if available) =====
SCALER_PATH = MODEL_DIR / "scaler.pkl"
RF_PATH = MODEL_DIR / "random_forest.pkl"
SVM_PATH = MODEL_DIR / "svm.pkl"

# ===== CNN PATH CONFIG =====
CNN_MODEL_PATH = MODEL_DIR / "cnn" / "cnn_model.pth"

# Add src to path and import CNN
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from cnn.model import SimpleCNN  # ensure this matches your class name

# ===== CNN CLASS LABELS (CURRENTLY PARTIAL, USED ONLY FOR TOP-1 LABEL) =====
CNN_CLASS_NAMES = [
    "Canon120-1",
    "Canon120-2",
    "Canon220",
    "Canon9000-1",
    "Canon9000-2",
    "EpsonV370-1",
    "EpsonV370-2",
    "EpsonV39-1",
    "EpsonV39-2",
    "EpsonV550",
    "HP",
    # remaining classes not listed; CNN still works for top-1 prediction
]

# Try local project PDF first, fallback to /mnt/data if available (useful for cloud)
DEFAULT_PDF_PATH = BASE_DIR / "AI_TraceFinder.pdf"
FALLBACK_PDF = Path("/mnt/data/AI_TraceFinder (1) (1).pdf")
if not DEFAULT_PDF_PATH.exists() and FALLBACK_PDF.exists():
    DEFAULT_PDF_PATH = FALLBACK_PDF

# Maximum bytes to safely embed as data URI (5 MB)
MAX_EMBED_BYTES = 5 * 1024 * 1024

# ------ HELPERS (PDF) -------
def load_pdf_bytes(path: Path):
    try:
        if path.exists() and path.is_file():
            return path.read_bytes(), path.name
    except Exception:
        pass
    return None, None


def pdf_to_data_uri(pdf_bytes: bytes) -> str:
    return "data:application/pdf;base64," + base64.b64encode(pdf_bytes).decode("utf-8")


# load default PDF
default_pdf_bytes, default_pdf_name = load_pdf_bytes(DEFAULT_PDF_PATH)

# ------ HELPERS (Prediction) -------
@st.cache_resource
def load_models():
    """Load RF/SVM baseline models if present."""
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Scaler not found at: {SCALER_PATH}")
    if not RF_PATH.exists():
        raise FileNotFoundError(f"Random Forest model not found at: {RF_PATH}")
    if not SVM_PATH.exists():
        raise FileNotFoundError(f"SVM model not found at: {SVM_PATH}")

    scaler = joblib.load(SCALER_PATH)
    rf = joblib.load(RF_PATH)
    svm = joblib.load(SVM_PATH)
    return scaler, rf, svm


# ===== CNN MODEL LOADER =====
@st.cache_resource
def load_cnn_model():
    """Load the trained CNN model."""
    if not CNN_MODEL_PATH.exists():
        raise FileNotFoundError(f"CNN model not found at: {CNN_MODEL_PATH}")

    NUM_CLASSES = 22  # must match training config
    device = torch.device("cpu")

    model = SimpleCNN(num_classes=NUM_CLASSES)
    state = torch.load(CNN_MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def load_and_preprocess_from_bytes(file_bytes: bytes, size=(512, 512)):
    """Decode uploaded image bytes to grayscale, normalize and resize."""
    file_array = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(file_array, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not decode uploaded image.")
    img = img.astype(np.float32) / 255.0
    img_resized = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img_resized


# ===== CNN PREPROCESSING =====
def preprocess_for_cnn(file_bytes: bytes, size=(128, 128)):
    """Prepare RGB image tensor for CNN (1 x 3 x H x W)."""
    file_array = np.frombuffer(file_bytes, np.uint8)
    img_bgr = cv2.imdecode(file_array, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Could not decode uploaded image for CNN.")

    img_bgr = cv2.resize(img_bgr, size, interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0  # H,W,3

    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0)  # 1,3,H,W
    return tensor


def compute_metadata_features_from_bytes(img, file_bytes: bytes):
    """Compute same metadata features used during training."""
    h, w = img.shape
    aspect_ratio = w / h if h != 0 else 0.0
    file_size_kb = len(file_bytes) / 1024.0 if file_bytes is not None else 0.0

    pixels = img.flatten()
    mean_intensity = float(np.mean(pixels))
    std_intensity = float(np.std(pixels))
    skewness = float(skew(pixels))
    kurt = float(kurtosis(pixels))

    # histogram for entropy
    counts = np.histogram(pixels, bins=256, range=(0, 1))[0].astype(np.float64) + 1e-6
    ent = float(entropy(counts))

    edges = sobel(img)
    edge_density = float(np.mean(edges > 0.1))

    # IMPORTANT: keep feature names identical to those used in metadata_features_for_training.csv
    return {
        "width": int(w),
        "height": int(h),
        "aspect_ratio": aspect_ratio,
        "file_size_kb": file_size_kb,
        "mean_intensity": mean_intensity,
        "std_intensity": std_intensity,
        "skewness": skewness,
        "kurtosis": kurt,
        "entropy": ent,
        "edge_density": edge_density,
    }


def run_prediction(uploaded_file, model_choice: str):
    """Run prediction for RF / SVM / CNN and return (label, probs, class_names)."""
    file_bytes = uploaded_file.getvalue()

    # ================= CNN =================
    if model_choice == "CNN":
        model = load_cnn_model()
        x = preprocess_for_cnn(file_bytes)

        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))

        # If class list is incomplete, fall back to index name
        if pred_idx < len(CNN_CLASS_NAMES):
            label = CNN_CLASS_NAMES[pred_idx]
        else:
            label = f"Class-{pred_idx}"

        # For now, do not expose per-class probabilities table,
        # only top-1 prediction (probs returned for possible future use).
        return label, None, None

    # ============== RF / SVM =================
    scaler, rf, svm = load_models()

    img = load_and_preprocess_from_bytes(file_bytes)
    features = compute_metadata_features_from_bytes(img, file_bytes)
    df_feat = pd.DataFrame([features])

    X_scaled = scaler.transform(df_feat)

    model = rf if model_choice == "Random Forest" else svm
    pred = model.predict(X_scaled)[0]

    probs = None
    class_names = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_scaled)[0]
        class_names = model.classes_

    return pred, probs, class_names


# theme toggle state (unique key)
if "theme" not in st.session_state:
    st.session_state.theme = "dark"
theme_choice = st.selectbox(
    "Theme",
    ["dark", "light"],
    index=0 if st.session_state.theme == "dark" else 1,
    key="theme_toggle_main_sticker",
)
st.session_state.theme = theme_choice

# ------ CSS (sticker badge + existing UI styles) -------
CSS_DARK = r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');

:root {
  --bg:#0b1220;
  --panel: rgba(255,255,255,0.03);
  --muted:#9fb0c6;
  --text:#eaf6ff;
  --accent1:#7c3aed;
  --accent2:#06b6d4;
  --accent3:#ff5c8a;
  --glass: rgba(255,255,255,0.04);
}

/* Sticker heading variables */
:root {
  --sticker-bg: linear-gradient(90deg, var(--accent2), var(--accent3));
  --sticker-color: #041;
}

/* base page */
html, body {
  background:
    radial-gradient(circle at 10% 10%, rgba(124,58,237,0.06), transparent 6%),
    radial-gradient(circle at 90% 90%, rgba(6,182,212,0.05), transparent 6%),
    var(--bg) !important;
  color:var(--text) !important;
  font-family: Inter, system-ui, -apple-system, 'Segoe UI', Roboto, Arial;
  margin:0; padding:0;
}
.wrap { max-width:1180px; margin:28px auto; padding:18px; }

/* Hero */
.hero { display:flex; align-items:center; justify-content:space-between; gap:16px; margin-bottom:18px; }
.brand { display:flex; gap:14px; align-items:center; }
.logo {
  width:78px; height:78px; border-radius:18px; display:flex; align-items:center; justify-content:center;
  background: linear-gradient(135deg,var(--accent1),var(--accent2)); font-weight:900; color:#042; font-size:20px;
  box-shadow: 0 18px 50px rgba(2,6,23,0.45); transform: rotate(-8deg);
  animation: logo-tilt 3s infinite ease-in-out;
}
@keyframes logo-tilt { 0% { transform: rotate(-8deg) translateY(0); } 50% { transform: rotate(-2deg) translateY(-6px); } 100% { transform: rotate(-8deg) translateY(0); } }
.title { font-size:30px; font-weight:900; margin:0; letter-spacing:-0.4px; }
.subtitle { color:var(--muted); margin-top:6px; font-size:13px; }

.right-actions { min-width:300px; border-radius:12px; padding:6px; text-align:right; }
.cta { display:inline-block; padding:9px 12px; border-radius:10px; font-weight:800; text-decoration:none; cursor:pointer;
      background: linear-gradient(90deg,var(--accent2),var(--accent3)); color:#041; box-shadow:0 10px 28px rgba(6,182,212,0.08); }

/* grid/cards */
.section { margin-top:22px; }
.grid { display:grid; grid-template-columns: repeat(3, 1fr); gap:16px; margin-top:12px; }
.card {
  background: var(--panel); border-radius:12px; padding:14px; position:relative; overflow:hidden;
  border: 1px solid rgba(255,255,255,0.02); transition: transform .32s ease, box-shadow .32s ease;
  box-shadow: 0 10px 26px rgba(2,6,23,0.10);
}
.card:hover { transform: translateY(-10px) scale(1.01); box-shadow: 0 36px 90px rgba(2,6,23,0.22); }

.icon-bubble { width:58px; height:58px; border-radius:12px; display:flex; align-items:center; justify-content:center; font-size:22px; font-weight:900;
  background: linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02)); box-shadow: 0 10px 26px rgba(2,6,23,0.18);
  animation: float 3s infinite ease-in-out;
}
@keyframes float { 0% { transform: translateY(0);} 50% { transform: translateY(-8px);} 100% { transform: translateY(0);} }

.card-title { font-size:15px; font-weight:800; margin-top:10px; }
.card-desc { color:var(--muted); margin-top:8px; font-size:13px; line-height:1.45; }

.ribbon { height:6px; border-radius:8px; margin-top:12px; background: linear-gradient(90deg,var(--accent1),var(--accent2),var(--accent3)); background-size:200% 100%; animation: slide 5s linear infinite; }
@keyframes slide { 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }

.shimmer { position:absolute; inset:0; pointer-events:none; background: linear-gradient(90deg, rgba(255,255,255,0.01), rgba(255,255,255,0.03) 50%, rgba(255,255,255,0.01)); transform:translateX(-140%); animation: shimmer 3.6s linear infinite; }
@keyframes shimmer { to { transform: translateX(140%); } }

/* ===== Option A: Sticker heading styles ===== */
.section-heading {
  display:flex;
  align-items:center;
  gap:14px;
  margin-top:28px;
  margin-bottom:8px;
}
.section-sticker {
  display:inline-block;
  padding:6px 14px;
  font-size:11px;
  font-weight:900;
  letter-spacing:1.4px;
  border-radius:6px;
  text-transform:uppercase;
  background: var(--sticker-bg);
  color: var(--sticker-color);
  box-shadow: 0 6px 20px rgba(6,182,212,0.14);
}
.section-title {
  margin:0;
  font-size:22px;
  font-weight:900;
  letter-spacing:-0.3px;
}

/* Flowchart styles (concise at end) */
.flowwrap { margin-top:28px; display:flex; align-items:center; justify-content:center; }
.flow { display:flex; align-items:center; gap:20px; flex-wrap:wrap; justify-content:center; }
.flow .step {
  background: linear-gradient(135deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
  color:var(--text);
  padding:12px 18px;
  border-radius:10px;
  font-weight:700;
  min-width:160px;
  text-align:center;
  position:relative;
  transition: transform .25s ease, box-shadow .25s ease;
  border:1px solid rgba(255,255,255,0.04);
  box-shadow: 0 8px 30px rgba(2,6,23,0.12);
}
.flow .step:hover {
  transform: translateY(-8px) scale(1.03);
  box-shadow: 0 30px 100px rgba(2,6,23,0.28);
}
.flow .arrow {
  width:48px; height:2px; background: linear-gradient(90deg,var(--accent2),var(--accent3));
  position:relative; display:inline-block; transition: transform .3s ease;
}
.flow .arrow:after {
  content: '';
  position:absolute;
  right:-6px; top:-6px;
  border-width:8px; border-style:solid;
  border-color: transparent transparent transparent var(--accent3);
  transform: rotate(0deg);
  filter: drop-shadow(0 4px 8px rgba(2,6,23,0.12));
}
.flow .arrow.pulse { animation: pulse 2s infinite; }
@keyframes pulse { 0% { transform: translateX(0) scaleX(1); } 50% { transform: translateX(3px) scaleX(1.05);} 100% { transform: translateX(0) scaleX(1); } }

@media (max-width:900px) {
  .grid { grid-template-columns: repeat(2, 1fr); }
  .flow { gap:12px; }
  .flow .step { min-width:140px; padding:10px 12px; font-size:14px; }
  .flow .arrow { width:36px; }
}

/* description bullets under title */
.desc-box {
  margin-top: 4px;
  margin-bottom: 14px;
  max-width: 780px;
  font-size: 13px;
  color: var(--muted);
  line-height: 1.55;
}
.desc-box ul {
  padding-left: 18px;
  margin: 6px 0;
}
</style>
"""

CSS_LIGHT = r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');

:root {
  --bg:#f7fbff;
  --panel:#ffffff;
  --muted:#475569;
  --text:#071028;
  --accent1:#5b21b6;
  --accent2:#06b6d4;
  --accent3:#ff4d6d;
  --glass: rgba(0,0,0,0.06);
}

/* Sticker heading variables */
:root {
  --sticker-bg: linear-gradient(90deg, var(--accent2), var(--accent3));
  --sticker-color: #fff;
}

html, body { background: var(--bg) !important; color:var(--text) !important; font-family: Inter, system-ui, -apple-system, 'Segoe UI', Roboto, Arial; margin:0; padding:0; }
.wrap { max-width:1180px; margin:28px auto; padding:18px; }

.hero { display:flex; align-items:center; justify-content:space-between; gap:16px; margin-bottom:18px; }
brand { display:flex; gap:14px; align-items:center; }
.logo { width:78px; height:78px; border-radius:18px; display:flex; align-items:center; justify-content:center;
  background: linear-gradient(135deg,var(--accent1),var(--accent2)); font-weight:900; color:#fff; font-size:20px;
  box-shadow: 0 18px 50px rgba(2,6,23,0.06); transform: rotate(-8deg); animation: logo-tilt 3s infinite ease-in-out; }
@keyframes logo-tilt { 0% { transform: rotate(-8deg) translateY(0); } 50% { transform: rotate(-2deg) translateY(-6px); } 100% { transform: rotate(-8deg) translateY(0); } }
.title { font-size:30px; font-weight:900; margin:0; letter-spacing:-0.4px; }
.subtitle { color:var(--muted); margin-top:6px; font-size:13px }

right-actions { min-width:300px; border-radius:12px; padding:6px; text-align:right; }
.cta { display:inline-block; padding:9px 12px; border-radius:10px; font-weight:800; text-decoration:none; cursor:pointer;
      background: linear-gradient(90deg,var(--accent2),var(--accent3)); color:#fff; box-shadow:0 10px 28px rgba(6,182,212,0.06); }

/* grid/cards */
.section { margin-top:22px; }
.grid { display:grid; grid-template-columns: repeat(3, 1fr); gap:16px; margin-top:12px; }
.card { background: var(--panel); border-radius:12px; padding:14px; position:relative; overflow:hidden; border: 1px solid rgba(0,0,0,0.04); transition: transform .32s ease, box-shadow .32s ease; box-shadow: 0 10px 26px rgba(2,6,23,0.04); }
.card:hover { transform: translateY(-10px) scale(1.01); box-shadow: 0 36px 90px rgba(2,6,23,0.08); }

.icon-bubble { width:58px; height:58px; border-radius:12px; display:flex; align-items:center; justify-content:center; font-size:22px; font-weight:900; background: linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02)); box-shadow: 0 10px 26px rgba(2,6,23,0.06); animation: float 3s infinite ease-in-out; }
@keyframes float { 0% { transform: translateY(0);} 50% { transform: translateY(-8px);} 100% { transform: translateY(0);} }

.card-title { font-size:15px; font-weight:800; margin-top:10px; }
.card-desc { color:var(--muted); margin-top:8px; font-size:13px; line-height:1.45; }

.ribbon { height:6px; border-radius:8px; margin-top:12px; background: linear-gradient(90deg,var(--accent1),var(--accent2),var(--accent3)); background-size:200% 100%; animation: slide 5s linear infinite; }
@keyframes slide { 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }

.shimmer { position:absolute; inset:0; pointer-events:none; background: linear-gradient(90deg, rgba(255,255,255,0.01), rgba(255,255,255,0.03) 50%, rgba(255,255,255,0.01)); transform:translateX(-140%); animation: shimmer 3.6s linear infinite; }
@keyframes shimmer { to { transform: translateX(140%); } }

/* description bullets under title */
.desc-box {
  margin-top: 4px;
  margin-bottom: 14px;
  max-width: 780px;
  font-size: 13px;
  color: var(--muted);
  line-height: 1.55;
}
.desc-box ul {
  padding-left: 18px;
  margin: 6px 0;
}
</style>
"""

# apply chosen CSS
st.markdown(CSS_DARK if st.session_state.theme == "dark" else CSS_LIGHT, unsafe_allow_html=True)

# ------ Page content -------
st.markdown('<div class="wrap">', unsafe_allow_html=True)

# HERO
st.markdown(
    """
<div class="hero">
  <div class="brand">
    <div class="logo">TF</div>
    <div>
      <div class="title">TraceFinder</div>
      <div class="subtitle">Scanner & Camera Source Identification — PRNU + Metadata</div>
    </div>
  </div>
  <div class="right-actions">
    <div style="font-size:13px; color:var(--muted); margin-bottom:6px;">Project brief</div>
    <div class="cta">Landing • Features • Use-cases</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ---- 6-point project description (just below title) ----
desc_html = """
<div class="desc-box">
  <ul>
    <li>Identify the scanner brand/model used to produce a scanned document by analysing device-specific noise patterns.</li>
    <li>Support digital forensics, document authentication and legal evidence verification workflows.</li>
    <li>Extract scanner-specific features such as noise residuals, entropy, edge density and other metadata signals.</li>
    <li>Train machine learning models (Random Forest &amp; SVM) to distinguish between multiple scanner devices.</li>
    <li>Evaluate performance using accuracy, precision, recall, F1-score and confusion matrix visualisations.</li>
    <li>Provide a simple UI where users upload a scan and receive the predicted scanner model with confidence.</li>
  </ul>
</div>
"""
st.markdown(desc_html, unsafe_allow_html=True)

# PDF viewer + button
st.markdown(
    '<div style="display:flex; gap:14px; align-items:center; margin-bottom:12px;">',
    unsafe_allow_html=True,
)
if default_pdf_bytes:
    pdf_uri = pdf_to_data_uri(default_pdf_bytes[:MAX_EMBED_BYTES])
    st.markdown(
        f'<a class="cta" href="{pdf_uri}" target="_blank">Open Project Brief (PDF)</a>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div style="color:#f3c; font-weight:700;">Project PDF not found in project folder.</div>',
        unsafe_allow_html=True,
    )
st.markdown("</div>", unsafe_allow_html=True)

# FEATURES
st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown(
    """
  <div class="section-heading">
    <div class="section-sticker">FEATURES</div>
    <h3 class="section-title">Core Capabilities</h3>
  </div>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="grid">', unsafe_allow_html=True)

features = [
    ("🔬", "Noise Residual Extraction", "Wavelet denoising + residual subtraction to reveal sensor PRNU patterns."),
    ("📊", "Compact Metadata Set", "Entropy, skewness, edge density and file-size metrics for explainable signals."),
    ("🧩", "Patch-based Multi-Scale", "Overlapping patches to boost sensor fingerprint and reduce content leakage."),
    ("⚡", "Fast CPU Inference", "Random Forest baseline for quick, low-cost predictions and prototypes."),
]
for icon, title, desc in features:
    st.markdown(
        f"""
        <div class="card">
          <div class="icon-bubble">{icon}</div>
          <div class="card-title">{title}</div>
          <div class="card-desc">{desc}</div>
          <div class="ribbon"></div>
          <div class="shimmer"></div>
        </div>
    """,
        unsafe_allow_html=True,
    )

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# USE CASES
st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown(
    """
  <div class="section-heading">
    <div class="section-sticker">USE CASES</div>
    <h3 class="section-title">Real-World Applications</h3>
  </div>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="grid">', unsafe_allow_html=True)

uses = [
    ("🛡️", "Digital Forensics", "Attribute scanned evidence to specific scanner models for investigations."),
    ("🏦", "Banking & KYC", "Detect reused or forged scans in identity verification flows."),
    ("🏛️", "Government Services", "Validate IDs and certificates to prevent fraud at scale."),
    ("🔬", "Research", "Benchmark PRNU/device-identification methods and datasets."),
]
for icon, title, desc in uses:
    st.markdown(
        f"""
        <div class="card">
          <div class="icon-bubble">{icon}</div>
          <div class="card-title">{title}</div>
          <div class="card-desc">{desc}</div>
          <div class="ribbon"></div>
          <div class="shimmer"></div>
        </div>
    """,
        unsafe_allow_html=True,
    )

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ADVANTAGES
st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown(
    """
  <div class="section-heading">
    <div class="section-sticker">ADVANTAGES</div>
    <h3 class="section-title">Why TraceFinder Stands Out</h3>
  </div>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="grid">', unsafe_allow_html=True)

advantages = [
    ("🏆", "High baseline accuracy", "Strong per-scanner performance using residual + metadata features."),
    ("🔍", "Explainable", "Feature-level evidence supports forensic reporting and traceability."),
    ("🔐", "Privacy-first", "Offline processing suitable for sensitive documents and workflows."),
    ("🔧", "Extensible", "Easily swap classifiers or add scanner classes as datasets grow."),
]
for icon, title, desc in advantages:
    st.markdown(
        f"""
        <div class="card">
          <div class="icon-bubble">{icon}</div>
          <div class="card-title">{title}</div>
          <div class="card-desc">{desc}</div>
          <div class="ribbon"></div>
          <div class="shimmer"></div>
        </div>
    """,
        unsafe_allow_html=True,
    )

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ---------- PREDICTION DEMO SECTION ----------
st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown(
    """
  <div class="section-heading">
    <div class="section-sticker">PREDICTION</div>
    <h3 class="section-title">Try TraceFinder on Your Own Scan</h3>
  </div>
""",
    unsafe_allow_html=True,
)

col_left, col_right = st.columns([1.1, 1])

with col_left:
    st.markdown(
        """
        **How prediction works:**
        1. Upload a scanned document image (TIFF / PNG / JPEG).
        2. The app converts it to grayscale and resizes to 512×512.
        3. Scanner-specific metadata features are extracted:
           - width, height, aspect ratio  
           - file size (KB), mean & std intensity  
           - skewness, kurtosis, entropy  
           - edge density from Sobel filter
        4. Features are normalized using the same `StandardScaler` used in training.
        5. A trained model (Random Forest, SVM, or CNN) predicts the scanner class.
        6. You see the predicted scanner and, for RF/SVM, per-class probability distribution.
        """
    )

with col_right:
    st.markdown("**Upload a scan and choose a model**")
    uploaded = st.file_uploader(
        "Upload scanned image",
        type=["tif", "tiff", "png", "jpg", "jpeg"],
        key="predict_upload",
    )
    model_choice = st.radio(
        "Select model",
        ["Random Forest", "SVM", "CNN"],
        horizontal=True,
        key="predict_model_choice",
    )

if uploaded is not None:
    # Preview image
    try:
        file_bytes = uploaded.getvalue()
        arr = np.frombuffer(file_bytes, np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is not None:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, caption="Uploaded scan")
    except Exception:
        pass

    if st.button("Run Prediction", type="primary"):

        # Block RF/SVM if models missing
        if model_choice != "CNN" and not SCALER_PATH.exists():
            st.error("RF / SVM models not found. Please select CNN.")
            st.stop()

        try:
            pred_label, probs, class_names = run_prediction(uploaded, model_choice)
            st.success(f"Predicted Scanner: **{pred_label}**")

            # RF/SVM: show probability table; CNN: we returned probs=None, class_names=None
            if probs is not None and class_names is not None:
                prob_df = (
                    pd.DataFrame(
                        {
                            "Scanner": list(class_names),
                            "Probability": list(probs),
                        }
                    )
                    .sort_values("Probability", ascending=False)
                )
                st.markdown("**Per-class probabilities:**")
                st.dataframe(prob_df, width="content")
            elif model_choice == "CNN":
                st.info("CNN currently shows only the top predicted scanner (probability table hidden).")
            else:
                st.info("This model does not expose probability scores.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
else:
    st.info("Upload a scanned image to run prediction.")

# Flowchart
flow_html = """
<div class="flowwrap">
  <div class="flow" aria-label="Project flowchart">
    <div class="step" title="1. Data Collection">1. Data Collection<br><small>Scan & label</small></div>
    <div class="arrow pulse" role="img" aria-hidden="true"></div>
    <div class="step" title="2. Preprocessing">2. Preprocessing<br><small>Resize, denoise</small></div>
    <div class="arrow pulse" role="img" aria-hidden="true"></div>
    <div class="step" title="3. Feature Extraction">3. Feature Extraction<br><small>PRNU, metadata</small></div>
    <div class="arrow pulse" role="img" aria-hidden="true"></div>
    <div class="step" title="4. Model Training">4. Model Training<br><small>RF / SVM / CNN</small></div>
    <div class="arrow pulse" role="img" aria-hidden="true"></div>
    <div class="step" title="5. Deploy & UI">5. Deploy & UI<br><small>Streamlit app</small></div>
  </div>
</div>
"""
st.markdown(flow_html, unsafe_allow_html=True)

# close wrapper div
st.markdown("</div>", unsafe_allow_html=True)
