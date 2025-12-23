import streamlit as st

st.title("About Project")

st.markdown("""
<div class="card">
<h3>Project Overview</h3>
<p>
TraceFinder is a machine learning–based system designed to identify the source
scanner of a scanned image by analyzing scanner-specific artifacts.
The system integrates deep learning and classical machine learning models
to achieve reliable scanner attribution.
</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="card">
<h3>Project Statement</h3>
<p>
The objective of TraceFinder is to identify the source scanner device used to scan
a document or image by analyzing noise patterns, texture variations, and
frequency-domain characteristics. Each scanner leaves unique traces that can be
learned using machine learning and deep learning techniques.
</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="card">
<h3>Use Cases</h3>
<ul>
<li><b>Digital Forensics:</b> Identifying scanners used in forged or duplicated documents.</li>
<li><b>Document Authentication:</b> Verifying whether scanned copies originate from authorized devices.</li>
<li><b>Legal Evidence Verification:</b> Validating scanned documents submitted as legal evidence.</li>
</ul>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="card">
<h3>System Modules</h3>
<ul>
<li><b>Data Collection and Labeling:</b> Collection of scanned images from multiple scanner devices.</li>
<li><b>Image Preprocessing:</b> Resizing, denoising, grayscale conversion, and normalization.</li>
<li><b>Feature Extraction:</b> Noise residuals, FFT-based frequency features, and edge descriptors.</li>
<li><b>Model Training:</b> CNN for deep feature learning and SVM/Random Forest for handcrafted features.</li>
<li><b>Prediction Module:</b> Upload an image and identify the probable scanner source.</li>
</ul>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="card">
<h3>Dataset</h3>
<p>
The dataset consists of scanned document images collected from official sources
and online repositories. All images are resized to 128 × 128 resolution and
normalized before training. The dataset is divided into training, validation,
and testing subsets to ensure reliable evaluation.
</p>
</div>
""", unsafe_allow_html=True)
