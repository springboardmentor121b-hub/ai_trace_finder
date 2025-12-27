import streamlit as st

st.markdown("""
<style>
.card {
    background-color: #EEF1FF;
    padding: 1.8rem;
    border-radius: 18px;
    box-shadow: 0px 12px 30px rgba(90,100,180,0.12);
    margin-bottom: 1.6rem;
}
.badge {
    background-color: #6C7BFF;
    color: white;
    padding: 5px 14px;
    border-radius: 18px;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="card">
<h2>📌 About TraceFinder</h2>
<span class="badge">Hybrid CNN Core</span>
<p>
TraceFinder is a machine learning–based forensic system that identifies
the source scanner of a scanned image by analyzing scanner-specific artifacts.
The system is centered around a <b>Hybrid CNN architecture</b> that combines
deep learning with handcrafted forensic features.
</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="card">
<h3>🎯 Project Objective</h3>
<p>
The goal is to identify the scanner device used to scan a document by learning
noise residuals, frequency patterns, and texture characteristics that are
unique to each scanner model.
</p>
</div>
""", unsafe_allow_html=True)

with st.expander("🧠 Why Hybrid CNN?"):
    st.markdown("""
    <ul>
    <li>Combines PRNU-based noise learning with CNN features</li>
    <li>Improves robustness over standalone CNN or ML models</li>
    <li>Better generalization across scanners and resolutions</li>
    </ul>
    """)

st.markdown("""
<div class="card">
<h3>⚙️ System Modules</h3>
<ul>
<li>Data Collection and Labeling</li>
<li>Image Preprocessing and Noise Residual Extraction</li>
<li>Hybrid CNN Feature Learning</li>
<li>Classical ML Comparison (SVM, Random Forest)</li>
<li>Interactive Prediction Interface</li>
</ul>
</div>
""", unsafe_allow_html=True)
