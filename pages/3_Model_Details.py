import streamlit as st
import pandas as pd
import os

st.set_page_config(
    page_title="Model Details | TraceFinder",
    layout="wide"
)

st.markdown("""
<style>
body {
    background-color: #F4F6FB;
}
.card {
    background-color: #E8ECF7;
    padding: 1.8rem;
    border-radius: 16px;
    box-shadow: 0px 8px 22px rgba(0,0,0,0.06);
    margin-bottom: 1.5rem;
}
h2, h3 {
    color: #2E2F4F;
}
pre {
    background-color: #F7F8FC;
    padding: 1rem;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

BASE_DIR = r"D:\Project\TraceFinder"
RESULTS_DIR = os.path.join(BASE_DIR, "results")

st.markdown("""
<div class="card">
<h2>📊 Model Details and Performance Analysis</h2>
<p>
This page presents the evaluation results obtained after training and testing
different machine learning and deep learning models for scanner identification.
All results shown here are generated experimentally.
</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="card">
<h3>📈 Experimental Accuracy Summary</h3>
<p>
The following accuracies are calculated from test data using confusion matrices
and classification reports generated during evaluation.
</p>
</div>
""", unsafe_allow_html=True)

accuracy_data = {
    "Model": [
        "CNN",
        "Support Vector Machine (SVM)",
        "Random Forest",
        "Hybrid CNN"
    ],
    "Test Accuracy (%)": [
        66,
        32,
        34,
        71
    ]
}

acc_df = pd.DataFrame(accuracy_data)

st.dataframe(acc_df, use_container_width=True)
st.bar_chart(acc_df.set_index("Model"), use_container_width=True)

st.markdown("""
<div class="card">
<h3>🧠 CNN Model Results</h3>
</div>
""", unsafe_allow_html=True)

cnn_dir = os.path.join(RESULTS_DIR, "cnn")

cnn_cm_img = os.path.join(cnn_dir, "confusion_matrix.png")
cnn_report = os.path.join(cnn_dir, "classification_report.txt")
cnn_arch = os.path.join(cnn_dir, "architecture.txt")

if os.path.exists(cnn_cm_img):
    st.image(cnn_cm_img, caption="CNN Confusion Matrix", use_container_width=True)

if os.path.exists(cnn_report):
    with open(cnn_report, "r") as f:
        st.subheader("CNN Classification Report")
        st.text(f.read())

with st.expander("CNN Architecture Details"):
    if os.path.exists(cnn_arch):
        with open(cnn_arch, "r") as f:
            st.text(f.read())
    else:
        st.info("CNN architecture file not found.")

st.markdown("""
<div class="card">
<h3>🚀 Hybrid CNN Model Results</h3>
<p>
Hybrid CNN combines deep CNN features with handcrafted forensic cues such as
noise residuals and frequency-based information.
</p>
</div>
""", unsafe_allow_html=True)

hybrid_dir = os.path.join(RESULTS_DIR, "hybrid_cnn")

hybrid_cm = os.path.join(hybrid_dir, "confusion_matrix.png")
hybrid_train_plot = os.path.join(hybrid_dir, "training_plot.png")

if os.path.exists(hybrid_train_plot):
    st.image(
        hybrid_train_plot,
        caption="Hybrid CNN Training Curve",
        use_container_width=True
    )

if os.path.exists(hybrid_cm):
    st.image(
        hybrid_cm,
        caption="Hybrid CNN Confusion Matrix",
        use_container_width=True
    )

st.markdown("""
<div class="card">
<h3>🌲 Random Forest Model Results</h3>
</div>
""", unsafe_allow_html=True)

rf_dir = os.path.join(RESULTS_DIR, "svm_Random_forest")
rf_cm = os.path.join(rf_dir, "confusion_matrix_rf.png")

if os.path.exists(rf_cm):
    st.image(
        rf_cm,
        caption="Random Forest Confusion Matrix",
        use_container_width=True
    )
else:
    st.warning("Random Forest confusion matrix not found.")

st.markdown("""
<div class="card">
<h3>🔍 Key Observations</h3>
<ul>
<li>Hybrid CNN achieved the highest accuracy of 71% due to combined feature learning.</li>
<li>Standalone CNN performed competitively but lacked handcrafted forensic cues.</li>
<li>Random Forest provided moderate accuracy using handcrafted features.</li>
<li>SVM achieved fast inference but lower robustness.</li>
</ul>
</div>
""", unsafe_allow_html=True)
