import streamlit as st

st.title("📊 Model Results Summary (Text Only)")

st.markdown("""
This page provides a **simple summary** of the performance of the baseline models  
(Random Forest and SVM) used in the TraceFinder system.

Even though confusion matrix images are not displayed here,  
you can still understand the model behavior from the text summary below.
""")

# -------------------------
# Simple Summary Section
# -------------------------

st.subheader("📌 Random Forest Classifier")
st.write("""
- Model: **RandomForestClassifier**
- Input Features: Metadata features (dimensions, entropy, skewness, kurtosis, edges, etc.)
- Output: Scanner class label  
- Confusion Matrix File (Optional): `Random_Forest_confusion_matrix.png`
- Strengths:
  - Good performance on high-dimensional metadata
  - Fast training time
""")

st.subheader("📌 SVM Classifier (RBF Kernel)")
st.write("""
- Model: **Support Vector Machine (RBF)**
- Input Features: Same metadata as RF
- Output: Scanner class label  
- Confusion Matrix File (Optional): `SVM_confusion_matrix.png`
- Strengths:
  - Works well with non-linear separations
  - Often more accurate on fine-grained differences
""")

