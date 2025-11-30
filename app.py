import streamlit as st
import cv2
import numpy as np
import joblib
import os
from scipy.stats import skew, kurtosis

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
MODEL_PATH = os.path.join(MODEL_DIR, "random_forest.pkl")

scaler = joblib.load(SCALER_PATH)
model = joblib.load(MODEL_PATH)

label_map = {
    0: "HP",
    1: "Canon 120-1",
    2: "Canon 120-2",
    3: "Canon 220",
    4: "Canon 8000-1",
    5: "Canon 8000-2",
    6: "Epson V39-1",
    7: "Epson V39-2",
    8: "Epson V370-1",
    9: "Epson V370-2",
    10: "Epson V550"
}

def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    mean_val = np.mean(gray)
    std_val = np.std(gray)
    skew_val = skew(gray.flatten())
    kurt_val = kurtosis(gray.flatten())

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_mean = np.mean(lap)
    lap_std = np.std(lap)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edges = np.sqrt(sobelx**2 + sobely**2)

    edge_mean = np.mean(edges)
    edge_std = np.std(edges)

    height, width = gray.shape

    features = [
        mean_val, std_val, skew_val, kurt_val,
        lap_mean, lap_std,
        edge_mean, edge_std,
        width, height
    ]

    return np.array(features).reshape(1, -1)

st.set_page_config(page_title="AI TraceFinder", layout="wide")

st.title("AI TraceFinder – Intelligent Device Identification System")

tab1, tab2 = st.tabs(["Prediction", "About Project"])

with tab1:
    st.subheader("Upload a Scanned Image")
    uploaded_file = st.file_uploader("Upload JPG, PNG or TIFF image", type=["jpg", "jpeg", "png", "tiff"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        st.image(image, caption="Uploaded Image", use_container_width=True)

        features = extract_features(image)
        features = scaler.transform(features)

        prediction = model.predict(features)[0]
        pred_class = int(prediction)
        folder_name = label_map.get(pred_class, f"Unknown Scanner (Class {pred_class})")


        st.success(f"Predicted Scanner: {folder_name}")

with tab2:
    st.markdown("""
    ## AI TraceFinder Intelligent Identification System

    ### Project Overview
    AI TraceFinder is an intelligent image-based forensic analysis system designed to identify the source device (scanner or printer) used to generate a digital image. The system uses image processing and machine learning to extract statistical and texture-based features from scanned images and predicts the corresponding device brand and model.

    ### Problem Statement
    With the rapid growth of digital documentation, verifying the origin of scanned documents has become increasingly difficult. Unauthorized reproduction, forgery, and digital tampering pose serious threats to legal, academic, and government systems. There is a strong need for an automated system capable of identifying the original scanner device used.

    ### Proposed Solution
    AI TraceFinder analyzes microscopic image artifacts using statistical and texture-based analysis and applies machine learning algorithms to accurately predict the scanning device.

    ### Features Used for Prediction
    1. Mean intensity  
    2. Standard deviation  
    3. Skewness  
    4. Kurtosis  
    5. Laplacian mean  
    6. Laplacian standard deviation  
    7. Edge magnitude mean  
    8. Edge magnitude standard deviation  
    9. Image width  
    10. Image height  

    ### Machine Learning Model
    Primary Model: Random Forest Classifier  
    Alternative Model: Support Vector Machine  
    Feature Scaling: StandardScaler  
    Model Storage: Joblib Serialization

    ### Technology Stack
    Python 3.11  
    OpenCV  
    NumPy  
    SciPy  
    Scikit-learn  
    Streamlit  
    Visual Studio Code  

    ### Applications
    Digital forensic investigations  
    Legal document verification  
    Certificate authentication  
    Corporate fraud detection  
    Government record validation  

    ### Advantages
    Automated identification  
    High prediction accuracy  
    Real-time web interface  
    Easy scalability  

    ### Future Enhancements
    CNN-based deep learning  
    Cloud deployment  
    Mobile application version  
    Multi-device expansion  

    ### Conclusion
    AI TraceFinder provides a robust forensic solution for identifying the origin of scanned documents using artificial intelligence and advanced image processing techniques.
    """)
