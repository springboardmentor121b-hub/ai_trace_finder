import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
import os
import sys

# -------------------------------------------------
# PATH SETUP
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "src", "cnn_model"))

from model import SimpleCNN

# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------
IMG_SIZE = 128
MODEL_PATH = os.path.join(BASE_DIR, "models", "cnn", "cnn_model.pth")
DATA_DIR = os.path.join(BASE_DIR, "data", "Official")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load class names from dataset folders
CLASSES = sorted([
    d for d in os.listdir(DATA_DIR)
    if os.path.isdir(os.path.join(DATA_DIR, d))
])

# -------------------------------------------------
# TRANSFORMS (SAME AS TRAINING)
# -------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

# -------------------------------------------------
# LOAD MODEL (CACHED)
# -------------------------------------------------
@st.cache_resource
def load_model():
    model = SimpleCNN(num_classes=len(CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# -------------------------------------------------
# USER INTERFACE
# -------------------------------------------------
st.title("CNN Image Prediction")

st.write(
    "Upload an image to classify its source using the trained convolutional "
    "neural network model."
)

confidence_threshold = st.slider(
    "Minimum confidence threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01
)

uploaded_file = st.file_uploader(
    "Upload an image for classification",
    type=["jpg", "jpeg", "png", "tif", "tiff"]
)

# -------------------------------------------------
# PREDICTION
# -------------------------------------------------
if uploaded_file is not None:
    try:
        # Load image safely
        image = Image.open(uploaded_file)

        # Convert grayscale / TIFF to RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)

        # Preprocess
        img_tensor = transform(image).unsqueeze(0).to(DEVICE)

        # Model inference
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)

        st.subheader("Prediction Result")

        if confidence.item() >= confidence_threshold:
            st.success(f"Predicted Class: {CLASSES[predicted_class.item()]}")
            st.write(f"Confidence Score: {confidence.item():.4f}")
        else:
            st.warning(
                "Prediction confidence is below the selected threshold. "
                "Please try another image."
            )

    except Exception as e:
        st.error(
            "The uploaded file could not be processed. "
            "Please upload a valid image file."
        )
