import torch
import joblib
from cnn_feature_extractor import CNNFeatureExtractor, extract_feature

CLASSES = ["official", "wikipedia", "Flatfield"]

img_path = input("Enter image path: ").strip()

cnn = CNNFeatureExtractor()
cnn.load_state_dict(torch.load("models/cnn_feature_model.pth"))
cnn.eval()

clf = joblib.load("models/hybrid_rf.pkl")
scaler = joblib.load("models/hybrid_scaler.pkl")

feat = extract_feature(cnn, img_path)
feat = scaler.transform([feat])

pred = clf.predict(feat)[0]
print("✅ Predicted class:", CLASSES[pred])