import os
import torch
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from cnn_feature_extractor import CNNFeatureExtractor, extract_feature

DATA_ROOT = "Data_Sets"
CLASSES = ["official", "wikipedia", "Flatfield"]

X, y = [], []

cnn = CNNFeatureExtractor()

for label, cls in enumerate(CLASSES):
    for root, _, files in os.walk(os.path.join(DATA_ROOT, cls)):
        for f in files:
            if f.lower().endswith((".tif", ".jpg", ".png")):
                path = os.path.join(root, f)
                feat = extract_feature(cnn, path)
                X.append(feat)
                y.append(label)

X = np.array(X)
y = np.array(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_scaled, y)

os.makedirs("models", exist_ok=True)
torch.save(cnn.state_dict(), "models/cnn_feature_model.pth")
joblib.dump(clf, "models/hybrid_rf.pkl")
joblib.dump(scaler, "models/hybrid_scaler.pkl")

print("✅ CNN Hybrid training completed")
