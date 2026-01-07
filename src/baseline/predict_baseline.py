import joblib
import numpy as np

model = joblib.load("models/baseline_rf.pkl")
scaler = joblib.load("models/baseline_scaler.pkl")
encoder = joblib.load("models/baseline_encoder.pkl")

print("Enter features in same order as CSV:")
vals = list(map(float, input().split()))

X = scaler.transform([vals])
pred = model.predict(X)

print("Prediction:", encoder.inverse_transform(pred)[0])