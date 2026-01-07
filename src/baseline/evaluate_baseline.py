import pandas as pd
import joblib
from sklearn.metrics import classification_report

df = pd.read_csv("processed_data\combined_features.csv")

X = df.select_dtypes(include=["int64", "float64"])
y = df["main_class"]

model = joblib.load("models/baseline_rf.pkl")
scaler = joblib.load("models/baseline_scaler.pkl")
encoder = joblib.load("models/baseline_encoder.pkl")

X_scaled = scaler.transform(X)
y_pred = encoder.inverse_transform(model.predict(X_scaled))

print(classification_report(y, y_pred))
