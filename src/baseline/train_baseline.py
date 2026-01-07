import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

CSV_PATH = "processed_data\combined_features.csv"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

print("Loading CSV...")
df = pd.read_csv(CSV_PATH)

y = df["main_class"]

X = df.drop(columns=[
    "file_name",
    "dataset_source",
    "main_class",
    "class_label"
], errors="ignore")

X = X.select_dtypes(include=["int64", "float64"])

le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

acc = accuracy_score(y_test, model.predict(X_test))
print("Baseline Accuracy:", acc)

joblib.dump(model, "models/baseline_rf.pkl")
joblib.dump(scaler, "models/baseline_scaler.pkl")
joblib.dump(le, "models/baseline_encoder.pkl")

print("Baseline model saved")
