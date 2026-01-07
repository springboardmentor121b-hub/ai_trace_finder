import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

# ================= PATHS =================
CSV_PATH = "processed_data/combined_features.csv"
MODEL_DIR = "models"
RESULT_DIR = "results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# ================= LOAD DATA =================
print("Loading CSV...")
df = pd.read_csv(CSV_PATH)

print("Columns:", list(df.columns))
print("Total samples:", len(df))

# ================= TARGET =================
y = df["main_class"]

# ================= FEATURE SELECTION =================
# Drop all non-numeric / ID columns
DROP_COLS = [
    "file_name",
    "dataset_source",
    "main_class"
]

X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

# Keep only numeric columns (SAFETY)
X = X.select_dtypes(include=[np.number])

print("Numeric feature columns:", list(X.columns))

# ================= CLEAN DATA =================
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.mean())

# ================= ENCODE LABEL =================
le = LabelEncoder()
y_enc = le.fit_transform(y)

# ================= SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

# ================= SCALE =================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ================= TRAIN MODEL =================
print("Training Random Forest...")
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# ================= EVALUATE =================
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("Model Accuracy:", round(acc, 4))

# ================= SAVE =================
joblib.dump(model, f"{MODEL_DIR}/baseline_rf.pkl")
joblib.dump(scaler, f"{MODEL_DIR}/baseline_scaler.pkl")
joblib.dump(le, f"{MODEL_DIR}/baseline_encoder.pkl")

print("Models saved successfully")
