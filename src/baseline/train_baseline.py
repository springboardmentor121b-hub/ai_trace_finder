import pandas as pd
import joblib
from pathlib import Path
import sys
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Ensure configs module can be imported
sys.path.append(str(Path(__file__).resolve().parents[2]))
try:
    from configs.config import METADATA_CSV, RF_MODEL_PATH, SVM_MODEL_PATH, SCALER_PATH, MODELS_DIR
    CSV_PATH = str(METADATA_CSV)
except ImportError:
    CSV_PATH = "data/metadata_features.csv"
    MODELS_DIR = Path("models")
    RF_MODEL_PATH = MODELS_DIR / "random_forest.pkl"
    SVM_MODEL_PATH = MODELS_DIR / "svm.pkl"
    SCALER_PATH = MODELS_DIR / "scaler.pkl"

os.makedirs(os.path.dirname(RF_MODEL_PATH), exist_ok=True)

def train_models():
    df = pd.read_csv(CSV_PATH)
    
    # Drop non-feature columns
    feature_cols = [c for c in df.columns if c not in ["file_name", "main_class", "resolution", "class_label"]]
    X = df[feature_cols]
    y = df["class_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=300, random_state=42)
    rf.fit(X_train, y_train)
    joblib.dump(rf, RF_MODEL_PATH)

    # Train SVM
    svm = SVC(kernel="rbf", C=10, gamma="scale", probability=True)
    svm.fit(X_train, y_train)
    joblib.dump(svm, SVM_MODEL_PATH)

    joblib.dump(scaler, SCALER_PATH)

    print(" Models trained and saved successfully!")

if __name__ == "__main__":
    train_models()
