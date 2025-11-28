import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Correct CSV PATH (relative from src/baseline -> up two -> processed_data)
CSV_PATH = "../../processed_data/metadata_features_for_training.csv"

# Correct MODEL directory (will create inside ai_trace_finder/models)
MODEL_DIR = "../../models"
os.makedirs(MODEL_DIR, exist_ok=True)

def train_models():
    df = pd.read_csv(CSV_PATH)

    # Drop non-feature column
    X = df.drop(columns=["class_label"])
    y = df["class_label"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=300, random_state=42)
    rf.fit(X_train, y_train)
    joblib.dump(rf, f"{MODEL_DIR}/random_forest.pkl")

    # SVM
    svm = SVC(kernel="rbf", C=10, gamma="scale", probability=True)
    svm.fit(X_train, y_train)
    joblib.dump(svm, f"{MODEL_DIR}/svm.pkl")

    # Save scaler
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")

    print("Models trained and saved successfully!")

if __name__ == "__main__":
    train_models()
