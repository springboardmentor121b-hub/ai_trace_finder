import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV_PATH = os.path.join(PROJECT_ROOT, "processed_data", "combined_metadata.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "baseline")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

def evaluate_model(model_filename, name):
    model_path = os.path.join(MODEL_DIR, model_filename)
    scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")
    label_encoder_path = os.path.join(MODEL_DIR, "label_encoder.joblib")

    print(f"Loading data from {CSV_PATH}...")
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print(f"Error: {CSV_PATH} not found.")
        return

    # Features and Target
    drop_cols = ["file_name", "main_class", "resolution", "class_label"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    X = df[feature_cols]
    y = df["class_label"]

    # Load scaler + model + label encoder
    print(f"Loading resources from {MODEL_DIR}...")
    try:
        scaler = joblib.load(scaler_path)
        model = joblib.load(model_path)
        le = joblib.load(label_encoder_path)
    except FileNotFoundError as e:
        print(f"Error loading models: {e}")
        return

    # Transform features
    X_scaled = scaler.transform(X)
    
    # Encode target for evaluation
    y_encoded = le.transform(y)
    
    y_pred = model.predict(X_scaled)

    # Print report
    print(f"\n=== {name} Evaluation ===")
    print(classification_report(y_encoded, y_pred, target_names=le.classes_))

    # Confusion Matrix
    cm = confusion_matrix(y_encoded, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=le.classes_,
                yticklabels=le.classes_,
                cmap="Blues")
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    # Ensure results directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_path = os.path.join(RESULTS_DIR, f"{name.replace(' ', '_')}_confusion_matrix.png")

    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f" Confusion matrix saved to: {save_path}")

if __name__ == "__main__":
    evaluate_model("random_forest.joblib", "Random Forest")
    evaluate_model("svm.joblib", "SVM")
