import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix


# ===================== PATHS =====================

CSV_PATH = r"D:\Project\TraceFinder\processed_data\combined_features.csv"


MODEL_PATH = r"D:\Project\TraceFinder\models\svm_random\random_forest.pkl"
SCALER_PATH = r"D:\Project\TraceFinder\models\svm_random\scaler.pkl"
LE_PATH = r"D:\Project\TraceFinder\models\svm_random\label_encoder.pkl"

RESULTS_DIR = r"D:\Project\TraceFinder\results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ===================== EVALUATION =====================

def evaluate_model():
    df = pd.read_csv(CSV_PATH)

    # Separate features and labels
    X = df.drop(columns=["file_name", "main_class", "resolution", "class_label"])
    y_true = df["class_label"]

    # Load model components
    scaler = joblib.load(SCALER_PATH)
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(LE_PATH)

    # 🔥 Enforce SAME feature order as training
    X = X[scaler.feature_names_in_]

    # Scale & predict
    X_scaled = scaler.transform(X)
    y_pred_enc = model.predict(X_scaled)

    # Decode labels
    y_pred = label_encoder.inverse_transform(y_pred_enc)

    # ===================== REPORT =====================

    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred))

    # Confusion matrix
    labels = label_encoder.classes_
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels
    )

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Random Forest Confusion Matrix")

    save_path = os.path.join(RESULTS_DIR, "confusion_matrix_rf.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print("Confusion matrix saved to:", save_path)


# ===================== MAIN =====================

if __name__ == "__main__":
    evaluate_model()
