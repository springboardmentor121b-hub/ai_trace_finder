import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

CSV_PATH = r"D:\project\Tracefinder\processed_data\metadata_features.csv"
MODEL_PATH = r"D:\project\Tracefinder\models\random_forest.pkl"
SCALER_PATH = r"D:\project\Tracefinder\models\scaler.pkl"
RESULTS_DIR = r"D:\project\Tracefinder\results"

os.makedirs(RESULTS_DIR, exist_ok=True)

def evaluate_model():
    df = pd.read_csv(CSV_PATH)

    X = df.drop(columns=["file_name", "main_class", "resolution", "class_label"])
    y = df["class_label"]

    scaler = joblib.load(SCALER_PATH)
    model = joblib.load(MODEL_PATH)

    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)

    print(classification_report(y, y_pred))

    cm = confusion_matrix(y, y_pred, labels=model.classes_)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=model.classes_,
                yticklabels=model.classes_,
                cmap="Blues")

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Random Forest Confusion Matrix")

    save_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    plt.savefig(save_path, dpi=300)
    plt.show()

    print("Saved to:", save_path)

if __name__ == "__main__":
    evaluate_model()
