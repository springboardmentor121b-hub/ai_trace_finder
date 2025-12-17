import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

CSV_PATH = r"D:\Project\TraceFinder\processed_data\combined_features.csv"
MODEL_DIR = r"D:\Project\TraceFinder\models\svm_random"

def train_models():
    df = pd.read_csv(CSV_PATH)

    X = df.drop(columns=[
        "file_name", "dataset_source", "main_class", "resolution", "class_label"
    ])
    y = df["class_label"]

    le = LabelEncoder()
    y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    svm = SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        probability=True
    )
    svm.fit(X_train, y_train)

    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(rf, os.path.join(MODEL_DIR, "random_forest.pkl"))
    joblib.dump(svm, os.path.join(MODEL_DIR, "svm.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.pkl"))

    rf_pred = rf.predict(X_test)
    svm_pred = svm.predict(X_test)

    print("\nRandom Forest Accuracy:", accuracy_score(y_test, rf_pred))
    print(classification_report(y_test, rf_pred))

    print("\nSVM Accuracy:", accuracy_score(y_test, svm_pred))
    print(classification_report(y_test, svm_pred))

    print("\nModels trained and saved successfully")

if __name__ == "__main__":
    train_models()
