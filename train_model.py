import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

CSV_PATH = "processed_data\combined_features.csv"

MODEL_DIR = "models"
RESULT_DIR = "results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

print("Loading CSV...")
df = pd.read_csv(CSV_PATH)

print("Columns:", list(df.columns))
print("Total samples:", len(df))

# -------- Target --------
y = df["main_class"]

# -------- Features (numeric only) --------
X = df.drop(columns=[
    "file_name",
    "dataset_source",
    "main_class",
    "class_label"
], errors="ignore")

X = X.select_dtypes(include=["int64", "float64"])

print("Numeric feature columns:", list(X.columns))

# -------- Encode labels --------
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# -------- Train-test split --------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# -------- Scale --------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------- Model --------
print("Training model...")
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_scaled, y_train)

# -------- Evaluation --------
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print("Model Accuracy:", acc)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=label_encoder.classes_)
disp.plot()
plt.savefig(os.path.join(RESULT_DIR, "Random_Forest_confusion_matrix.png"))
plt.close()

# -------- Save artifacts --------
joblib.dump(model, os.path.join(MODEL_DIR, "random_forest.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(label_encoder, os.path.join(MODEL_DIR, "label_encoder.pkl"))

print("Model training completed successfully")
print("Saved files:")
print("- models/random_forest.pkl")
print("- models/scaler.pkl")
print("- models/label_encoder.pkl")
print("- results/Random_Forest_confusion_matrix.png")
