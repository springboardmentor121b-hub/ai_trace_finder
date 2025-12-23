
# import pandas as pd
# import joblib
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC

# CSV_PATH = "../processed_data/Official/metadata_features.csv"

# def train_models():
#     df = pd.read_csv(CSV_PATH)
#     X = df.drop(columns=["file_name", "main_class", "resolution", "class_label"])
#     y = df["class_label"]

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, stratify=y, random_state=42
#     )

#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)

#     # Train Random Forest
#     rf = RandomForestClassifier(n_estimators=300, random_state=42)
#     rf.fit(X_train, y_train)
#     joblib.dump(rf, "models/random_forest.pkl")

#     # Train SVM
#     svm = SVC(kernel="rbf", C=10, gamma="scale", probability=True)
#     svm.fit(X_train, y_train)
#     joblib.dump(svm, "models/svm.pkl")

    
#     joblib.dump(scaler, "models/scaler.pkl")

#     print(" Models trained and saved successfully!")

# if __name__ == "__main__":
#     train_models()


import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

CSV_PATH = "../processed_data/Wikipedia/metadata_features_wikipedia.csv"

def train_models():
    # Create models folder automatically
    os.makedirs("models_wikipedia", exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    X = df.drop(columns=["file_name", "main_class", "resolution", "class_label"])
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
    joblib.dump(rf, "models_wikipedia/random_forest.pkl")

    # Train SVM
    svm = SVC(kernel="rbf", C=10, gamma="scale", probability=True)
    svm.fit(X_train, y_train)
    joblib.dump(svm, "models_wikipedia/svm.pkl")

    # Save scaler
    joblib.dump(scaler, "models_wikipedia/scaler.pkl")

    print(" Models trained and saved successfully!")

if __name__ == "__main__":
    train_models()
