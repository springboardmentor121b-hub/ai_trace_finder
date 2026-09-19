# 🕵️ AI TraceFinder — Forensic Scanner Identification

AI TraceFinder is an end-to-end digital forensic platform designed to identify the **source scanner device** (brand and model) used to produce a scanned document or digital image. 

Every scanner leaves unique micro-level artifacts—including **metadata profiles, noise residuals, wavelet frequency signatures, and statistical intensity distributions**. By combining statistical feature extraction with Machine Learning (Random Forest and Support Vector Machines) and an interactive Streamlit forensic dashboard, AI TraceFinder enables rapid, non-destructive source identification and document authentication.

---

## 🌟 Key Features

- **Metadata Feature Extraction**: Computes 10 core statistical parameters (dimensions, aspect ratio, file size, mean intensity, standard deviation, skewness, kurtosis, entropy, and Sobel edge density).
- **Noise Residual & Wavelet Denoising**: Extracts subtle sensor/optics noise fingerprints via 2D wavelet decomposition (`skimage.restoration.denoise_wavelet`).
- **Machine Learning Classification**: Trained baseline models (Random Forest Classifier & SVM RBF) for robust multi-class scanner attribution.
- **Multi-Page Forensic Streamlit Dashboard**:
  - 📘 **System Flowchart**: Visual overview of the complete processing pipeline.
  - ⚙️ **Internal Mechanism**: In-depth look at wavelet noise residual & patch extraction.
  - 🧪 **Metadata Feature Extraction**: Interactive CSV preview and feature distribution charts.
  - 📊 **Model Results Summary**: Detailed evaluation metrics & confusion matrices.
  - 🔍 **Predict Scanner**: Real-time image upload & scanner prediction engine.
  - 📌 **Applications**: Real-world legal, enterprise, and forensic use cases.
  - 🎥 **About Project**: Project walkthrough & video introduction.
- **Cross-Platform & Modular Architecture**: Fully decoupled paths managed via `configs/config.py`.

---

## 🏗️ Architecture

```
+------------------+     +--------------------------+     +-----------------------------+
| Scanned Document | --> | Preprocessing & Denoise  | --> | Feature Vector Construction |
|  (TIF / PNG/ JPG)|     | (Grayscale, Resizing 512)|     | (10 Metadata Features)      |
+------------------+     +--------------------------+     +-----------------------------+
                                                                         |
                                                                         v
+------------------+     +--------------------------+     +-----------------------------+
| Predict Scanner  | <-- | Standard Scaler          | <-- | Trained ML Classifier       |
| & Class Prob.    |     | Normalization            |     | (Random Forest / SVM RBF)   |
+------------------+     +--------------------------+     +-----------------------------+
```

---

## 📂 Project Directory Structure

```
Forensic Scanner Identification/
├── configs/
│   ├── __init__.py
│   └── config.py                   # Centralized path configurations & settings
├── data/
│   ├── Official/                   # Raw official dataset directory (git-ignored)
│   ├── Wikipedia/                  # Raw Wikipedia dataset directory (git-ignored)
│   ├── processed/                  # Processed patches & feature extracts
│   └── metadata_features.csv       # Consolidated metadata dataset
├── doc/
│   ├── assets/
│   │   ├── flowchart.png           # Pipeline flowchart diagram
│   │   └── mechanism.png           # Internal mechanism architecture diagram
│   └── AI_TraceFinder.pdf          # Official project documentation PDF
├── experiments/
│   └── eda_official.py             # Exploratory Data Analysis script
├── models/
│   ├── random_forest.pkl           # Pre-trained Random Forest model
│   ├── svm.pkl                     # Pre-trained Support Vector Machine model
│   └── scaler.pkl                  # Fitted StandardScaler artifact
├── notebooks/
│   └── AI_TraceFinder_Master.ipynb # Master Colab / Jupyter notebook
├── pages/
│   ├── 1_Flowchart.py              # Streamlit page: System Flowchart
│   ├── 2_Mechanism.py              # Streamlit page: Internal Mechanism
│   ├── 3_Feature_Extraction.py     # Streamlit page: Feature Data Explorer
│   ├── 4_Model_Results.py          # Streamlit page: Model Performance
│   ├── 5_Predict_Scanner.py        # Streamlit page: Live Inference Engine
│   ├── 6_Applications.py           # Streamlit page: Forensic Use Cases
│   └── 7_About_Project.py          # Streamlit page: About & Video Demo
├── results/
│   ├── Random_Forest_confusion_matrix.png
│   └── eda_official/               # EDA visualizations output
├── src/
│   ├── __init__.py
│   └── baseline/
│       ├── __init__.py
│       ├── clean_metadata.py       # Metadata cleaning utility
│       ├── combine_csv.py          # Merges metadata CSVs
│       ├── combine_metadata.py     # Advanced CSV aggregation
│       ├── evaluate_baseline.py    # Generates classification reports & heatmaps
│       ├── predict_baseline.py     # Standalone CLI prediction module
│       ├── preprocess_official.py  # Preprocessing for Official dataset
│       ├── preprocess_wikipedia.py # Preprocessing for Wikipedia dataset
│       └── train_baseline.py       # Model training entry script
├── .gitattributes
├── .gitignore
├── README.md                       # Project documentation
├── landing_page.py                 # Streamlit main application entry point
└── requirements.txt                # Dependencies specification
```

---

## ⚙️ Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/springboardmentor121b-hub/ai_trace_finder.git
   cd ai_trace_finder
   ```

2. **Create and Activate a Virtual Environment**:
   ```bash
   # On Windows
   python -m venv .venv
   .venv\Scripts\activate

   # On macOS / Linux
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 How to Run the Application

Launch the Streamlit dashboard using `landing_page.py`:

```bash
streamlit run landing_page.py
```

Open your browser at `http://localhost:8501` to interact with the forensic dashboard.

---

## 📊 Dataset Information

AI TraceFinder is designed to handle multi-source scanner datasets:

1. **Official Dataset**: High-resolution flatbed scanner captures (e.g., Epson V39) collected across varied DPI resolutions (300/600 DPI).
2. **Wikipedia Scanned Document Dataset**: Scanned document images sourced from public archives.
3. **Features Extracted (`data/metadata_features.csv`)**:
   - `width`, `height`, `aspect_ratio`, `file_size_kb`
   - `mean_intensity`, `std_intensity`, `skewness`, `kurtosis`, `entropy`, `edge_density`

> *Note: Raw dataset folders (`data/Official/` and `data/Wikipedia/`) are excluded from Git to keep the repository lightweight. Place your raw image folders inside `data/Official/` or `data/Wikipedia/` before re-running preprocessing scripts.*

---

## 🤖 Model Information & Baseline Training

- **Random Forest Classifier**: 300 estimators, stratified splits. Fast, robust against outliers, handles non-linear feature spaces effectively.
- **Support Vector Machine (SVM)**: RBF kernel ($C=10$, gamma='scale', `probability=True`). Provides clear decision boundaries for fine-grained intensity variations.
- **Scaler**: `StandardScaler` fitted on training set metadata vectors.

To retrain baseline models locally:
```bash
python -m src.baseline.train_baseline
```

To evaluate models and update confusion matrices:
```bash
python -m src.baseline.evaluate_baseline
```

---

## 🔬 Exploratory Data Analysis & Experiments

To execute the EDA script on official dataset samples:
```bash
python -m experiments.eda_official
```
Output charts will be saved directly into `results/eda_official/`.

---

## 📈 Results

Evaluation results and confusion matrices are saved under `results/`.
Pre-trained model artifacts (`random_forest.pkl`, `svm.pkl`, `scaler.pkl`) are loaded automatically by the Streamlit prediction page for instant inference.

---

## 🛠️ Technologies Used

- **Python 3.10+**
- **Streamlit** (UI & Dashboard)
- **Scikit-Learn** & **Joblib** (Machine Learning & Serialization)
- **OpenCV** & **Scikit-Image** (Image Processing & Wavelet Denoising)
- **SciPy** (Statistical Analysis & Entropy Computations)
- **Pandas** & **NumPy** (Data Structures & Numerical Computation)
- **Matplotlib** & **Seaborn** (Visualizations)
- **PyTorch / Torchvision** (Dataset Loaders & Deep Learning Support)
