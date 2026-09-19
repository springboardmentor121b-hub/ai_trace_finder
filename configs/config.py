"""
AI TraceFinder Configuration Module
Centralized path definitions and global project settings.
"""

import os
from pathlib import Path

# Project root directory (parent directory of configs/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Key Directories
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
DOC_DIR = PROJECT_ROOT / "doc"
ASSETS_DIR = DOC_DIR / "assets"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
PAGES_DIR = PROJECT_ROOT / "pages"
RESULTS_DIR = PROJECT_ROOT / "results"
SRC_DIR = PROJECT_ROOT / "src"

# Data Paths
RAW_OFFICIAL_DIR = DATA_DIR / "Official"
RAW_WIKI_DIR = DATA_DIR / "Wikipedia"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
METADATA_CSV = DATA_DIR / "metadata_features.csv"

# Model Paths
RF_MODEL_PATH = MODELS_DIR / "random_forest.pkl"
SVM_MODEL_PATH = MODELS_DIR / "svm.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"

# Documentation / Asset Paths
FLOWCHART_PATH = ASSETS_DIR / "flowchart.png"
MECHANISM_PATH = ASSETS_DIR / "mechanism.png"
PDF_DOC_PATH = DOC_DIR / "AI_TraceFinder.pdf"

# Ensure output directories exist when imported
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
