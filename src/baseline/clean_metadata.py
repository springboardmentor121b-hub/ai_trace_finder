import pandas as pd
from pathlib import Path
import sys

# Ensure configs module can be imported
sys.path.append(str(Path(__file__).resolve().parents[2]))
try:
    from configs.config import PROCESSED_DATA_DIR
    OFFICIAL_IN = PROCESSED_DATA_DIR / "Official" / "metadata_features.csv"
    OFFICIAL_OUT = PROCESSED_DATA_DIR / "Official" / "official_metadata_cleaned.csv"
    WIKI_IN = PROCESSED_DATA_DIR / "Wikipedia" / "metadata_features.csv"
    WIKI_OUT = PROCESSED_DATA_DIR / "Wikipedia" / "wikipedia_metadata_cleaned.csv"
except ImportError:
    OFFICIAL_IN = Path('data/processed/Official/metadata_features.csv')
    OFFICIAL_OUT = Path('data/processed/Official/official_metadata_cleaned.csv')
    WIKI_IN = Path('data/processed/Wikipedia/metadata_features.csv')
    WIKI_OUT = Path('data/processed/Wikipedia/wikipedia_metadata_cleaned.csv')

def clean_metadata(input_path, output_path):
    print("-" * 60)
    print("Cleaning:", input_path)
    try:
        data = pd.read_csv(input_path)
        print("Original data shape:", data.shape)
        data = data.drop_duplicates()
        data = data.dropna(how='all')
        print("Cleaned data shape:", data.shape)
        data.to_csv(output_path, index=False)
        print("File saved successfully at:", output_path)
    except Exception as e:
        print("ERROR processing", input_path)
        print(str(e))

if __name__ == "__main__":
    if OFFICIAL_IN.exists():
        clean_metadata(OFFICIAL_IN, OFFICIAL_OUT)
    if WIKI_IN.exists():
        clean_metadata(WIKI_IN, WIKI_OUT)
