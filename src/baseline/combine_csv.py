import pandas as pd
from pathlib import Path
import sys

# Ensure configs module can be imported
sys.path.append(str(Path(__file__).resolve().parents[2]))
try:
    from configs.config import PROCESSED_DATA_DIR, METADATA_CSV
    OFFICIAL_CSV = PROCESSED_DATA_DIR / "Official" / "metadata_features.csv"
    WIKI_CSV = PROCESSED_DATA_DIR / "Wikipedia" / "metadata_features.csv"
    OUT_CSV = METADATA_CSV
except ImportError:
    OFFICIAL_CSV = Path("data/processed/Official/metadata_features.csv")
    WIKI_CSV = Path("data/processed/Wikipedia/metadata_features.csv")
    OUT_CSV = Path("data/metadata_features.csv")

def combine():
    dfs = []
    if OFFICIAL_CSV.exists():
        dfs.append(pd.read_csv(OFFICIAL_CSV))
    if WIKI_CSV.exists():
        dfs.append(pd.read_csv(WIKI_CSV))

    if dfs:
        df = pd.concat(dfs, ignore_index=True)
        df.to_csv(OUT_CSV, index=False)
        print(f" Combined CSV saved to {OUT_CSV}")
    else:
        print("No input CSV files found to combine.")

if __name__ == "__main__":
    combine()
