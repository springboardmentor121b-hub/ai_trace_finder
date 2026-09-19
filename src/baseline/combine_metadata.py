import pandas as pd
from pathlib import Path
import sys

# Ensure configs module can be imported
sys.path.append(str(Path(__file__).resolve().parents[2]))
try:
    from configs.config import PROCESSED_DATA_DIR, METADATA_CSV
    OFFICIAL_CLEANED = PROCESSED_DATA_DIR / "Official" / "official_metadata_cleaned.csv"
    WIKI_CLEANED = PROCESSED_DATA_DIR / "Wikipedia" / "wikipedia_metadata_cleaned.csv"
    COMBINED_OUT = METADATA_CSV
except ImportError:
    OFFICIAL_CLEANED = Path('data/processed/Official/official_metadata_cleaned.csv')
    WIKI_CLEANED = Path('data/processed/Wikipedia/wikipedia_metadata_cleaned.csv')
    COMBINED_OUT = Path('data/metadata_features.csv')

def combine_metadata(official_csv, wikipedia_csv, output_csv):
    dfs = []
    if Path(official_csv).exists():
        dfs.append(pd.read_csv(official_csv))
    if Path(wikipedia_csv).exists():
        dfs.append(pd.read_csv(wikipedia_csv))

    if dfs:
        df_combined = pd.concat(dfs, ignore_index=True)
        df_combined = df_combined.drop_duplicates().dropna(how='all')
        df_combined.to_csv(output_csv, index=False)
        print("Metadata combining done! Output:", output_csv)
    else:
        print("No cleaned metadata CSV files found to combine.")

if __name__ == "__main__":
    combine_metadata(OFFICIAL_CLEANED, WIKI_CLEANED, COMBINED_OUT)