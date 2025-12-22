import os
import pandas as pd
df = pd.read_csv("processed_data/combined_features.csv")
print("Shape:", df.shape)


OFFICIAL_CLEANED = "processed_data/official/official_metadata_cleaned.csv"
WIKI_CLEANED     = "processed_data/wikipedia/wikipedia_metadata_cleaned.csv"
OUTPUT_COMBINED  = "processed_data/combined_features.csv"


def combine_metadata():
    print("-" * 60)
    print("Reading cleaned official and wikipedia metadata...")

    df_official = pd.read_csv(OFFICIAL_CLEANED)
    df_wiki     = pd.read_csv(WIKI_CLEANED)

    print("Official shape:", df_official.shape)
    print("Wikipedia shape:", df_wiki.shape)

    # Columns align (safety)
    common_cols = [c for c in df_official.columns if c in df_wiki.columns]
    df_official = df_official[common_cols]
    df_wiki     = df_wiki[common_cols]

    # Add dataset_source column
    df_official["dataset_source"] = "official"
    df_wiki["dataset_source"]     = "wikipedia"

    combined = pd.concat([df_official, df_wiki], ignore_index=True)

    print("Combined shape:", combined.shape)
    os.makedirs(os.path.dirname(OUTPUT_COMBINED), exist_ok=True)
    combined.to_csv(OUTPUT_COMBINED, index=False)
    print("Saved global combined features to:", OUTPUT_COMBINED)


if __name__ == "__main__":
    combine_metadata()
