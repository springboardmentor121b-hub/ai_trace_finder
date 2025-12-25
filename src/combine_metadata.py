import os
import pandas as pd


def combine_metadata(official_csv, wikipedia_csv, output_csv):
    # Read CSVs (paths may be absolute or relative)
    df_official = pd.read_csv(official_csv)
    df_wiki = pd.read_csv(wikipedia_csv)

    # Concatenate and clean
    df_combined = pd.concat([df_official, df_wiki], ignore_index=True)
    df_combined = df_combined.drop_duplicates()
    df_combined = df_combined.dropna(how="all")

    # Ensure output directory exists
    out_dir = os.path.dirname(output_csv)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    df_combined.to_csv(output_csv, index=False)
    print("Metadata combining done! Output:", output_csv)


if __name__ == "__main__":
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    official_path = os.path.join(base, "processed_data", "Official", "official_metadata_cleaned.csv")
    wiki_path = os.path.join(base, "processed_data", "Wikipedia", "wikipedia_metadata_cleaned.csv")
    out_path = os.path.join(base, "processed_data", "combined_metadata_features.csv")

    # Fallback if files are in lowercase folders
    if not os.path.exists(official_path):
        alt = os.path.join(base, "processed_data", "official", "official_metadata_cleaned.csv")
        if os.path.exists(alt):
            official_path = alt

    if not os.path.exists(wiki_path):
        altw = os.path.join(base, "processed_data", "wikipedia", "wikipedia_metadata_cleaned.csv")
        if os.path.exists(altw):
            wiki_path = altw

    print("Using paths:\n -", official_path, "\n -", wiki_path, "\n ->", out_path)
    combine_metadata(official_path, wiki_path, out_path)