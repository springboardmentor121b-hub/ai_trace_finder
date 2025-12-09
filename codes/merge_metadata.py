import os, pandas as pd, sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
processed_dir = os.path.join(project_root, "processed_data")
official_path = os.path.join(processed_dir, "Official", "metadata_features.csv")
wiki_path = os.path.join(processed_dir, "WikiPedia", "metadata_features_wikipedia.csv")
out_path = os.path.join(processed_dir, "merged_metadata.csv")

if not os.path.isfile(official_path):
    print("Official file missing:", official_path); sys.exit(1)
if not os.path.isfile(wiki_path):
    print("Wiki file missing:", wiki_path); sys.exit(1)

df_official = pd.read_csv(official_path)
df_wiki = pd.read_csv(wiki_path)

# -------- ADD THIS PART (MERGE + SAVE) --------

df_official["dataset"] = "Official"
df_wiki["dataset"] = "Wikipedia"

df = pd.concat([df_official, df_wiki], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df.to_csv(out_path, index=False)

print("\nMerged file created successfully!")
print("Saved at:", out_path)
print("Total rows:", len(df))
