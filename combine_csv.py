import pandas as pd

df1 = pd.read_csv("processed_data/Official/metadata_features.csv")
df2 = pd.read_csv("processed_data/Wikipedia/metadata_features.csv")

df = pd.concat([df1, df2], ignore_index=True)

df.to_csv("processed_data/metadata_features.csv", index=False)

print(" Combined CSV saved to processed_data/metadata_features.csv")
