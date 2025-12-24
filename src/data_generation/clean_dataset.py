import pandas as pd

# Load dataset
df = pd.read_csv("../../data/processed/autojudge_dataset.csv")

print("Original dataset shape:", df.shape)

# Drop rows with missing or empty input/output descriptions
df_cleaned = df.dropna(
    subset=["input_description", "output_description"]
)

# Also drop rows where input/output is empty string
df_cleaned = df_cleaned[
    (df_cleaned["input_description"].str.strip() != "") &
    (df_cleaned["output_description"].str.strip() != "")
]

print("Cleaned dataset shape:", df_cleaned.shape)
print("Rows removed:", df.shape[0] - df_cleaned.shape[0])

# Save cleaned dataset
df_cleaned.to_csv(
    "../../data/processed/autojudge_dataset_cleaned.csv",
    index=False
)

print("Saved autojudge_dataset_cleaned.csv")
