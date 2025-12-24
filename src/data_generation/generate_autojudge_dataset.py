from datasets import load_dataset
import pandas as pd

# Load Codeforces dataset
dataset = load_dataset("open-r1/codeforces", split="train")

records = []

for item in dataset:
    desc = item.get("description") or ""
    inp = item.get("input_format") or ""
    out = item.get("output_format") or ""
    rating = item.get("rating")

    if desc.strip() and rating is not None:
        # Assign difficulty class
        if rating <= 1200:
            diff_class = "Easy"
        elif rating < 1800:
            diff_class = "Medium"
        else:
            diff_class = "Hard"

        records.append({
            "description": desc.strip(),
            "input_description": inp.strip(),
            "output_description": out.strip(),
            "difficulty_class": diff_class,
            "difficulty_score": rating
        })

df = pd.DataFrame(records)

print("Final dataset shape:", df.shape)
print(df["difficulty_class"].value_counts())
print(df.head())

# Save final dataset
df.to_csv(
    "../../data/processed/autojudge_dataset.csv",
    index=False
)

print("Saved autojudge_dataset.csv")
