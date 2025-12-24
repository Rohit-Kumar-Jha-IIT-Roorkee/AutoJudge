import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Load data
df = pd.read_csv("cf_text_rating.csv")

X = df["full_text"]
y = df["rating"]

# TF-IDF
tfidf = TfidfVectorizer(
    max_features=30000,
    ngram_range=(1, 2),
    stop_words="english"
)

X_vec = tfidf.fit_transform(X)

# Train-test split (just for sanity)
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

# Model
reg = Ridge(alpha=1.0)
reg.fit(X_train, y_train)

# Sanity check
preds = reg.predict(X_test)
print("MAE (rating):", mean_absolute_error(y_test, preds))

# Save
joblib.dump(tfidf, "tfidf_cf.pkl")
joblib.dump(reg, "cf_rating_model.pkl")

print("Model trained & saved.")
