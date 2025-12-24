from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

import joblib

# =====================
# Resolve PROJECT ROOT
# =====================
ROOT_DIR = Path(__file__).resolve().parents[3]

DATA_PATH = ROOT_DIR / "data" / "processed" / "autojudge_dataset_cleaned.csv"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

# =====================
# Load dataset
# =====================
df = pd.read_csv(DATA_PATH)

df["full_text"] = (
    df["description"] + " " +
    df["input_description"] + " " +
    df["output_description"]
)

X = df["full_text"]
y = df["difficulty_class"]

print("Dataset shape:", df.shape)
print("\nClass distribution:")
print(y.value_counts())

# =====================
# Train-test split
# =====================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# =====================
# Pipeline with SMOTE
# =====================
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=6000,
        ngram_range=(1, 2),
        stop_words="english"
    )),
    ("svd", TruncatedSVD(n_components=200, random_state=42)),
    ("smote", SMOTE(random_state=42)),
    ("clf", LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        n_jobs=-1
    ))
])

# =====================
# Train
# =====================
pipeline.fit(X_train, y_train)

# =====================
# Evaluate
# =====================
y_pred = pipeline.predict(X_test)

print("\n=== Logistic Regression + SMOTE ===\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# =====================
# Save model
# =====================
joblib.dump(
    pipeline,
    ARTIFACTS_DIR / "logistic_regression_smote.pkl"
)

print("\nSaved SMOTE-enhanced Logistic Regression model.")
