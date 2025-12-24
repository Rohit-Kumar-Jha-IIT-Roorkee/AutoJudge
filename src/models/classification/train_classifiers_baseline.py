from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

import joblib

# =====================
# Resolve PROJECT ROOT
# =====================
ROOT_DIR = Path(__file__).resolve().parents[3]

DATA_PATH = ROOT_DIR / "data" / "processed" / "autojudge_dataset_cleaned.csv"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

# =====================
# Load CLEANED dataset
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
# Encode labels (ONLY for XGBoost)
# =====================
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# =====================
# Train-test split
# =====================
X_train, X_test, y_train, y_test, y_train_enc, y_test_enc = train_test_split(
    X,
    y,
    y_encoded,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# =====================
# Feature pipeline
# =====================
feature_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=6000,
        ngram_range=(1, 2),
        stop_words="english"
    )),
    ("svd", TruncatedSVD(n_components=200, random_state=42))
])

X_train_vec = feature_pipeline.fit_transform(X_train)
X_test_vec = feature_pipeline.transform(X_test)

joblib.dump(
    feature_pipeline,
    ARTIFACTS_DIR / "feature_pipeline_classification.pkl"
)

# =====================
# Models
# =====================
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        n_jobs=-1
    ),
    "Linear SVM": LinearSVC(
        class_weight="balanced"
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1
    )
}

# =====================
# Train & Evaluate
# =====================
for name, model in models.items():
    print(f"\n================ {name} ================")

    if name == "XGBoost":
        model.fit(X_train_vec, y_train_enc)
        y_pred_enc = model.predict(X_test_vec)
        y_pred = label_encoder.inverse_transform(y_pred_enc)
    else:
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    model_path = ARTIFACTS_DIR / f"{name.replace(' ', '_').lower()}_baseline.pkl"
    joblib.dump(model, model_path)

    print(f"Saved model to {model_path}")
