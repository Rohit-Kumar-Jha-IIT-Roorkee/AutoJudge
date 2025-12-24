from pathlib import Path
from flask import Flask, render_template, request, jsonify
import joblib

# =====================
# Resolve PROJECT ROOT
# =====================
ROOT_DIR = Path(__file__).resolve().parents[1]  # AutoJudge_Final

# =====================
# Classification (TEXT → Easy/Medium/Hard)
# =====================
clf_pipeline = joblib.load(
    ROOT_DIR / "artifacts" / "feature_pipeline_classification.pkl"
)

clf_model = joblib.load(
    ROOT_DIR / "artifacts" / "logistic_regression_baseline.pkl"
)

# =====================
# Regression (CODEFORCES RATING MODEL)
# =====================
REG_DIR = ROOT_DIR / "src" / "models" / "regression"

reg_pipeline = joblib.load(
    REG_DIR / "tfidf_cf.pkl"
)

reg_model = joblib.load(
    REG_DIR / "cf_rating_model.pkl"
)

# =====================
# Flask App
# =====================
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        desc = data.get("problem_desc", "")
        inp = data.get("input_desc", "")
        out = data.get("output_desc", "")

        full_text = f"{desc} {inp} {out}"

        # ---------- Regression (Codeforces rating) ----------
        X_reg = reg_pipeline.transform([full_text])
        raw_rating = float(reg_model.predict(X_reg)[0])

        # Round to nearest 100 (Codeforces-style)
        rating = int(round(raw_rating / 100) * 100)

        # Safety bounds (optional but recommended)
        rating = max(800, min(rating, 3500))

        # ---------- Classification (Easy / Medium / Hard) ----------
        X_clf = clf_pipeline.transform([full_text])
        difficulty_class = clf_model.predict(X_clf)[0]

        return jsonify({
            "difficulty_class": difficulty_class,
            "difficulty_score": rating
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
