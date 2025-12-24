# AutoJudge – Predicting Programming Problem Difficulty

## 1. Project Overview

AutoJudge is a machine learning system that predicts the difficulty of programming problems using only textual information.

The system produces:

- A difficulty class: Attachment  Easy / Medium / Hard  
- A numerical difficulty score aligned with Codeforces-style ratings (e.g., 800–3000)

The project strictly follows the problem statement by implementing both classification and regression models, along with proper evaluation and a working web interface.

All models use **classical machine learning** and **classical text feature extraction techniques**.  
**No deep learning models are used anywhere in the pipeline.**

---

## 2. Problem Statement Requirements

The problem statement required:

- Difficulty classification  
- Difficulty score prediction  
- Proper model evaluation  
- A web interface for predictions  

✅ **All requirements have been fully implemented using classical ML methods.**

---

## 3. Datasets Used

### 3.1 Initial Provided Dataset

The initially provided dataset contained programming problem descriptions with limited difficulty-related information.

**Issues observed during experimentation:**

- No explicit difficulty class labels  
- Difficulty scores clustered in a narrow range  
- Regression models collapsed to near-constant predictions  
- Classification performance was unstable due to weak label signals  

Despite extensive experimentation, this dataset lacked sufficient variance for reliable difficulty prediction.

➡️ These experiments are retained for transparency and comparison.

---

### 3.2 Improved Dataset (Codeforces-Based)

To address the limitations above, a larger and more diverse dataset was constructed using Codeforces problem data.

Each problem includes:

- Problem description  
- Input format  
- Output format  
- Official Codeforces difficulty rating  

Rows with missing input or output descriptions were removed to ensure data quality.

---

## 4. Difficulty Class Generation

Difficulty classes were generated only during dataset construction using Codeforces ratings:

| Rating Range | Difficulty Class |
|-------------|------------------|
| ≤ 1200      | Easy             |
| 1200–1799   | Medium           |
| ≥ 1800      | Hard             |

⚠️ During classification training, the rating column is **not used**, preventing any label leakage.

---

## 5. Exploratory Data Analysis (EDA)

**Key observations from EDA:**

- Strong class imbalance (Hard problems dominate)  
- Significant variation in text length and vocabulary  
- Difficulty correlates with textual and structural complexity  

These observations motivated the use of class-weighted models.

📊 A detailed EDA with visualizations is available in:  
`notebooks/eda.ipynb`

---

## 6. Feature Engineering

The project uses **classical text feature extraction techniques**, including:

- TF-IDF vectorization  
- Unigrams and bigrams (n-grams)  
- Dimensionality reduction using Truncated SVD  

No pretrained embeddings, neural networks, or transformers are used.

---

## 7. Classification Models

### 7.1 Models Evaluated

- Logistic Regression (class-weighted)  
- Linear SVM (class-weighted)  
- XGBoost Classifier  

---

### 7.2 Handling Class Imbalance

The dataset shows significant class imbalance, especially for Hard problems.

**Strategies explored:**

- Class weighting in linear models  
- SMOTE-based oversampling during experimentation  

Although SMOTE was evaluated, it did not provide consistent improvements.  
The final system therefore relies on **class-weighted learning**, not synthetic oversampling.

---

### 7.3 Evaluation Metrics

- Accuracy  
- Precision, Recall, F1-score  
- Confusion Matrix  

---

### 7.4 Final Classification Model

**Logistic Regression (class-weighted)** was selected due to:

- Stable performance across all classes  
- Better balance between precision and recall  
- Lower overfitting risk compared to tree-based models  

---

## 8. Regression Models (Difficulty Score Prediction)

### 8.1 Objective

Predict a numerical difficulty score aligned with Codeforces ratings using text-only information.

---

### 8.2 Models Evaluated

- Linear Regression  
- Gradient Boosting Regressor  
- XGBoost Regressor  
- Random Forest Regressor (with hyperparameter tuning)  

---

### 8.3 Evaluation Metrics

- MAE (Mean Absolute Error)  
- MSE (Mean Squared Error)  
- RMSE (Root Mean Squared Error)  

---

### 8.4 Final Regression Model

The final regression model achieves:

- **MAE ≈ 470 rating points**

Given the subjective and noisy nature of difficulty ratings, this performance is considered reasonable for text-based prediction.

---

## 9. Final System Architecture

The system uses two independent models:

- **Regression model** → predicts numerical difficulty rating  
- **Classification model** → predicts Easy / Medium / Hard directly from text  

Both models operate independently and are combined at inference time.

---

## 10. Web Application

- Flask backend  
- HTML/CSS frontend inspired by Codeforces UI  
- Real-time predictions  

**User inputs:**

- Problem description  
- Input format  
- Output format  

---

## 11. Results Summary

| Task           | Model                     | Key Metric        |
|---------------|---------------------------|-------------------|
| Classification| Logistic Regression        | ~60% accuracy    |
| Regression    | Codeforces Rating Model    | MAE ≈ 470        |

---

## 12. Project Structure

AutoJudge_Final/
│
├── app/
│ ├── app.py
│ ├── templates/
│ │ └── index.html
│ └── static/
│
├── artifacts/
│ ├── feature_pipeline_classification.pkl
│ ├── logistic_regression_baseline.pkl
│ ├── tfidf_cf.pkl
│ └── cf_rating_model.pkl
│
├── data/
│ ├── raw/
│ └── processed/
│
├── src/
│ ├── data_generation/
│ └── models/
│ ├── classification/
│ └── regression/
│
├── notebooks/
│ └── eda.ipynb
│
├── README.md
└── requirements.txt


---

## 13. Limitations & Future Work

**Limitations:**

- Class imbalance remains a challenge  
- Difficulty prediction is inherently subjective  

**Possible improvements:**

- Incorporating symbolic features (constraints, limits)  
- Larger curated datasets  
- Multi-task learning approaches  

---

## 14. Conclusion

AutoJudge successfully implements both classification and regression for programming problem difficulty prediction using classical machine learning techniques.

The system is fully functional, empirically evaluated, and strictly adheres to the given constraints.

---

## 15. How to Run the Project

### 15.1 Prerequisites

- Python 3.9 or higher
- pip (Python package manager)

---

### 15.2 Setup

Clone the repository and navigate to the project root:

git clone <repository-url>
cd AutoJudge_Final

---

### Create Virtual Environment

python -m venv venv

---

### Activate Virtual Environment (Windows)

venv\Scripts\activate

---

### Activate Virtual Environment (Linux / macOS)

source venv/bin/activate

---

### Install Dependencies

pip install -r requirements.txt

---

### 15.3 Run the Web Application

From the project root, run:

python app/app.py

---

### Open the Application in Browser

http://127.0.0.1:5000/

---

### 15.4 Usage

1. Paste the problem description  
2. Paste the input format  
3. Paste the output format  
4. Click **Predict Difficulty**

---

### Output Provided

- Difficulty Class: Easy / Medium / Hard  
- Predicted Rating: Codeforces-style numerical rating (e.g., 800–3000)

---

### Notes

- The application relies on pretrained classical machine learning models stored in the `artifacts/` directory.
- These `.pkl` files are required for inference and are intentionally included in the repository.
- No deep learning models, transformers, or neural networks are used anywhere in the project.

---

### End of README

