AutoJudge – Predicting Programming Problem Difficulty
1. Project Overview

AutoJudge is a machine learning system that predicts the difficulty of programming problems based solely on their textual descriptions.
The system outputs:

A difficulty class: Easy / Medium / Hard

A numerical difficulty score aligned with Codeforces-style ratings (e.g., 800–3000)

The project fulfills the requirements of the problem statement by implementing both classification and regression models, along with proper evaluation and a web-based interface.

All models use classical machine learning techniques and classical text feature extraction methods.
No deep learning models are used anywhere in the pipeline.

2. Problem Statement Requirements

The problem statement required:

Difficulty classification

Difficulty score prediction

Model evaluation

A web interface for predictions

All of the above have been implemented using classical machine learning models, strictly adhering to the given constraints.

3. Datasets Used
3.1 Initial Provided Dataset

The initially provided dataset contained programming problem descriptions along with limited difficulty-related information.

Issues observed during experimentation:

No explicit difficulty class labels

Difficulty scores clustered in a narrow range

Regression models collapsed to near-constant predictions

Classification performance was unstable due to weak label signals

Despite extensive feature engineering and model tuning, this dataset did not provide sufficient variance for reliable difficulty prediction.

The dataset and experiments from this phase are retained for transparency and comparison.

3.2 Improved Dataset (Codeforces-Based)

To address the above limitations, a larger and more diverse dataset was constructed using Codeforces problem data.

Each problem includes:

Problem description

Input format

Output format

Official Codeforces difficulty rating

Rows with missing input or output descriptions were removed to ensure data quality.

4. Difficulty Class Generation

The Codeforces rating was used to generate difficulty class labels only during dataset construction.

Rating Range	Difficulty Class
≤ 1200	Easy
1200–1799	Medium
≥ 1800	Hard

During classification model training, the rating column is not used, preventing any form of label leakage.

5. Exploratory Data Analysis (EDA)

Key observations from EDA:

Strong class imbalance (Hard problems dominate)

Significant variation in text length and vocabulary

Difficulty correlates with textual and structural complexity

Class imbalance handling strategies were therefore required.

A detailed exploratory analysis, including class distribution plots and text statistics, can be found in
notebooks/eda.ipynb, which provides additional visualization and clarification of these observations.

6. Feature Engineering

The project uses classical text feature extraction techniques, including:

TF-IDF vectorization

Unigrams and bigrams (n-grams)

Dimensionality reduction using Truncated SVD

No pretrained embeddings, transformers, or neural networks are used.

7. Classification Models
7.1 Models Evaluated

Logistic Regression (class-weighted)

Linear SVM (class-weighted)

XGBoost Classifier

7.2 Handling Class Imbalance

The dataset exhibits significant class imbalance, particularly with a higher proportion of Hard problems.

The following strategies were explored:

Class weighting in linear models

SMOTE-based oversampling during experimentation

While SMOTE was evaluated, it did not lead to consistent improvements in validation performance.
Therefore, the final classification model relies on class-weighted learning rather than synthetic oversampling.

7.3 Evaluation Metrics

Accuracy

Precision, Recall, F1-score

Confusion Matrix

7.4 Final Classification Model

Logistic Regression (class-weighted) was selected as the final classifier due to:

Stable performance across all classes

Better balance between precision and recall

Lower overfitting risk compared to tree-based models

8. Regression Models (Difficulty Score Prediction)
8.1 Objective

Predict a numerical difficulty score aligned with Codeforces ratings using text-only information.

8.2 Models Evaluated

Linear Regression

Gradient Boosting Regressor

XGBoost Regressor

Random Forest Regressor (with hyperparameter tuning)

8.3 Evaluation Metrics

MAE (Mean Absolute Error)

MSE (Mean Squared Error)

RMSE (Root Mean Squared Error)

8.4 Final Regression Model

The final regression model achieves:

MAE ≈ 470 rating points

Given the subjective and noisy nature of difficulty ratings, this performance is considered reasonable for text-based prediction.

9. Final System Architecture

The system uses two independent models:

Regression model predicts a numerical difficulty rating

Classification model predicts Easy / Medium / Hard directly from text

The outputs are combined at inference time without threshold-based post-processing.

10. Web Application

Flask backend

HTML/CSS frontend inspired by Codeforces UI

Real-time predictions

User inputs:

Problem description

Input format

Output format

11. Results Summary
Task	Model	Key Metric
Classification	Logistic Regression	~60% accuracy
Regression	Codeforces Rating Model	MAE ≈ 470
12. Project Structure
AutoJudge_Final/
│
├── app/
│   ├── app.py
│   ├── templates/
│   └── static/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── data_generation/
│   └── models/
│
├── artifacts/
│   ├── feature_pipeline_classification.pkl
│   ├── logistic_regression_baseline.pkl
│   ├── tfidf_cf.pkl
│   └── cf_rating_model.pkl
│
├── notebooks/
├── README.md
└── requirements.txt

13. Limitations & Future Work

Class imbalance remains a challenge

Difficulty prediction is inherently subjective

Possible improvements:

Incorporating symbolic features (constraints, limits)

Larger curated datasets

Multi-task learning approaches

14. Conclusion

AutoJudge successfully implements both classification and regression models for programming problem difficulty prediction using classical machine learning techniques.
The system is fully functional, empirically evaluated, and strictly adheres to the problem constraints.

15. How to Run the Project
15.1 Prerequisites

Python 3.9+

pip

15.2 Setup

Clone the repository and navigate to the project root:

git clone <repository-url>
cd AutoJudge_Final


(Optional but recommended) Create a virtual environment:

Windows

python -m venv venv
venv\Scripts\activate


Linux / macOS

python3 -m venv venv
source venv/bin/activate


Install dependencies:

pip install -r requirements.txt

15.3 Run the Web Application

From the project root:

python app/app.py


The application will be available at:

http://127.0.0.1:5000/

15.4 Usage

Paste the problem description

Paste the input format

Paste the output format

Click Predict Difficulty

The system outputs:

Difficulty class (Easy / Medium / Hard)

Predicted Codeforces-style rating