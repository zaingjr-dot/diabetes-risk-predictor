# Diabetes Risk Predictor 

A machine learning system for diabetes risk prediction built with XGBoost 
and SHAP explainability, deployed as an interactive Streamlit web application.

## Project Overview

This project predicts diabetes risk from eight clinical measurements using 
a complete end-to-end machine learning pipeline. It was developed to 
demonstrate applied ML in a healthcare context, with emphasis on data 
quality, clinical evaluation metrics, and model interpretability.

## Results

| Metric | Score |
|---|---|
| ROC-AUC (Test Set) | 0.95 |
| Diabetic Recall | 85% |
| Cross-Validation AUC | 0.97 ± 0.008 |
| Test Patients | 154 |

## Technical Highlights

- Detected and resolved biologically impossible zero values across five 
  clinical features using conditional median imputation grouped by outcome
- Addressed 65/35 class imbalance using SMOTE applied exclusively to 
  training data to preserve test set integrity
- Selected XGBoost for its capacity to model non-linear feature interactions
- Integrated SHAP explainability to produce feature-level interpretations 
  for individual patient predictions
 

## Technologies Used

Python, pandas, NumPy, scikit-learn, XGBoost, imbalanced-learn, 
SHAP, Streamlit, matplotlib, seaborn, Jupyter Notebook

## Dataset

PIMA Indians Diabetes Dataset — UCI Machine Learning Repository  
768 patients, 8 clinical features, binary outcome (diabetes present/absent)

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Author

[Muhammad Zain] — [zaingjr@gmail.com]

