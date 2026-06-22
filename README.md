# American Express Default Prediction

## Overview

This project predicts customer credit default risk using the American Express Default Prediction dataset. It is organized as a practical portfolio-quality machine learning project focused on feature engineering, model development, model comparison, explainability, lightweight experiment tracking, and simple model serving.

Current best out-of-fold ROC-AUC is approximately `0.95-0.96`.

## Project Structure

```text
amex-credit-default/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/
│   ├── processed/
│   │   ├── train_features.parquet
│   │   └── model_comparison.csv
│   └── predictions/
│       ├── lightgbm_oof.parquet
│       └── xgboost_oof.parquet
├── notebooks/
│   ├── 01_eda_preprocessing.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_lightgbm_training.ipynb
│   ├── 04_xgboost_training.ipynb
│   ├── 05_model_comparison.ipynb
│   ├── 06_model_interpretation_shap.ipynb
│   ├── 07_mlflow_tracking.ipynb
│   └── 08_fastapi_demo.ipynb
├── src/amex_default/
├── app/
└── artifacts/
```

## Workflow

1. EDA and missing value analysis
2. High-missing column removal and dtype optimization
3. Customer-level feature engineering and aggregation
4. Model training with consistent 5-fold Stratified Cross Validation
5. Model comparison across LightGBM and XGBoost
6. Feature importance and SHAP explainability
7. Lightweight MLflow experiment tracking
8. FastAPI inference service for the final LightGBM model

## Model Results

The standardized comparison file is saved at:

```text
data/processed/model_comparison.csv
```

Tracked metrics:

- ROC-AUC
- PR-AUC
- Precision
- Recall
- F1
- Training time
- Inference time

LightGBM is selected as the final serving model because it offers comparable OOF performance to XGBoost with faster training and simpler serving. XGBoost remains in the project as the stronger-comparison benchmark.

## Explainability

Model interpretation is handled in `notebooks/07_model_interpretation_shap.ipynb`.

Generated artifacts should include:

```text
artifacts/plots/lightgbm_feature_importance.png
artifacts/plots/xgboost_feature_importance.png
artifacts/plots/lightgbm_shap_summary.png
artifacts/plots/lightgbm_shap_bar.png
artifacts/plots/xgboost_shap_summary.png
artifacts/plots/xgboost_shap_bar.png
```

## MLflow

MLflow is used locally to track:

- Model family
- Parameters
- Metrics
- OOF prediction artifacts
- Feature importance plots
- SHAP plots
- Model artifacts

Start the local UI with:

```bash
mlflow ui
```

## FastAPI Demo

The API serves the selected final model and expects customer-level engineered features.

Run from the project root:

```bash
PYTHONPATH=src uvicorn app.main:app --reload
```

Endpoints:

```text
GET  /health
GET  /model-info
POST /predict
```

Example request:

```json
{
  "features": {
    "P_2_mean": 0.81,
    "B_1_mean": 0.03
  }
}
```

Example response:

```json
{
  "default_probability": 0.18,
  "risk_category": "low"
}
```

## Reproduce

Install dependencies:

```bash
pip install -r requirements.txt
```

Run notebooks in order:

```text
01_eda_preprocessing.ipynb
02_feature_engineering.ipynb
03_lightgbm_training.ipynb
04_xgboost_training.ipynb
05_model_comparison.ipynb
06_model_interpretation_shap.ipynb
07_mlflow_tracking.ipynb
08_fastapi_demo.ipynb
```

## Key Skills Demonstrated

- Customer-level feature engineering on longitudinal credit data
- Missing value analysis and dtype optimization
- Cross-validated model development
- LightGBM and XGBoost comparison
- OOF validation and imbalanced-class metrics
- SHAP-based explainability
- MLflow experiment tracking
- Simple FastAPI model serving

## Resume Bullets

- Built an end-to-end credit default prediction project using American Express customer data, including missing value analysis, dtype optimization, customer-level feature engineering, model training, explainability, and deployment.
- Trained and compared LightGBM and XGBoost models using 5-fold Stratified Cross Validation, achieving approximately `0.95-0.96` out-of-fold ROC-AUC.
- Created a standardized evaluation workflow using ROC-AUC, PR-AUC, precision, recall, F1, training time, and inference time to support transparent model selection.
- Applied feature importance and SHAP analysis to identify key drivers of default risk and communicate model behavior.
- Integrated MLflow for lightweight local experiment tracking across model families.
- Built a FastAPI inference service exposing `/health`, `/model-info`, and `/predict` endpoints for serving default probabilities and risk categories.
