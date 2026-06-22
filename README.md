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
