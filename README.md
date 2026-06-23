# American Express Credit Default Prediction

End-to-end machine learning project for predicting customer credit default risk from monthly American Express statement data. The project covers exploratory analysis, customer-level feature engineering, LightGBM/XGBoost model training, model comparison, SHAP explainability, MLflow experiment tracking, and FastAPI model serving.

## Highlights

- Built a binary classification pipeline for credit default prediction using the American Express Default Prediction dataset.
- Engineered `3,418` customer-level features from raw monthly statement records using aggregation, lag, recent-window, first-value, and difference features.
- Compared LightGBM and XGBoost with 5-fold cross-validation on `229,456` customer-level rows.
- Deployed the final LightGBM model with FastAPI through a single `/predict` endpoint that accepts raw customer statement rows and runs feature engineering before prediction.

## Results

| Model | Rows | Features | ROC-AUC | PR-AUC | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LightGBM | 229,456 | 3,418 | 0.9593 | 0.8938 | 0.8104 | 0.8059 | 0.8081 |
| XGBoost | 229,456 | 3,418 | 0.9597 | 0.8948 | 0.8124 | 0.8035 | 0.8079 |

The final serving model is LightGBM trained on all engineered training rows. Reported performance uses 5-fold out-of-fold cross-validation metrics.

## LightGBM Evaluation

![LightGBM ROC curve](docs/images/lightgbm_roc_curve.png)

![LightGBM precision recall curve](docs/images/lightgbm_pr_curve.png)

![LightGBM confusion matrix](docs/images/lightgbm_confusion_matrix.png)

## Explainability

![LightGBM feature importance](docs/images/lightgbm_feature_importance.png)

![LightGBM SHAP summary](docs/images/lightgbm_shap_summary.png)

## Project Structure

```text
amex-credit-default/
├── app/                         # FastAPI application
│   ├── main.py
│   ├── model_loader.py
│   └── schemas.py
├── docs/images/                 # README plots
├── notebooks/                   # EDA, training, comparison, SHAP, MLflow, API demo
├── src/amex_default/            # Reusable ML pipeline code
│   ├── data.py
│   ├── evaluate.py
│   ├── features.py
│   ├── interpret.py
│   ├── predict.py
│   ├── tracking.py
│   ├── train_lightgbm.py
│   └── train_xgboost.py
├── artifacts/                   # Local generated models, plots, reports
├── data/                        # Local raw, processed, and prediction data
├── mlruns/                      # Local MLflow experiment store
├── requirements.txt
└── README.md
```

`data/`, `artifacts/`, and `mlruns/` are local generated outputs and are ignored by Git.

## Pipeline

1. Preprocess monthly customer statement rows.
2. Build customer-level features from raw statement history.
3. Train LightGBM and XGBoost with stratified 5-fold cross-validation.
4. Save out-of-fold predictions, metrics, reports, plots, and model artifacts.
5. Compare model performance and inference/training time.
6. Generate feature importance and SHAP explainability plots.
7. Serve the final LightGBM model through FastAPI.

## FastAPI Serving

Start the API from the project root:

```bash
PYTHONPATH=src uvicorn app.main:app --reload
```

Open the interactive API docs:

```text
http://127.0.0.1:8000/docs
```

Prediction endpoint:

```text
POST /predict
```

The endpoint accepts raw monthly statement rows for one `customer_ID`, runs feature engineering, aligns the generated features to the trained model, and returns a default probability.

Example request:

```json
{
  "statements": [
    {
      "customer_ID": "customer-1",
      "S_2": "2018-03-01",
      "P_2": 0.72,
      "D_39": 0.01,
      "B_1": 0.05,
      "B_2": 0.81,
      "D_63": "CR",
      "D_64": "O"
    }
  ]
}
```

Example response:

```json
{
  "default_probability": 0.0008122114740618201,
  "risk_category": "low",
  "customer_id": "customer-1",
  "n_statements": 13,
  "n_engineered_features": 3418
}
```

## MLflow Tracking

Start the local MLflow UI:

```bash
MLFLOW_ALLOW_FILE_STORE=true mlflow ui --backend-store-uri ./mlruns --port 5001
```

Open:

```text
http://127.0.0.1:5001
```

Tracked runs include:

- `lightgbm`
- `xgboost`
- `model_comparison`
- `final_lightgbm_model`

Tracked metrics include ROC-AUC, PR-AUC, precision, recall, F1, confusion matrix counts, training time, inference time, and fold-level cross-validation metrics.

Tracked artifacts include metrics reports, feature importance files, out-of-fold predictions, ROC/PR curves, confusion matrices, SHAP plots, model comparison plots, and final model files.

## Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- LightGBM
- XGBoost
- SHAP
- MLflow
- FastAPI
