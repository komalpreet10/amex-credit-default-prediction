from fastapi import FastAPI, HTTPException

from amex_default.predict import (
    assign_risk_category,
    predict_default_probability_from_statements,
)
from app.model_loader import get_model
from app.schemas import (
    PredictionRequest,
    PredictionResponse,
)

app = FastAPI(
    title="AmEx Default Prediction API",
    version="0.1.0",
    swagger_ui_parameters={"defaultModelsExpandDepth": -1},
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        model = get_model()
        probability, engineered_features = predict_default_probability_from_statements(
            model,
            request.statements,
        )
        customer_id = str(engineered_features["customer_ID"].iloc[0])
        return PredictionResponse(
            default_probability=probability,
            risk_category=assign_risk_category(probability),
            customer_id=customer_id,
            n_statements=len(request.statements),
            n_engineered_features=len(engineered_features.columns) - 1,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
