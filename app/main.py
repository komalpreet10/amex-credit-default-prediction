from fastapi import FastAPI, HTTPException

from amex_default.predict import assign_risk_category, predict_default_probability
from app.model_loader import get_feature_list, get_model
from app.schemas import ModelInfoResponse, PredictionRequest, PredictionResponse

app = FastAPI(title="AmEx Default Prediction API", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    features = get_feature_list()
    return ModelInfoResponse(
        model_name="final_model",
        problem_type="binary_classification",
        output="default_probability",
        n_features=len(features),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        model = get_model()
        probability = predict_default_probability(model, request.features)
        return PredictionResponse(
            default_probability=probability,
            risk_category=assign_risk_category(probability),
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
