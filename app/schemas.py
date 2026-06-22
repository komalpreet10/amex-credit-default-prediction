from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    features: dict[str, float | str] = Field(
        ..., description="Customer-level model features keyed by feature name."
    )


class PredictionResponse(BaseModel):
    default_probability: float
    risk_category: str


class ModelInfoResponse(BaseModel):
    model_name: str
    problem_type: str
    output: str
    n_features: int
