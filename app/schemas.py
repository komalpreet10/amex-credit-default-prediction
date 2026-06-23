from typing import Any

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    statements: list[dict[str, Any]] = Field(
        ...,
        description=(
            "Raw monthly AMEX statement rows. Each row must include customer_ID; "
            "S_2 is recommended so recent-window and diff features are ordered correctly."
        ),
    )


class PredictionResponse(BaseModel):
    default_probability: float
    risk_category: str
    customer_id: str | None = None
    n_statements: int | None = None
    n_engineered_features: int | None = None
