from __future__ import annotations

from functools import lru_cache

from amex_default.predict import load_feature_list, load_final_model


@lru_cache(maxsize=1)
def get_model():
    return load_final_model()


@lru_cache(maxsize=1)
def get_feature_list() -> list[str]:
    return load_feature_list()
