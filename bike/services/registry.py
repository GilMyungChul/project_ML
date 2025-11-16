# bike/services/registry.py
from typing import Dict
from .Ridge_ml import qs_to_df, train_eval_predict_dual  # Ridge모델
from .CatBoost_ml import train_eval_predict_dual as catboost_dual  # CatBoost모델
from .RandomForest_ml import train_eval_predict_dual as rf_dual   # RandomForest모델



# 모델 레지스트리: 추후 XGB, CatBoost 추가 시 여기만 늘리면 됨
_REGISTRY: Dict[str, Dict] = {
    "ridge_dual": {
        "label": "Ridge",
        "train": train_eval_predict_dual,        # (df, alpha, drop_leak) -> metrics, out_csv, model_paths
        "template": "bike/model_predict.html",   # 결과 템플릿(공용)
    },
    "catboost_dual": {
    "label": "CatBoost",
    "train": catboost_dual,                     # ← 방금 만든 함수
    "template": "bike/model_predict.html",
    },
    "rf_dual": {  # [필수 추가]
        "label": "RandomForest",
        "train": rf_dual,
        "template": "bike/model_predict.html",
    },
}

def available_models() -> Dict[str, Dict]:
    return _REGISTRY

def get_model_spec(key: str) -> Dict:
    if key not in _REGISTRY:
        raise ValueError(f"등록되지 않은 모델 키: {key}")
    return _REGISTRY[key]
