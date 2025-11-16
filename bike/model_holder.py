import os, joblib
from django.conf import settings

MODEL_PATH_RENT = os.path.join(getattr(settings, "BASE_DIR"), "model_rent.pkl")
MODEL_PATH_RETURN = os.path.join(getattr(settings, "BASE_DIR"), "model_return.pkl")

class MlModelHolder:
    _model_rent = None
    _model_return = None

    @classmethod
    def try_lazy_load(cls):
        if cls._model_rent is None and os.path.exists(MODEL_PATH_RENT):
            cls._model_rent = joblib.load(MODEL_PATH_RENT)
        if cls._model_return is None and os.path.exists(MODEL_PATH_RETURN):
            cls._model_return = joblib.load(MODEL_PATH_RETURN)

    @classmethod
    def force_load(cls, rent_path: str = None, return_path: str = None):
        cls._model_rent = joblib.load(rent_path or MODEL_PATH_RENT)
        cls._model_return = joblib.load(return_path or MODEL_PATH_RETURN)

    @classmethod
    def predict_df(cls, df):
        if cls._model_rent is None or cls._model_return is None:
            raise RuntimeError("model_rent.pkl / model_return.pkl이 로드되지 않았어. 먼저 학습을 실행해.")

        y_pred_rent = cls._model_rent.predict(df)
        y_pred_return = cls._model_return.predict(df)

        # 순변화량 = 대여건수 - 반납건수
        y_pred_change = y_pred_rent - y_pred_return

        return {
            "y_pred_rent": y_pred_rent,
            "y_pred_return": y_pred_return,
            "y_pred_change": y_pred_change
        }