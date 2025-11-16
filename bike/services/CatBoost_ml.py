# bike/services/CatBoost_ml.py
# ---------------------------------------------------------------------------------
# 목적: CatBoost로 "대여건수/반납건수" 각각 예측 → 두 값을 빼서 예측 순변화량 산출
# 시그니처: train_eval_predict_dual(df, alpha, drop_leak)
# 반환물: (metrics, out_csv, (rent_model_path, return_model_path))
# 저장물: BASE_DIR/model_rent_cat.pkl, BASE_DIR/model_return_cat.pkl
# 주의: 기존 Ridge 파이프라인과 동일하게 "영문 컬럼" DataFrame(df)을 입력으로 받음
#       (date, location_name, rental_count, return_count, avg_temp, daily_rainfall, month, dow)
# ---------------------------------------------------------------------------------

# CatBoost_ml.py — (lag/rolling 전처리 포함, 기존 시그니처/흐름 유지)
from typing import Dict, Tuple, List
import os
import numpy as np
import pandas as pd
import joblib

from django.conf import settings

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from catboost import CatBoostRegressor

# [필수 추가] 기존 Ridge_ml의 qs_to_df 를 재사용 (DB → DataFrame 표준화)
from .Ridge_ml import qs_to_df


# ===== 유틸 =====
def _score(y_true, y_pred, with_train=False, y_tr=None, y_tr_pred=None) -> Dict[str, float]:
    # 주의: 기존 키 이름(rmse_te 등) 유지 (내부는 MSE/MAE/R2 계산, 기존 호환 목적)
    s = {
        "rmse_te": float(mean_squared_error(y_true, y_pred)),  # 기존 코드 호환 그대로
        "mae_te":  float(mean_absolute_error(y_true, y_pred)),
        "r2_te":   float(r2_score(y_true, y_pred)),
    }
    if with_train and (y_tr is not None) and (y_tr_pred is not None):
        s.update({
            "rmse_tr": float(mean_squared_error(y_tr, y_tr_pred)),
            "mae_tr":  float(mean_absolute_error(y_tr, y_tr_pred)),
            "r2_tr":   float(r2_score(y_tr, y_tr_pred)),
        })
    return s


# ===== NEW: 캘린더 & lag/rolling 전처리 =====
def _ensure_calendar_cols(df: pd.DataFrame) -> pd.DataFrame:
    """month/dow 없으면 date로부터 생성, 수치 컬럼 캐스팅 보정."""
    out = df.copy()
    if "date" not in out.columns:
        raise ValueError("df에 'date' 컬럼이 없습니다. qs_to_df() 결과를 사용하세요.")
    out["date"] = pd.to_datetime(out["date"], errors="coerce")

    if "month" not in out.columns:
        out["month"] = out["date"].dt.month
    if "dow" not in out.columns:
        out["dow"] = out["date"].dt.weekday

    # 숫자형 보정
    for c in ["avg_temp", "daily_rainfall", "rental_count", "return_count", "month", "dow"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _add_lag_rolling(df: pd.DataFrame, group_col: str = "location_name") -> pd.DataFrame:
    """
    스테이션별 lag/rolling (누수 방지: shift(1)).
    - rent_lag1, rent_lag7, rent_ma7, rent_ma14
    - ret_lag1, ret_lag7, ret_ma7, ret_ma14
    """
    d = df.sort_values([group_col, "date"]).copy()
    g = d.groupby(group_col, sort=False)

    # rental_count 기반
    d["rent_lag1"]  = g["rental_count"].shift(1)
    d["rent_lag7"]  = g["rental_count"].shift(7)
    d["rent_ma7"]  = g["rental_count"].transform(lambda s: s.rolling(7,  min_periods=1).mean().shift(1))
    d["rent_ma14"] = g["rental_count"].transform(lambda s: s.rolling(14, min_periods=1).mean().shift(1))

    # return_count 기반
    d["ret_lag1"]   = g["return_count"].shift(1)
    d["ret_lag7"]   = g["return_count"].shift(7)
    d["ret_ma7"]   = g["return_count"].transform(lambda s: s.rolling(7,  min_periods=1).mean().shift(1))
    d["ret_ma14"]  = g["return_count"].transform(lambda s: s.rolling(14, min_periods=1).mean().shift(1))

    return d


def _split_train_test_by_year(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tr = df[df["date"].dt.year == 2023].copy()
    te = df[df["date"].dt.year == 2024].copy()
    if tr.empty or te.empty:
        raise ValueError("2023/2024 데이터가 모두 필요합니다.")
    return tr, te


def _fill_lag_nas_with_train_stats(tr: pd.DataFrame, te: pd.DataFrame, lag_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    초기 구간 lag NaN 보완: 학습(2023)에서 location_name별 중앙값 → 없으면 전체 중앙값.
    """
    tr = tr.copy()
    te = te.copy()

    # 스테이션별 중앙값
    per_loc = tr.groupby("location_name")[lag_cols].median()
    global_med = tr[lag_cols].median()

    def _fill(df: pd.DataFrame) -> pd.DataFrame:
        df = df.merge(per_loc, left_on="location_name", right_index=True, how="left", suffixes=("", "_med"))
        for c in lag_cols:
            med_col = f"{c}_med"
            if med_col not in df.columns:
                df[med_col] = np.nan
            df[c] = df[c].fillna(df[med_col]).fillna(global_med[c])
            df.drop(columns=[med_col], inplace=True, errors="ignore")
        return df

    return _fill(tr), _fill(te)


# ===== CatBoost 파이프라인 (기존 구조 존중) =====
def _build_cb_pipe(features: List[str], *, drop_leak: bool, for_rent: bool) -> Pipeline:
    """
    기존 파이프 철학 유지: OneHot + StandardScaler → CatBoost.
    (대상 타깃 컬럼 자체는 drop_leak=True면 제외)
    """
    # 기본 수치 + lag 수치
    base_num = ["avg_temp", "daily_rainfall"]
    lag_num  = ["rent_lag1", "rent_lag7", "rent_ma7", "rent_ma14",
                "ret_lag1", "ret_lag7", "ret_ma7", "ret_ma14"]

    # 타깃 유출 방지(옵션)
    leak = []
    if not drop_leak:
        leak = ["rental_count"] if for_rent else ["return_count"]

    num_cols = [c for c in (base_num + lag_num + leak) if c in features]
    cat_cols = [c for c in ["location_name", "month", "dow"] if c in features]

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop",
    )

    cb = CatBoostRegressor(
        iterations=600,
        depth=8,
        learning_rate=0.06,
        loss_function="RMSE",  # 네가 준 코드 그대로 유지 (Poisson으로 바꾸고 싶으면 여기만 바꾸면 됨)
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
    )
    pipe = Pipeline([("pre", pre), ("model", cb)])
    return pipe


# ===== 메인 함수 =====
def train_eval_predict_dual(
    df: pd.DataFrame,
    alpha: float,          # 시그니처 유지(사용하지 않음)
    drop_leak: bool,
) -> Tuple[Dict, str, Tuple[str, str]]:
    """
    입력:
      - df: qs_to_df(BikeRental.objects.all()) 결과 (영문 컬럼 표준)
    처리:
      - 2023 → train, 2024 → test
      - CatBoost로 대여건수/반납건수 각각 예측
      - 예측 순변화량 = 예측 대여건수 - 예측 반납건수
      - (NEW) lag/rolling 전처리 반영
    출력:
      - metrics: {"rent": {...}, "return": {...}}   (순변화 오차 제외)
      - out_csv: MEDIA_ROOT/pred_dual_cat_2024.csv
      - (rent_model_path, return_model_path): 저장된 두 모델 경로
    """
    # 0) 캘린더/타입 보정
    df = _ensure_calendar_cols(df)

    # 1) lag/rolling 추가 (전 기간 기준으로 만들어야 2024에서 2023 정보 활용 가능)
    df = _add_lag_rolling(df, group_col="location_name")

    # 2) 학습/평가 분할
    tr, te = _split_train_test_by_year(df)

    # 3) lag 결측 보완 (학습統계 기반, 스테이션별 → 전체값 fallback)
    lag_cols = ["rent_lag1", "rent_lag7", "rent_ma7", "rent_ma14",
                "ret_lag1", "ret_lag7", "ret_ma7", "ret_ma14"]
    tr, te = _fill_lag_nas_with_train_stats(tr, te, lag_cols)

    # 4) 공통 입력 특성 (기존 + lag)
    base_features = ["location_name", "avg_temp", "daily_rainfall", "month", "dow"]
    features = base_features + [c for c in lag_cols if c in tr.columns]

    # ====== (A) 대여건수 모델 ======
    X_tr_r, y_tr_r = tr[features], tr["rental_count"].values
    X_te_r, y_te_r = te[features], te["rental_count"].values

    pipe_rent = _build_cb_pipe(features, drop_leak=drop_leak, for_rent=True)
    pipe_rent.fit(X_tr_r, y_tr_r)
    tr_pred_r = pipe_rent.predict(X_tr_r)
    te_pred_r = pipe_rent.predict(X_te_r)

    # ====== (B) 반납건수 모델 ======
    X_tr_t, y_tr_t = tr[features], tr["return_count"].values
    X_te_t, y_te_t = te[features], te["return_count"].values

    pipe_return = _build_cb_pipe(features, drop_leak=drop_leak, for_rent=False)
    pipe_return.fit(X_tr_t, y_tr_t)
    tr_pred_t = pipe_return.predict(X_tr_t)
    te_pred_t = pipe_return.predict(X_te_t)

    # 5) 예측 순변화량
    te_pred_change = te_pred_r - te_pred_t

    # 6) 지표(순변화 오차는 제외)
    metrics = {
        "rent":   _score(y_te_r, te_pred_r, with_train=True, y_tr=y_tr_r, y_tr_pred=tr_pred_r) | {"features": features},
        "return": _score(y_te_t, te_pred_t, with_train=True, y_tr=y_tr_t, y_tr_pred=tr_pred_t) | {"features": features},
    }

    # 7) CSV 저장 (경로/파일명 기존 유지)
    os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
    out = te[["date", "location_name", "rental_count", "return_count"]].copy()
    if "net_change" in te.columns:
        out["net_change"] = te["net_change"].values  # 참고용(지표에는 미사용)
    out["pred_rental_count"] = te_pred_r
    out["pred_return_count"] = te_pred_t
    out["pred_net_change"]   = te_pred_change

    out_csv = os.path.join(settings.MEDIA_ROOT, "pred_dual_cat_2024.csv")
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # 8) 모델 저장 (joblib → 기존 Holder와 호환)
    rent_model_path   = os.path.join(getattr(settings, "BASE_DIR"), "model_rent_cat.pkl")
    return_model_path = os.path.join(getattr(settings, "BASE_DIR"), "model_return_cat.pkl")
    joblib.dump(pipe_rent, rent_model_path)
    joblib.dump(pipe_return, return_model_path)


    # 9) [추가] 예측용 lag/rolling 워밍업을 위한 "히스토리 꼬리" 저장
    #      (대여소별 최근 30일 실제값)
    hist_cols = ["date", "location_name", "rental_count", "return_count"]
    hist_all = df[hist_cols].dropna().copy()          # 학습에 쓴 동일 df 기반
    hist_all = hist_all.sort_values(["location_name", "date"])
    hist_tail = (
        hist_all.groupby("location_name", group_keys=False)
                .tail(30)                              # 필요에 따라 14~60일로 조절
                .reset_index(drop=True)
    )
    hist_tail_path = os.path.join(getattr(settings, "BASE_DIR"), "cat_hist_tail.pkl")
    joblib.dump(hist_tail, hist_tail_path)

    return metrics, out_csv, (rent_model_path, return_model_path)
