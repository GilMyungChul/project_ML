# bike/services/RandomForest_ml.py (FAST)
from typing import Dict, Tuple, List
import os, joblib, numpy as np, pandas as pd
from django.conf import settings
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# [필수] DB→DF 표준화
from .Ridge_ml import qs_to_df

# ======================== 유틸 & 스코어 =========================
DATE_COL = "date"
CAT_LOC  = "location_name"
COL_TEMP = "avg_temp"
COL_RAIN = "daily_rainfall"

def _rmse(y, p) -> float:
    return float(np.sqrt(mean_squared_error(y, p)))

def _score(y_true, y_pred, with_train=False, y_tr=None, y_tr_pred=None) -> Dict[str, float]:
    s = {"rmse_te": _rmse(y_true, y_pred),
         "mae_te":  float(mean_absolute_error(y_true, y_pred)),
         "r2_te":   float(r2_score(y_true, y_pred))}
    if with_train:
        s |= {"rmse_tr": _rmse(y_tr, y_tr_pred),
              "mae_tr":  float(mean_absolute_error(y_tr, y_tr_pred)),
              "r2_tr":   float(r2_score(y_tr, y_tr_pred))}
    return s

def _to_num(s, default=0):
    return pd.to_numeric(s, errors="coerce").fillna(default)

# ===================== 빠른 피처 엔지니어링 =====================
def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL])
    # 정렬(그룹 lag 계산 안정)
    sort_cols = [c for c in [CAT_LOC, DATE_COL] if c in df.columns]
    df = df.sort_values(sort_cols)
    # 기본 파생
    df["month"] = df[DATE_COL].dt.month.astype("int16")
    df["dow"]   = df[DATE_COL].dt.weekday.astype("int16")
    # float32 캐스팅
    if COL_TEMP in df.columns: df[COL_TEMP] = _to_num(df[COL_TEMP]).astype("float32")
    if COL_RAIN in df.columns: df[COL_RAIN] = _to_num(df[COL_RAIN]).astype("float32")
    # 위치 정수코드(원핫 대신) → 속도/메모리 절약
    if CAT_LOC in df.columns:
        cat = df[CAT_LOC].astype("category")
        df["loc_code"] = cat.cat.codes.astype("int32")
        df.attrs["loc_categories"] = list(cat.cat.categories)  # 모델 저장 시 같이 남김
    else:
        df["loc_code"] = 0
        df.attrs["loc_categories"] = []
    return df

def _add_common_features_fast(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 가벼운 사이클릭
    df["dow_sin"] = np.sin(2*np.pi*df["dow"]/7).astype("float32")
    df["dow_cos"] = np.cos(2*np.pi*df["dow"]/7).astype("float32")
    df["mon_sin"] = np.sin(2*np.pi*df["month"]/12).astype("float32")
    df["mon_cos"] = np.cos(2*np.pi*df["month"]/12).astype("float32")
    # 강수/온도 최소 파생
    rain = _to_num(df.get(COL_RAIN, 0)).astype("float32")
    temp = _to_num(df.get(COL_TEMP, 0)).astype("float32")
    df["rain_flag"] = (rain > 0).astype("int8")
    df["is_weekend"] = (df["dow"] >= 5).astype("int8")
    # 선택: 필요한 정도만
    return df

def _add_dual_lags(df: pd.DataFrame) -> pd.DataFrame:
    """대여/반납 각각의 lag/rolling을 한 번에 계산(그룹별)."""
    df = df.copy()
    if CAT_LOC in df.columns:
        g_r = df.groupby(CAT_LOC)["rental_count"]
        g_t = df.groupby(CAT_LOC)["return_count"]
    else:
        g_r = df["rental_count"]
        g_t = df["return_count"]

    # rental 타깃용
    df["rent_lag1"]   = _to_num(g_r.shift(1)).fillna(0).astype("float32")
    df["rent_lag7"]   = _to_num(g_r.shift(7)).fillna(0).astype("float32")
    df["rent_rmean7"] = _to_num(g_r.shift(1).rolling(7, min_periods=1).mean()).fillna(0).astype("float32")
    # return 타깃용
    df["ret_lag1"]   = _to_num(g_t.shift(1)).fillna(0).astype("float32")
    df["ret_lag7"]   = _to_num(g_t.shift(7)).fillna(0).astype("float32")
    df["ret_rmean7"] = _to_num(g_t.shift(1).rolling(7, min_periods=1).mean()).fillna(0).astype("float32")

    # 상호작용(가벼운 것만)
    df["rent_lag1_x_rain"] = (df["rent_lag1"] * (df["rain_flag"] > 0).astype("int8")).astype("float32")
    df["ret_lag1_x_rain"]  = (df["ret_lag1"]  * (df["rain_flag"] > 0).astype("int8")).astype("float32")
    df["rent_lag1_x_wknd"] = (df["rent_lag1"] * df["is_weekend"].astype("int8")).astype("float32")
    df["ret_lag1_x_wknd"]  = (df["ret_lag1"]  * df["is_weekend"].astype("int8")).astype("float32")
    return df

def _build_design_mats(df: pd.DataFrame, *, drop_leak: bool):
    """공통+듀얼 lag 생성 후, 대여/반납용 X 컬럼 나눠서 반환."""
    df = _prepare(df)
    df = _add_common_features_fast(df)
    df = _add_dual_lags(df)
    # 공통/가벼운 피처 리스트
    base = ["loc_code","month","dow","dow_sin","dow_cos","mon_sin","mon_cos","rain_flag"]
    if COL_TEMP in df.columns: base.append(COL_TEMP)
    if COL_RAIN in df.columns: base.append(COL_RAIN)

    rent_feats = base + ["rent_lag1","rent_lag7","rent_rmean7","rent_lag1_x_rain","rent_lag1_x_wknd"]
    ret_feats  = base + ["ret_lag1","ret_lag7","ret_rmean7","ret_lag1_x_rain","ret_lag1_x_wknd"]

    # 누수 방지: 현재시점 타깃/상대타깃은 X에서 제외(기본 True)
    if not drop_leak:
        pass  # 정말 필요하면 여기서 현재시점 카운트 넣어라(권장X)

    X_all_r = df[rent_feats].astype("float32")
    X_all_t = df[ret_feats].astype("float32")
    y_all_r = _to_num(df["rental_count"]).astype("float32")
    y_all_t = _to_num(df["return_count"]).astype("float32")
    year    = df[DATE_COL].dt.year
    meta    = {"loc_categories": df.attrs.get("loc_categories", [])}
    return (X_all_r, y_all_r), (X_all_t, y_all_t), year, meta

# ======================= 모델 구성(빠른 설정) =======================
def _make_rf_fast(n_estimators=200, max_depth=18, min_samples_leaf=5, random_state=42):
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        n_jobs=-1,
        random_state=random_state,
    )

# ======================= 학습/평가/저장/예측 =======================
def train_eval_predict_dual(
    df: pd.DataFrame,
    alpha: float,    # 시그니처 유지(미사용)
    drop_leak: bool,
) -> Tuple[Dict, str, Tuple[str, str]]:
    """
    2023→train, 2024→test. 빠른 피처/빠른 RF로 속도 개선.
    반환: (metrics, out_csv, (rent_model_path, return_model_path))
    """
    if DATE_COL not in df.columns:
        raise ValueError(f"df에 '{DATE_COL}' 컬럼이 없어.")

    # 디자인 매트릭스(공통 생성 1회)
    (X_all_r, y_all_r), (X_all_t, y_all_t), year, meta = _build_design_mats(df, drop_leak=drop_leak)

    tr = (year == 2023); te = (year == 2024)
    if not tr.any() or not te.any():
        raise ValueError("2023/2024 데이터가 모두 필요해.")

    # 대여 모델
    rf_r = _make_rf_fast()
    X_tr_r, y_tr_r = X_all_r.loc[tr].values, y_all_r.loc[tr].values
    X_te_r, y_te_r = X_all_r.loc[te].values, y_all_r.loc[te].values
    rf_r.fit(X_tr_r, y_tr_r)
    tr_pred_r = rf_r.predict(X_tr_r); te_pred_r = rf_r.predict(X_te_r)

    # 반납 모델
    rf_t = _make_rf_fast()
    X_tr_t, y_tr_t = X_all_t.loc[tr].values, y_all_t.loc[tr].values
    X_te_t, y_te_t = X_all_t.loc[te].values, y_all_t.loc[te].values
    rf_t.fit(X_tr_t, y_tr_t)
    tr_pred_t = rf_t.predict(X_tr_t); te_pred_t = rf_t.predict(X_te_t)

    # 순변화 예측
    te_pred_change = te_pred_r - te_pred_t

    # 지표
    metrics = {
        "rent":   _score(y_te_r, te_pred_r, with_train=True, y_tr=y_tr_r, y_tr_pred=tr_pred_r) |
                  {"n_features": X_all_r.shape[1]},
        "return": _score(y_te_t, te_pred_t, with_train=True, y_tr=y_tr_t, y_tr_pred=tr_pred_t) |
                  {"n_features": X_all_t.shape[1]},
    }

    # CSV 저장
    dfx = df.copy()
    dfx[DATE_COL] = pd.to_datetime(dfx[DATE_COL], errors="coerce")
    out = dfx.loc[te, [DATE_COL, CAT_LOC, "rental_count", "return_count"]].copy()
    if "net_change" in dfx.columns:
        out["net_change"] = dfx.loc[te, "net_change"].values
    out["pred_rental_count"] = te_pred_r
    out["pred_return_count"] = te_pred_t
    out["pred_net_change"]   = te_pred_change

    os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
    out_csv = os.path.join(settings.MEDIA_ROOT, "pred_dual_rf_2024.csv")
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # 모델+메타 저장(로케이션 카테고리 매핑 같이 저장)
    base_dir = getattr(settings, "BASE_DIR")
    rent_model_path   = os.path.join(base_dir, "model_rent_rf.pkl")
    return_model_path = os.path.join(base_dir, "model_return_rf.pkl")
    joblib.dump({"model": rf_r, "loc_categories": meta["loc_categories"]}, rent_model_path)
    joblib.dump({"model": rf_t, "loc_categories": meta["loc_categories"]}, return_model_path)

    return metrics, out_csv, (rent_model_path, return_model_path)
