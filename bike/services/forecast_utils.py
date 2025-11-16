from __future__ import annotations
import os, joblib
from collections import defaultdict
from typing import Tuple, Dict, List, Optional
import numpy as np
import pandas as pd
from django.conf import settings
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# 업로드 CSV에서 필요한 최소 컬럼: date, location_name, avg_temp, daily_rainfall
REQUIRED_PRED_COLS = ["date","location_name","avg_temp","daily_rainfall"]

def load_future_csv_to_df(fpath: str) -> pd.DataFrame:
    # 인코딩 유연
    try:
        df = pd.read_csv(fpath, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(fpath, encoding="cp949")
    df = df.copy()
    # 컬럼 이름 표준화(사용자 CSV가 한글일 수 있으면 여기서 매핑 추가해도 됨)
    # 필요시: {'일시':'date','대여소명':'location_name','평균기온(°C)':'avg_temp','일강수량(mm)':'daily_rainfall'}
    mapping = {
        "일시": "date",
        "DATE": "date",
        "날짜": "date",

        "대여소명": "location_name",
        "지점명": "location_name",

        "평균기온(°C)": "avg_temp",
        "평균기온": "avg_temp",
        "temp": "avg_temp",

        "일강수량(mm)": "daily_rainfall",
        "강수량": "daily_rainfall",
        "rain": "daily_rainfall",
    }
    # df = df.rename(columns={...})
    df = df.rename(columns=lambda c: mapping.get(c.strip(), c.strip()))

    # 최소 컬럼 보정
    for c in REQUIRED_PRED_COLS:
        if c not in df.columns:
            raise ValueError(f"필수 컬럼 누락: {c}")

    # 타입/파생
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["location_name"] = df["location_name"].astype(str)
    for c in ["avg_temp","daily_rainfall"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["month"] = df["date"].dt.month
    df["dow"] = df["date"].dt.dayofweek

    if df[["date","location_name","avg_temp","daily_rainfall","month","dow"]].isna().any().any():
        raise ValueError("전처리 후 결측치가 있어. CSV 내용을 확인해줘.")
    return df

# 모델 파일 경로 매핑 (기존 저장 규칙 활용)
MODEL_FILES = {
    "ridge_dual": ("model_rent_ridge.pkl", "model_return_ridge.pkl"),
    "catboost_dual": ("model_rent_cat.pkl", "model_return_cat.pkl"),
    "rf_dual": ("model_rent_rf.pkl", "model_return_rf.pkl"),
    # fast/full 같은 파생키가 있다면 같은 파일을 가리키거나 별도 저장 규칙을 쓰도록 매핑
    "rf_dual_fast": ("model_rent_rf.pkl", "model_return_rf.pkl"),
    "rf_dual_full": ("model_rent_rf.pkl", "model_return_rf.pkl"),
}

def load_models_for_key(model_key: str):
    if model_key not in MODEL_FILES:
        raise ValueError(f"알 수 없는 모델 키: {model_key}")
    rent_name, return_name = MODEL_FILES[model_key]
    base = getattr(settings, "BASE_DIR")
    rent_path = os.path.join(base, rent_name)
    return_path = os.path.join(base, return_name)
    rent_model = joblib.load(rent_path)
    return_model = joblib.load(return_path)
    return rent_model, return_model, (rent_path, return_path)
'''
def predict_future(df: pd.DataFrame, rent_model, return_model):
    # 학습 때 쓴 공통 피처와 동일하게
    features = ["location_name","avg_temp","daily_rainfall","month","dow"]
    X = df[features]
    pred_r = rent_model.predict(X)
    pred_t = return_model.predict(X)
    df_out = df.copy()
    df_out["pred_rental_count"] = pred_r
    df_out["pred_return_count"] = pred_t
    df_out["pred_net_change"] = df_out["pred_rental_count"] - df_out["pred_return_count"]
    return df_out
'''
'''
def predict_future(df: pd.DataFrame, rent_model, return_model):
    """
    업로드 DF: [일시, 대여소명, 기온, 강수량]
    - Ridge/RandomForest 등 CatBoost가 아니면 기존 방식(rolling/lag 미사용)
    - CatBoost일 때만: DB 히스토리로 lag/rolling 워밍업 후 날짜별 재귀 예측
      (히스토리가 없으면 0으로 부트스트랩)
    """
    import numpy as np
    from collections import defaultdict
    from sklearn.pipeline import Pipeline
    from catboost import CatBoostRegressor

    # ---- 0) 한글 → 내부 표준 컬럼명 정규화 ----
    colmap = {
        "일시": "date",
        "대여소명": "location_name",
        "기온": "avg_temp",
        "강수량": "daily_rainfall",
    }
    df_std = df.rename(columns={k: v for k, v in colmap.items() if k in df.columns}).copy()

    need = {"date", "location_name", "avg_temp", "daily_rainfall"}
    miss = need - set(df_std.columns)
    if miss:
        raise ValueError(f"예측 입력에 필요한 컬럼이 없습니다: {sorted(miss)}")

    df_std["date"] = pd.to_datetime(df_std["date"], errors="coerce")
    if df_std["date"].isna().any():
        bad = df_std[df_std["date"].isna()].head(3).to_dict(orient="records")
        raise ValueError(f"'date(일시)' 파싱 실패 행이 있습니다. 예: {bad}")

    df_std["location_name"] = df_std["location_name"].astype(str).str.strip()
    df_std["avg_temp"] = pd.to_numeric(df_std["avg_temp"], errors="coerce")
    df_std["daily_rainfall"] = pd.to_numeric(df_std["daily_rainfall"], errors="coerce")
    df_std["month"] = df_std["date"].dt.month
    df_std["dow"]   = df_std["date"].dt.weekday

    # ---- 1) 모델 타입 판별 ----
    def _final_est(m):
        if isinstance(m, Pipeline):
            return m.steps[-1][1]
        return m
    is_catboost = isinstance(_final_est(rent_model), CatBoostRegressor) and isinstance(_final_est(return_model), CatBoostRegressor)

    base_features = ["location_name", "avg_temp", "daily_rainfall", "month", "dow"]

    # ---- 2) CatBoost가 아니면: 기존 방식 그대로 ----
    if not is_catboost:
        features = [c for c in base_features if c in df_std.columns]
        X = df_std[features]
        pred_r = rent_model.predict(X)
        pred_t = return_model.predict(X)
        out = df_std.copy()
        out["pred_rental_count"] = np.clip(pred_r, 0, None)
        out["pred_return_count"] = np.clip(pred_t, 0, None)
        out["pred_net_change"]   = out["pred_rental_count"] - out["pred_return_count"]
        return out

    # ---- 3) CatBoost: lag/rolling 포함 재귀 예측 (히스토리 워밍업) ----
    lag_cols = ["rent_lag1","rent_lag7","rent_ma7","rent_ma14",
                "ret_lag1","ret_lag7","ret_ma7","ret_ma14"]
    feats = base_features + lag_cols

    # 3-1) 스테이션별 히스토리 시드: cat_hist_tail.pkl → 없으면 DB → 없으면 0
    import os, joblib
    rent_hist = defaultdict(list)
    ret_hist  = defaultdict(list)

    first_day = df_std["date"].min()
    stations  = df_std["location_name"].unique().tolist()

    # (1) 학습 시 저장한 꼬리 파일 시도
    tail_path = os.path.join(getattr(settings, "BASE_DIR"), "cat_hist_tail.pkl")
    if os.path.exists(tail_path):
        hist_tail = joblib.load(tail_path)
        # 스키마 안전화
        hist_tail["date"] = pd.to_datetime(hist_tail["date"], errors="coerce")
        seed = (
            hist_tail[ (hist_tail["location_name"].isin(stations)) &
                       (hist_tail["date"] < first_day) ]
            .sort_values(["location_name","date"])
        )
        for st, g in seed.groupby("location_name"):
            rc = pd.to_numeric(g["rental_count"], errors="coerce").fillna(0).tolist()
            tc = pd.to_numeric(g["return_count"], errors="coerce").fillna(0).tolist()
            rent_hist[st].extend(rc[-14:])  # 최근 14개만
            ret_hist[st].extend(tc[-14:])
    else:
        # (2) 꼬리 파일이 없으면 DB에서 최근 값으로 대체 (없으면 0 부트스트랩)
        try:
            from .Ridge_ml import qs_to_df
            df_hist = qs_to_df()
            df_hist["date"] = pd.to_datetime(df_hist["date"], errors="coerce")
            hist = df_hist[(df_hist["location_name"].isin(stations)) & (df_hist["date"] < first_day)].copy()
            if not hist.empty:
                last_date = hist["date"].max()
                cutoff = last_date - pd.Timedelta(days=21)  # 여유 버퍼
                hist = hist[hist["date"] >= cutoff]
                for st, g in hist.sort_values(["location_name","date"]).groupby("location_name"):
                    rc = pd.to_numeric(g.get("rental_count", pd.Series([], dtype=float)), errors="coerce").fillna(0).tolist()
                    tc = pd.to_numeric(g.get("return_count", pd.Series([], dtype=float)), errors="coerce").fillna(0).tolist()
                    rent_hist[st].extend(rc[-14:])
                    ret_hist[st].extend(tc[-14:])
        except Exception:
            # (3) DB도 못 읽으면 0으로 시작 (첫날만 비슷할 수 있음)
            pass

    # 3-2) 날짜 순으로 진행
    rows = []
    for d, day_df in df_std.sort_values(["date", "location_name"]).groupby("date", sort=True):
        today = day_df.copy()

        # 날씨 결측 보정 (스케일러/인코더 호환)
        for c in ["avg_temp", "daily_rainfall"]:
            if c in today.columns and today[c].isna().any():
                fillv = np.nanmedian(today[c].to_numpy())
                today[c] = today[c].fillna(fillv)

        # 스테이션별 lag/rolling 계산 (히스토리가 부족하면 0 보정)
        lag_frame = []
        for _, r in today.iterrows():
            st = r["location_name"]
            r_hist = rent_hist[st]
            t_hist = ret_hist[st]

            def _lag(seq, k, default=0.0):
                return float(seq[-k]) if len(seq) >= k else float(default)

            def _ma(seq, k, default=0.0):
                if len(seq) == 0:
                    return float(default)
                w = seq[-min(len(seq), k):]
                return float(np.mean(w)) if len(w) > 0 else float(default)

            lag_frame.append({
                "location_name": st,
                "avg_temp": float(r["avg_temp"]),
                "daily_rainfall": float(r["daily_rainfall"]),
                "month": int(r["month"]),
                "dow": int(r["dow"]),
                "rent_lag1": _lag(r_hist, 1, 0.0),
                "rent_lag7": _lag(r_hist, 7, _lag(r_hist, 1, 0.0)),
                "rent_ma7":  _ma(r_hist, 7, _lag(r_hist, 1, 0.0)),
                "rent_ma14": _ma(r_hist, 14, _lag(r_hist, 1, 0.0)),
                "ret_lag1":  _lag(t_hist, 1, 0.0),
                "ret_lag7":  _lag(t_hist, 7, _lag(t_hist, 1, 0.0)),
                "ret_ma7":   _ma(t_hist, 7, _lag(t_hist, 1, 0.0)),
                "ret_ma14":  _ma(t_hist, 14, _lag(t_hist, 1, 0.0)),
            })

        X_today = pd.DataFrame(lag_frame, columns=feats)

        # 예측 (일반 predict; Poisson이면 log→exp를 써야 하지만 파이프라인일 때 kwargs 전달이 제한됨)
        # 모델이 Poisson이면 최종 추정기를 직접 꺼내서 처리하는 게 더 정확.
        def _predict_any(m, X_):
            est = _final_est(m)
            # CatBoostRegressor인지 확인
            if isinstance(est, CatBoostRegressor):
                # Poisson이면 지수 변환
                loss = None
                if hasattr(est, "get_params"):
                    loss = est.get_params().get("loss_function", None)
                if loss and isinstance(loss, str) and ("poisson" in loss.lower()):
                    # 파이프라인이더라도 최종단 직접 호출
                    return np.asarray(est.predict(X_, prediction_type="Exponent"), dtype=float)
            # 일반
            return np.asarray(m.predict(X_), dtype=float)

        yhat_r = _predict_any(rent_model, X_today)
        yhat_t = _predict_any(return_model, X_today)
        yhat_r = np.clip(yhat_r, 0, None)
        yhat_t = np.clip(yhat_t, 0, None)

        # 히스토리 업데이트 → 다음 날짜 lag 계산에 사용
        for (st, yr, yt) in zip(today["location_name"].to_numpy(), yhat_r, yhat_t):
            rent_hist[st].append(float(yr))
            ret_hist[st].append(float(yt))

        # 오늘 결과 저장
        today_out = today.copy()
        today_out["pred_rental_count"] = yhat_r
        today_out["pred_return_count"] = yhat_t
        today_out["pred_net_change"]   = today_out["pred_rental_count"] - today_out["pred_return_count"]
        rows.append(today_out)

    out = pd.concat(rows, ignore_index=True)
    return out
'''
def _unwrap_model(m):
    """joblib 로드 결과가 dict({'model':..., 'loc_categories':...})일 수 있으니 언랩."""
    meta = {}
    if isinstance(m, dict):
        meta = m
        m = m.get("model", m)
    return m, meta

def _final_estimator(m):
    return m.steps[-1][1] if isinstance(m, Pipeline) else m

def _std_input(df: pd.DataFrame) -> pd.DataFrame:
    colmap = {
        "일시":"date","DATE":"date","날짜":"date",
        "대여소명":"location_name","지점명":"location_name",
        "평균기온(°C)":"avg_temp","평균기온":"avg_temp","temp":"avg_temp",
        "일강수량(mm)":"daily_rainfall","강수량":"daily_rainfall","rain":"daily_rainfall",
    }
    z = df.rename(columns=lambda c: colmap.get(str(c).strip(), str(c).strip())).copy()
    need = {"date","location_name","avg_temp","daily_rainfall"}
    miss = need - set(z.columns)
    if miss:
        raise ValueError(f"필수 컬럼 누락: {sorted(miss)}")
    z["date"] = pd.to_datetime(z["date"], errors="coerce")
    if z["date"].isna().any():
        raise ValueError("date 파싱 실패 행 존재")
    z["location_name"] = z["location_name"].astype(str).str.strip()
    z["avg_temp"] = pd.to_numeric(z["avg_temp"], errors="coerce")
    z["daily_rainfall"] = pd.to_numeric(z["daily_rainfall"], errors="coerce")
    z["month"] = z["date"].dt.month
    z["dow"]   = z["date"].dt.weekday
    return z

def _load_hist_seed(stations: List[str], first_day: pd.Timestamp) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    """히스토리 시드(cat_hist_tail.pkl)를 읽어 대여/반납 최근값 제공(없으면 빈 dict)."""
    rent_hist, ret_hist = defaultdict(list), defaultdict(list)
    tail_path = os.path.join(getattr(settings, "BASE_DIR"), "cat_hist_tail.pkl")
    if os.path.exists(tail_path):
        hist_tail = joblib.load(tail_path)
        hist_tail["date"] = pd.to_datetime(hist_tail["date"], errors="coerce")
        seed = (hist_tail[(hist_tail["location_name"].isin(stations)) & (hist_tail["date"] < first_day)]
                .sort_values(["location_name","date"]))
        for st, g in seed.groupby("location_name"):
            rc = pd.to_numeric(g.get("rental_count"), errors="coerce").fillna(0).tolist()
            tc = pd.to_numeric(g.get("return_count"), errors="coerce").fillna(0).tolist()
            rent_hist[st].extend(rc[-14:]); ret_hist[st].extend(tc[-14:])
    return rent_hist, ret_hist

def _lag(seq, k, default=0.0): return float(seq[-k]) if len(seq) >= k else float(default)
def _ma(seq, k, default=0.0):
    if len(seq)==0: return float(default)
    w = seq[-min(len(seq),k):]
    return float(np.mean(w)) if len(w) else float(default)

# ======================= RF 전처리 함수 =======================
RF_RENT_FEATS = [
    "loc_code","month","dow","dow_sin","dow_cos","mon_sin","mon_cos","rain_flag",
    "avg_temp","daily_rainfall",
    "rent_lag1","rent_lag7","rent_rmean7","rent_lag1_x_rain","rent_lag1_x_wknd"
]
RF_RET_FEATS  = [
    "loc_code","month","dow","dow_sin","dow_cos","mon_sin","mon_cos","rain_flag",
    "avg_temp","daily_rainfall",
    "ret_lag1","ret_lag7","ret_rmean7","ret_lag1_x_rain","ret_lag1_x_wknd"
]

def rf_static_fe(df_std: pd.DataFrame, loc_categories: Optional[List[str]]=None) -> Tuple[pd.DataFrame, Dict]:
    """RF 공통(정적) 파생만 생성(카테고리 코딩, 사이클릭, 플래그)."""
    df = df_std.copy()
    # loc_code 매핑
    cats = list(loc_categories or sorted(df["location_name"].unique()))
    code_map = {s:i for i,s in enumerate(cats)}
    df["loc_code"] = df["location_name"].map(lambda s: code_map.get(s, -1)).astype("int32")
    # cyclic & flags
    df["dow_sin"] = np.sin(2*np.pi*df["dow"]/7).astype("float32")
    df["dow_cos"] = np.cos(2*np.pi*df["dow"]/7).astype("float32")
    df["mon_sin"] = np.sin(2*np.pi*df["month"]/12).astype("float32")
    df["mon_cos"] = np.cos(2*np.pi*df["month"]/12).astype("float32")
    df["rain_flag"] = (pd.to_numeric(df["daily_rainfall"], errors="coerce").fillna(0) > 0).astype("int8")
    df["is_weekend"] = (df["dow"] >= 5).astype("int8")
    meta = {"loc_categories": cats}
    return df, meta

def rf_day_design(today_rf: pd.DataFrame,
                  rent_hist: Dict[str,List[float]],
                  ret_hist: Dict[str,List[float]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """하루치 입력에서 RF용 (대여/반납) 피처 행렬 생성."""
    rent_rows, ret_rows = [], []
    for _, r in today_rf.iterrows():
        st = r["location_name"]; rls = rent_hist[st]; rts = ret_hist[st]
        base = {
            "loc_code": int(r["loc_code"]), "month": int(r["month"]), "dow": int(r["dow"]),
            "dow_sin": float(r["dow_sin"]), "dow_cos": float(r["dow_cos"]),
            "mon_sin": float(r["mon_sin"]), "mon_cos": float(r["mon_cos"]),
            "rain_flag": int(r["rain_flag"]),
            "avg_temp": float(r["avg_temp"]), "daily_rainfall": float(r["daily_rainfall"]),
        }
        rent_rows.append({**base,
            "rent_lag1": _lag(rls,1,0.0), "rent_lag7": _lag(rls,7,_lag(rls,1,0.0)),
            "rent_rmean7": _ma(rls,7,_lag(rls,1,0.0)),
            "rent_lag1_x_rain": _lag(rls,1,0.0) * int(r["rain_flag"]>0),
            "rent_lag1_x_wknd": _lag(rls,1,0.0) * int(r["is_weekend"]),
        })
        ret_rows.append({**base,
            "ret_lag1": _lag(rts,1,0.0), "ret_lag7": _lag(rts,7,_lag(rts,1,0.0)),
            "ret_rmean7": _ma(rts,7,_lag(rts,1,0.0)),
            "ret_lag1_x_rain": _lag(rts,1,0.0) * int(r["rain_flag"]>0),
            "ret_lag1_x_wknd": _lag(rts,1,0.0) * int(r["is_weekend"]),
        })
    Xr = pd.DataFrame(rent_rows, columns=RF_RENT_FEATS)
    Xt = pd.DataFrame(ret_rows,  columns=RF_RET_FEATS)
    return Xr, Xt

# ======================= CatBoost 전처리 함수 =======================
CAT_BASE_FEATS = ["location_name","avg_temp","daily_rainfall","month","dow"]
CAT_LAG_COLS   = ["rent_lag1","rent_lag7","rent_ma7","rent_ma14",
                  "ret_lag1","ret_lag7","ret_ma7","ret_ma14"]
CAT_FEATS      = CAT_BASE_FEATS + CAT_LAG_COLS

def cat_day_design(today: pd.DataFrame,
                   rent_hist: Dict[str,List[float]],
                   ret_hist: Dict[str,List[float]]) -> pd.DataFrame:
    """하루치 입력에서 CatBoost용 피처 행렬 생성(카테고리는 문자열 그대로)."""
    rows = []
    for _, r in today.iterrows():
        st = r["location_name"]; rls = rent_hist[st]; rts = ret_hist[st]
        rows.append({
            "location_name": st,
            "avg_temp": float(r["avg_temp"]), "daily_rainfall": float(r["daily_rainfall"]),
            "month": int(r["month"]), "dow": int(r["dow"]),
            "rent_lag1": _lag(rls,1,0.0), "rent_lag7": _lag(rls,7,_lag(rls,1,0.0)),
            "rent_ma7":  _ma(rls,7,_lag(rls,1,0.0)), "rent_ma14": _ma(rls,14,_lag(rls,1,0.0)),
            "ret_lag1":  _lag(rts,1,0.0), "ret_lag7": _lag(rts,7,_lag(rts,1,0.0)),
            "ret_ma7":   _ma(rts,7,_lag(rts,1,0.0)), "ret_ma14": _ma(rts,14,_lag(rts,1,0.0)),
        })
    return pd.DataFrame(rows, columns=CAT_FEATS)

# ======================= 메인: predict_future =======================
def predict_future(df: pd.DataFrame, rent_model_loaded, return_model_loaded):
    """
    업로드 DF를 받아 미래 예측. 내부에서 모델 타입에 맞는 전처리 함수를 호출.
    - RF(dict 저장형 포함) : rf_static_fe + rf_day_design
    - CatBoost           : cat_day_design
    - 일반 파이프라인    : 원본 컬럼 그대로 predict
    """
    df_std = _std_input(df)
    rent_model, rent_meta = _unwrap_model(rent_model_loaded)
    return_model, return_meta = _unwrap_model(return_model_loaded)
    est_r, est_t = _final_estimator(rent_model), _final_estimator(return_model)

    # 타입 판별
    try:
        from catboost import CatBoostRegressor
    except Exception:
        CatBoostRegressor = object  # type: ignore
    is_rf  = isinstance(est_r, RandomForestRegressor) and isinstance(est_t, RandomForestRegressor)
    is_cat = isinstance(est_r, CatBoostRegressor) and isinstance(est_t, CatBoostRegressor)

    # (A) 일반 파이프라인
    if not is_rf and not is_cat:
        X = df_std[["location_name","avg_temp","daily_rainfall","month","dow"]]
        pr = np.asarray(rent_model.predict(X), dtype=float)
        pt = np.asarray(return_model.predict(X), dtype=float)
        out = df_std.copy()
        out["pred_rental_count"] = np.clip(pr,0,None)
        out["pred_return_count"] = np.clip(pt,0,None)
        out["pred_net_change"]   = out["pred_rental_count"] - out["pred_return_count"]
        return out

    # 공통 히스토리 시드
    first_day = df_std["date"].min()
    stations  = df_std["location_name"].unique().tolist()
    rent_hist, ret_hist = _load_hist_seed(stations, first_day)

    rows = []
    # (B) RF
    if is_rf:
        df_rf, meta = rf_static_fe(df_std, rent_meta.get("loc_categories"))
        for d, today in df_rf.sort_values(["date","location_name"]).groupby("date", sort=True):
            Xr, Xt = rf_day_design(today, rent_hist, ret_hist)
            yhat_r = np.asarray(rent_model.predict(Xr.values), dtype=float)
            yhat_t = np.asarray(return_model.predict(Xt.values), dtype=float)
            yhat_r, yhat_t = np.clip(yhat_r,0,None), np.clip(yhat_t,0,None)
            # 히스토리 업데이트
            for (st, yr, yt) in zip(today["location_name"].to_numpy(), yhat_r, yhat_t):
                rent_hist[st].append(float(yr)); ret_hist[st].append(float(yt))
            out_day = today[["date","location_name","avg_temp","daily_rainfall","month","dow"]].copy()
            out_day["pred_rental_count"] = yhat_r
            out_day["pred_return_count"] = yhat_t
            out_day["pred_net_change"]   = out_day["pred_rental_count"] - out_day["pred_return_count"]
            rows.append(out_day)
        return pd.concat(rows, ignore_index=True)

    # (C) CatBoost
    def _predict_cat(m, X_):
        est = _final_estimator(m)
        loss = None
        if hasattr(est, "get_params"):
            loss = est.get_params().get("loss_function", None)
        if isinstance(est, CatBoostRegressor) and isinstance(loss, str) and ("poisson" in loss.lower()):
            return np.asarray(est.predict(X_, prediction_type="Exponent"), dtype=float)
        return np.asarray(m.predict(X_), dtype=float)

    for d, today in df_std.sort_values(["date","location_name"]).groupby("date", sort=True):
        X_today = cat_day_design(today, rent_hist, ret_hist)
        yhat_r = _predict_cat(rent_model, X_today)
        yhat_t = _predict_cat(return_model, X_today)
        yhat_r, yhat_t = np.clip(yhat_r,0,None), np.clip(yhat_t,0,None)
        for (st, yr, yt) in zip(today["location_name"].to_numpy(), yhat_r, yhat_t):
            rent_hist[st].append(float(yr)); ret_hist[st].append(float(yt))
        out_day = today[["date","location_name","avg_temp","daily_rainfall","month","dow"]].copy()
        out_day["pred_rental_count"] = yhat_r
        out_day["pred_return_count"] = yhat_t
        out_day["pred_net_change"]   = out_day["pred_rental_count"] - out_day["pred_return_count"]
        rows.append(out_day)
    return pd.concat(rows, ignore_index=True)