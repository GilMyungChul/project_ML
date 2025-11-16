import os
from typing import List, Tuple, Dict, Optional
import pandas as pd
import numpy as np
import joblib
from django.conf import settings
from django.db.models import QuerySet

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin

# ===== 공통 설정 =====
REQUIRED_COLS = [
    "date","location_name","rental_count","return_count",
    "net_change","avg_temp","daily_rainfall"
]
BASE_NUM = ["rental_count","return_count","avg_temp","daily_rainfall"]
CAT_COLS = ["location_name","month","dow"]
TARGET = "net_change"


# ===== 유틸: QuerySet -> DataFrame =====
def qs_to_df(qs: QuerySet) -> pd.DataFrame:
    df = pd.DataFrame.from_records(qs.values(*REQUIRED_COLS)).copy()
    if df.empty:
        raise ValueError("DB에 학습/예측할 데이터가 없어.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["month"] = df["date"].dt.month
    df["dow"] = df["date"].dt.dayofweek
    df["location_name"] = df["location_name"].astype(str)
    for c in ["rental_count","return_count","avg_temp","daily_rainfall","net_change"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if df.isna().any().any():
        raise ValueError("파싱 후 결측치가 있습니다. 데이터를 확인해 주세요.")
    return df


# ===== 공선성 완화용: 인덱스 드랍 변환기 =====
class ColDropper(BaseEstimator, TransformerMixin):
    def __init__(self, drop_idx: Optional[List[int]] = None):
        self.drop_idx = set(drop_idx or [])
        self.keep_idx_: Optional[np.ndarray] = None

    def fit(self, X, y=None):
        if len(self.drop_idx) == 0:
            self.keep_idx_ = None
        else:
            self.keep_idx_ = np.array([i for i in range(X.shape[1]) if i not in self.drop_idx], dtype=int)
        return self

    def transform(self, X):
        if self.keep_idx_ is None:
            return X
        return X[:, self.keep_idx_]


# ===== 공선성 진단/완화 헬퍼들 =====
def _onehot(drop_first: bool) -> OneHotEncoder:
    # drop='first'로 더미 트랩(완전 공선성) 예방, 미지 범주 무시
    return OneHotEncoder(handle_unknown="ignore", drop="first" if drop_first else None, sparse_output=False)

def _build_preprocessor(features: List[str], drop_first_ohe: bool, num_for_scale: List[str]) -> ColumnTransformer:
    num_in_feat = [c for c in num_for_scale if c in features]
    cat_in_feat = [c for c in CAT_COLS if c in features]
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), num_in_feat),
            ("cat", _onehot(drop_first_ohe), cat_in_feat),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return pre

def _fit_pre_and_get_names(pre: ColumnTransformer, X_tr: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    # 전처리 적합 및 변환 + 피처 이름 추출
    Xtr_pre = pre.fit_transform(X_tr)
    # sklearn 1.0+에서 지원
    try:
        names = pre.get_feature_names_out(X_tr.columns).tolist()
    except Exception:
        # 수동 조립 (숫자 이름 + OHE 이름)
        names = []
        if "num" in pre.named_transformers_ and pre.named_transformers_["num"] != "drop":
            names += [c for c in pre.transformers_[0][2]]  # num 컬럼 그대로
        if "cat" in pre.named_transformers_ and pre.named_transformers_["cat"] != "drop":
            ohe = pre.named_transformers_["cat"]
            cat_cols_used = pre.transformers_[1][2]
            if hasattr(ohe, "get_feature_names_out"):
                names += ohe.get_feature_names_out(cat_cols_used).tolist()
            else:
                # 구버전 호환 - 대략적 이름
                for col, cats in zip(cat_cols_used, getattr(ohe, "categories_", [[]])):
                    drop_first = getattr(ohe, "drop", None) in ("first", "if_binary")
                    start = 1 if drop_first else 0
                    names += [f"{col}_{str(cat)}" for cat in cats[start:]]
    return pre, Xtr_pre, names

def _high_corr_drop_idx(X: np.ndarray, thresh: float = 0.98) -> List[int]:
    """절대 상관계수 >= thresh인 컬럼을 탐욕적으로 제거할 인덱스 선택."""
    if thresh is None or thresh <= 0 or thresh >= 1:
        return []
    corr = np.corrcoef(X, rowvar=False)
    n = corr.shape[0]
    drop = set()
    for i in range(n):
        if i in drop: 
            continue
        for j in range(i + 1, n):
            if j in drop:
                continue
            c = corr[i, j]
            if np.isnan(c):
                continue
            if abs(c) >= thresh:
                drop.add(j)  # 후행 컬럼 드랍(간단 규칙)
    return sorted(drop)

def _compute_vif_table(X: np.ndarray, names: List[str], max_cols: int = 200):
    """statsmodels가 있으면 VIF 계산(열 많으면 상위 max_cols개만). 실패하면 None."""
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
    except Exception:
        return None
    X_np = np.asarray(X, dtype=float)
    # 너무 큰 디자인매트릭스는 샘플링
    if X_np.shape[1] > max_cols:
        idx = np.arange(X_np.shape[1])
        # 열 분산 기준 상위 max_cols 유지 (정보량 보존 느낌)
        var = X_np.var(axis=0)
        keep = idx[np.argsort(-var)[:max_cols]]
        X_np = X_np[:, keep]
        names = [names[i] for i in keep]
    vif_rows = []
    for i in range(X_np.shape[1]):
        try:
            v = float(variance_inflation_factor(X_np, i))
        except Exception:
            v = float("inf")
        vif_rows.append({"feature": names[i], "vif": v})
    vif_rows.sort(key=lambda d: d["vif"], reverse=True)
    return vif_rows


# ===== 기존 빌더 확장: drop_first_ohe / pca / corr 드랍 옵션 =====
def build_pipeline(
    alpha: float,
    drop_leak: bool,
    drop_first_ohe: bool = True,
    corr_drop_thresh: Optional[float] = None,
    pca_n_components: Optional[int] = None,
) -> Tuple[Pipeline, List[str]]:
    # 피처 구성
    features: List[str] = ["location_name","avg_temp","daily_rainfall","month","dow"]
    if not drop_leak:
        features = ["location_name","rental_count","return_count","avg_temp","daily_rainfall","month","dow"]

    # 전처리 1차 빌드 & 임시 적합으로 상관 기반 드랍 인덱스 계산
    num_for_scale = BASE_NUM[:]  # scale 대상 후보
    if drop_leak:
        # 누수 방지 시 학습 타깃과 쌍인 수치 제외
        for c in ["rental_count","return_count"]:
            if c in num_for_scale:
                num_for_scale.remove(c)

    pre = _build_preprocessor(features, drop_first_ohe, num_for_scale)
    pre_fitted, Xtr_pre_dummy, names = pre, None, None  # build_pipeline 단독 호출 시 학습 데이터가 없어 드랍계산은 나중에

    drop_idx = []  # 여기선 드랍 미결정(실제 학습 함수 내부에서 계산)
    steps = [("pre", pre), ("drop", ColDropper(drop_idx))]
    if pca_n_components and pca_n_components > 0:
        steps.append(("pca", PCA(n_components=pca_n_components, random_state=42)))
    steps.append(("model", Ridge(alpha=alpha)))
    pipe = Pipeline(steps)
    return pipe, features


# ====== (핵심) 대여/반납 각각 예측 + 공선성 진단/완화 옵션 ======
def train_eval_predict_dual(
    df: pd.DataFrame,
    alpha: float,
    drop_leak: bool,
    *,
    drop_first_ohe: bool = True,          # 더미 트랩 방지
    corr_drop_thresh: Optional[float] = 0.98,  # 상관 절대값 임계(예: 0.97~0.995 권장). None이면 사용 안 함
    pca_n_components: Optional[int] = None,    # PCA로 직교화(원하면 지정)
    compute_vif: bool = False,            # VIF 리포트 생성(통계모듈 필요)
) -> Tuple[Dict, str, Tuple[str, str], Dict]:
    """
    출력 추가:
      - diag: {"rent": {...}, "return": {...}}  # 공선성 진단 정보(VIF 상위, 드랍 열 등)
    """

    # 분할
    tr = df[df["date"].dt.year == 2023].copy()
    te = df[df["date"].dt.year == 2024].copy()
    if tr.empty or te.empty:
        raise ValueError("2023/2024 데이터가 모두 필요해.")

    # 공통: 카테고리/수치 후보
    cat_cols_common = ["location_name","month","dow"]

    # === (1) 대여건수 모델 구성 ===
    if not drop_leak:
        features_rent = ["location_name","month","dow","avg_temp","daily_rainfall","return_count"]
        num_for_scale_r = ["avg_temp","daily_rainfall","return_count"]
    else:
        features_rent = ["location_name","month","dow","avg_temp","daily_rainfall"]
        num_for_scale_r = ["avg_temp","daily_rainfall"]

    X_tr_r = tr[features_rent].copy()
    X_te_r = te[features_rent].copy()
    y_tr_r = tr["rental_count"].values
    y_te_r = te["rental_count"].values

    pre_r = _build_preprocessor(features_rent, drop_first_ohe, num_for_scale_r)
    pre_r, Xtr_r_pre, names_r = _fit_pre_and_get_names(pre_r, X_tr_r)

    # 상관기반 드랍 인덱스 계산(학습데이터 기준)
    drop_idx_r = _high_corr_drop_idx(Xtr_r_pre, thresh=corr_drop_thresh or 0.0)

    # VIF (옵션)
    vif_r = _compute_vif_table(Xtr_r_pre, names_r) if compute_vif else None

    # 최종 파이프라인 조립
    steps_r = [("pre", pre_r), ("drop", ColDropper(drop_idx_r))]
    if pca_n_components and pca_n_components > 0:
        steps_r.append(("pca", PCA(n_components=pca_n_components, random_state=42)))
    steps_r.append(("model", Ridge(alpha=alpha, random_state=42)))
    pipe_rent = Pipeline(steps_r)

    # === (2) 반납건수 모델 구성 ===
    if not drop_leak:
        features_ret = ["location_name","month","dow","avg_temp","daily_rainfall","rental_count"]
        num_for_scale_t = ["avg_temp","daily_rainfall","rental_count"]
    else:
        features_ret = ["location_name","month","dow","avg_temp","daily_rainfall"]
        num_for_scale_t = ["avg_temp","daily_rainfall"]

    X_tr_t = tr[features_ret].copy()
    X_te_t = te[features_ret].copy()
    y_tr_t = tr["return_count"].values
    y_te_t = te["return_count"].values

    pre_t = _build_preprocessor(features_ret, drop_first_ohe, num_for_scale_t)
    pre_t, Xtr_t_pre, names_t = _fit_pre_and_get_names(pre_t, X_tr_t)

    drop_idx_t = _high_corr_drop_idx(Xtr_t_pre, thresh=corr_drop_thresh or 0.0)
    vif_t = _compute_vif_table(Xtr_t_pre, names_t) if compute_vif else None

    steps_t = [("pre", pre_t), ("drop", ColDropper(drop_idx_t))]
    if pca_n_components and pca_n_components > 0:
        steps_t.append(("pca", PCA(n_components=pca_n_components, random_state=42)))
    steps_t.append(("model", Ridge(alpha=alpha, random_state=42)))
    pipe_return = Pipeline(steps_t)

    # ===== 학습 =====
    pipe_rent.fit(X_tr_r, y_tr_r)
    pipe_return.fit(X_tr_t, y_tr_t)

    # ===== 예측 =====
    tr_pred_r = pipe_rent.predict(X_tr_r)
    te_pred_r = pipe_rent.predict(X_te_r)
    tr_pred_t = pipe_return.predict(X_tr_t)
    te_pred_t = pipe_return.predict(X_te_t)
    te_pred_change = te_pred_r - te_pred_t

    # ===== 지표(주의: 기존 코드 호환 위해 'rmse' 키지만 값은 MSE임) =====
    metrics = {
        "rent": {
            "rmse_tr": float(mean_squared_error(y_tr_r, tr_pred_r)),
            "mae_tr":  float(mean_absolute_error(y_tr_r, tr_pred_r)),
            "r2_tr":   float(r2_score(y_tr_r, tr_pred_r)),
            "rmse_te": float(mean_squared_error(y_te_r, te_pred_r)),
            "mae_te":  float(mean_absolute_error(y_te_r, te_pred_r)),
            "r2_te":   float(r2_score(y_te_r, te_pred_r)),
            "features": features_rent,
        },
        "return": {
            "rmse_tr": float(mean_squared_error(y_tr_t, tr_pred_t)),
            "mae_tr":  float(mean_absolute_error(y_tr_t, tr_pred_t)),
            "r2_tr":   float(r2_score(y_tr_t, tr_pred_t)),
            "rmse_te": float(mean_squared_error(y_te_t, te_pred_t)),
            "mae_te":  float(mean_absolute_error(y_te_t, te_pred_t)),
            "r2_te":   float(r2_score(y_te_t, te_pred_t)),
            "features": features_ret,
        },
    }

    # ===== 결과 CSV =====
    os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
    out = te[["date","location_name","rental_count","return_count"]].copy()
    if "net_change" in te.columns:
        out["net_change"] = te["net_change"].values
    out["pred_rental_count"]  = te_pred_r
    out["pred_return_count"]  = te_pred_t
    out["pred_net_change"]    = te_pred_change
    out_csv = os.path.join(settings.MEDIA_ROOT, "pred_dual_2024.csv")
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # ===== 모델 저장 =====
    rent_model_path   = os.path.join(getattr(settings, "BASE_DIR"), "model_rent_ridge.pkl")
    return_model_path = os.path.join(getattr(settings, "BASE_DIR"), "model_return_ridge.pkl")
    joblib.dump(pipe_rent, rent_model_path)
    joblib.dump(pipe_return, return_model_path)

    # ===== 공선성 진단 리포트 =====
    def _after_names(pre: ColumnTransformer, drop_idx: List[int]) -> List[str]:
        try:
            names_all = pre.get_feature_names_out()
        except Exception:
            # 간단 대체: 갯수만큼 f0..fn
            n_out = pre.transform(pd.DataFrame({c: [] for c in pre.feature_names_in_})).shape[1]
            names_all = [f"f{i}" for i in range(n_out)]
        keep = [i for i in range(len(names_all)) if i not in (drop_idx or [])]
        return [names_all[i] for i in keep]

    diag = {
        "rent": {
            "corr_thresh": corr_drop_thresh,
            "n_before": int(Xtr_r_pre.shape[1]),
            "n_dropped": int(len(drop_idx_r)),
            "feature_names_after": _after_names(pre_r, drop_idx_r),
            "vif_top": (vif_r[:20] if isinstance(vif_r, list) else None),
        },
        "return": {
            "corr_thresh": corr_drop_thresh,
            "n_before": int(Xtr_t_pre.shape[1]),
            "n_dropped": int(len(drop_idx_t)),
            "feature_names_after": _after_names(pre_t, drop_idx_t),
            "vif_top": (vif_t[:20] if isinstance(vif_t, list) else None),
        },
        "options": {
            "drop_first_ohe": drop_first_ohe,
            "pca_n_components": pca_n_components,
            "compute_vif": compute_vif,
        }
    }

    return metrics, out_csv, (rent_model_path, return_model_path), diag
