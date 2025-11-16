import io
from django.shortcuts import render, redirect
from django.core.paginator import Paginator
import os, pandas as pd
from django.urls import reverse
from django.conf import settings
from django import forms
from .forms import TrainPredictForm, ModelSelectForm, ForecastUploadForm
from .services.Ridge_ml import qs_to_df, train_eval_predict_dual
from .model_holder import MlModelHolder
from .models import BikeRental
import matplotlib
import matplotlib.pyplot as plt
from django.http import Http404, HttpResponse
from .services.registry import get_model_spec, available_models  # 모델바꿔서 사용하는 용도??
from .services.forecast_utils import load_future_csv_to_df, load_models_for_key, predict_future  # 미래 예측
import time


plt.rcParams["font.family"] = "Malgun Gothic"
matplotlib.use("Agg")  # 화면 없는 서버에서 렌더

PRED_NAME = "pred_2024.csv"

def _pred_csv_path() -> str:
    """
    MEDIA_ROOT 안의 pred_2024.csv 전체 경로 반환
    """
    return os.path.join(settings.MEDIA_ROOT, PRED_NAME)

def ml_home(request):
    model_key = request.GET.get("model")
    model_label = None
    if model_key:
        try:
            model_label = available_models()[model_key]["label"]
        except KeyError:
            model_key = None

    form = TrainPredictForm()

    search_txt = request.GET.get("loc")
    csv_path = os.path.join(settings.MEDIA_ROOT, "pred_2024.csv")
    ctx_rows, total_rows, pred_csv_url = [], 0, None

    show_result = False

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)

        if "loc" in request.GET:  # 검색 조건이 들어왔을 때만 보여줌
            show_result = True
            if search_txt:
                df = df[df["location_name"].str.contains(search_txt, na=False)]

            ctx_rows = df.head(100).to_dict(orient="records")
            total_rows = len(df)
            pred_csv_url = settings.MEDIA_URL + "pred_2024.csv"
        else:
            ctx_rows = []
    else:
        ctx_rows = []
        total_rows = 0
        pred_csv_url = None

    ctx = {
        "form": form,
        "model_key": model_key,
        "model_label": model_label,
        "rows": ctx_rows,
        "total_rows": total_rows,
        "pred_csv_url": pred_csv_url,
        "show_result": show_result,
    }

    # AJAX 요청이면 리스트 부분만 반환
    if request.headers.get("x-requested-with") == "XMLHttpRequest":
        return render(request, "bike/_predict_table.html", ctx)

    return render(request, "bike/model_predict.html", ctx)


def ml_train_predict(request):
    if request.method != "POST":
        # 직접 진입이면 모델 선택 화면으로
        return redirect("bike:ml_home")

    form = TrainPredictForm(request.POST)
    model_key = request.POST.get("model")  # hidden으로 넘어오는 값

    if not model_key:
        # 모델이 지정되지 않았으면 선택 페이지로
        return redirect("bike:ml_select")

    if not form.is_valid():
        # 에러 시 다시 결과 페이지로 (선택 모델 유지)
        return render(request, "bike/model_predict.html", {
            "form": form,
            "model_key": model_key,
            "model_label": available_models().get(model_key, {}).get("label"),
            "error": "폼 유효성 에러",
        })

    alpha = form.cleaned_data["alpha"]
    drop_leak = form.cleaned_data["drop_leak"]

    try:
        spec = get_model_spec(model_key)
        train_fn = spec["train"]
        result_template = spec.get("template", "bike/model_predict.html")

        # DB -> DataFrame
        df = qs_to_df(BikeRental.objects.all())

        # 학습/예측
        metrics, out_csv, model_paths = train_fn(df, alpha=alpha, drop_leak=drop_leak)

        # 모델 로드 및 이름 표시
        if isinstance(model_paths, (list, tuple)):
            saved_model_str = ", ".join(os.path.basename(p) for p in model_paths)
            if len(model_paths) >= 2:
                rent_path, return_path = model_paths[:2]
                MlModelHolder.force_load(rent_path, return_path)
            else:
                MlModelHolder.force_load(model_paths[0])
        else:
            saved_model_str = os.path.basename(model_paths)
            MlModelHolder.force_load(model_paths)

        # 결과 미리보기
        df_out = pd.read_csv(out_csv)

        out_csv = os.path.join(settings.MEDIA_ROOT, f"pred_2024.csv")
        df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")

        show_result = True

        ctx = {
            "form": form,  # 값 유지
            "model_key": model_key,
            "model_label": available_models()[model_key]["label"],
            "metrics": metrics,
            "rows": df_out.head(100).to_dict(orient="records"),
            "pred_csv_url": settings.MEDIA_URL + os.path.basename(out_csv),
            "total_rows": len(df_out),
            "saved_model": saved_model_str,
            "show_result": show_result,
        }
        return render(request, result_template, ctx)

    except Exception as e:
        return render(request, "bike/model_predict.html", {
            "form": form,
            "model_key": model_key,
            "model_label": available_models().get(model_key, {}).get("label"),
            "error": f"에러: {e}",
        })


def ml_select(request):  # 모델 선택 페이지 뷰
    if request.method == "GET":
        form = ModelSelectForm()
        return render(request, "bike/model_select.html", {"form": form})
    # (POST로 오는 경우가 있어도) 선택만 하는 페이지이므로 ml_home으로 보냄
    model_key = request.POST.get("model") or request.GET.get("model")
    url = f"{reverse('bike:ml_home')}?model={model_key}" if model_key else reverse('bike:ml_home')
    return redirect(url)




# =============미래예측용================
# (A) 모델 선택 (미래 예측용)
def forecast_select(request):
    # model_select와 거의 동일하되, action만 forecast_home으로
    form = ModelSelectForm()
    return render(request, "bike/forecast_select.html", {"form": form})

# (B) 업로드 & 예측 실행 페이지
def forecast_home(request):
    if request.method == "GET":
        model_key = request.GET.get("model")
        model_label = None
        if model_key:
            model_label = available_models().get(model_key, {}).get("label")
        # 업로드 폼만 보여줌
        return render(request, "bike/forecast_predict.html", {
            "model_key": model_key,
            "model_label": model_label,
            "form": ForecastUploadForm(),
        })

    # POST: 파일 업로드 처리 + 예측
    model_key = request.POST.get("model")
    if not model_key:
        return redirect("bike:forecast_select")

    form = ForecastUploadForm(request.POST, request.FILES)
    if not form.is_valid():
        return render(request, "bike/forecast_predict.html", {
            "model_key": model_key,
            "model_label": available_models().get(model_key, {}).get("label"),
            "form": form, "error": "업로드 폼 유효성 에러",
        })

    try:
        # 1) 업로드 저장
        os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
        up = form.cleaned_data["file"]
        # ts = int(time.time())
        up_path = os.path.join(settings.MEDIA_ROOT, f"future.csv")
        # up_path = os.path.join(settings.MEDIA_ROOT, f"future_{ts}.csv")
        with open(up_path, "wb") as f:
            for chunk in up.chunks():
                f.write(chunk)

        # 2) CSV→DF 전처리
        df_future = load_future_csv_to_df(up_path)

        # 3) 모델 로드(저장된 pkl)
        rent_model, return_model, (rent_path, return_path) = load_models_for_key(model_key)

        # 4) 예측
        df_pred = predict_future(df_future, rent_model, return_model)

        # 5) 결과 CSV 저장
        out_csv = os.path.join(settings.MEDIA_ROOT, f"forecast_pred.csv")
        # out_csv = os.path.join(settings.MEDIA_ROOT, f"forecast_pred_{ts}.csv")
        df_pred.to_csv(out_csv, index=False, encoding="utf-8-sig")

        # 6) 결과 렌더 (지표 없음, 표 + 다운로드)
        return render(request, "bike/forecast_predict.html", {
            "model_key": model_key,
            "model_label": available_models().get(model_key, {}).get("label"),
            "form": ForecastUploadForm(),
            "rows": df_pred.head(100).to_dict(orient="records"),
            "total_rows": len(df_pred),
            "pred_csv_url": settings.MEDIA_URL + os.path.basename(out_csv),
            "saved_model": f"{os.path.basename(rent_path)}, {os.path.basename(return_path)}",
        })

    except Exception as e:
        return render(request, "bike/forecast_predict.html", {
            "model_key": model_key,
            "model_label": available_models().get(model_key, {}).get("label"),
            "form": ForecastUploadForm(),
            "error": f"에러: {e}",
        })