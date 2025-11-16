from django import forms
from .services.registry import available_models

class TrainPredictForm(forms.Form):
    alpha = forms.FloatField(
        label="Ridge α",
        min_value=0.0, initial=1.0, help_text="정규화 강도 (0이면 OLS에 가까움)"
    )
    drop_leak = forms.BooleanField(
        label="누수 방지 (rental/return 제외)",
        required=False, initial=False,
        help_text="net_change = return - rental 구조면 체크 권장"
    )


class ModelSelectForm(forms.Form):
    model = forms.ChoiceField(
        label="머신러닝 모델",
        choices=[(k, v["label"]) for k, v in available_models().items()],
        initial=list(available_models().keys())[0],
    )
    alpha = forms.FloatField(
        label="Ridge α",
        min_value=0.0, initial=1.0, help_text="정규화 강도(모델에 따라 무시될 수 있음)"
    )
    drop_leak = forms.BooleanField(
        label="누수 방지", required=False, initial=True,
        help_text="타깃(대여/반납) 자기자신 카운트를 입력 피처에서 제외"
    )

# 미래 예측용 CSV 업로드
class ForecastUploadForm(forms.Form):
    file = forms.FileField(label="미래 데이터 CSV 업로드")
