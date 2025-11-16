from django.apps import AppConfig


class BikeConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "bike"

    def ready(self):
        from .model_holder import MlModelHolder
        MlModelHolder.try_lazy_load()
