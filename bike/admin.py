from django.contrib import admin
from import_export import resources
from import_export.admin import ImportExportModelAdmin
from .models import BikeRental

# CSV 컬럼과 모델 필드를 매핑하는 Resource 클래스
class BikeRentalResource(resources.ModelResource):
    class Meta:
        model = BikeRental
        # CSV 파일의 컬럼명과 모델 필드명을 매핑
        fields = ('id', 'date', 'location_name', 'rental_count', 'return_count', 'net_change', 'avg_temp', 'daily_rainfall',)
        # CSV 파일의 헤더 순서와 맞추기 위해 import_id_fields를 사용.
        # 이 필드는 유니크한 값으로 사용되지 않으므로, 그냥 첫번째 필드로 지정해줘도 돼.
        import_id_fields = ('id',)

# 관리자 페이지에 등록할 모델
@admin.register(BikeRental)
class BikeRentalAdmin(ImportExportModelAdmin):
    resource_class = BikeRentalResource
    list_display = ('id', 'date', 'location_name', 'rental_count', 'return_count', 'net_change')