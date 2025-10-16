from django.contrib import admin
from .models import TripDataUpload

@admin.register(TripDataUpload)
class TripDataUploadAdmin(admin.ModelAdmin):
    list_display = ('name', 'file_type', 'uploaded_at')
    list_filter = ('file_type',)
    readonly_fields = ('uploaded_at',)
    search_fields = ('name',)
