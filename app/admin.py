from django.contrib import admin
from .models import FileUpload

# Register your models here.
@admin.register(FileUpload)
class FileUploadAdmin(admin.ModelAdmin):
    list_display = ("id", "file", "created_at")
    list_filter = ("created_at",)
    search_fields = ("file__name",)
