from rest_framework import serializers
from app.models import FileUpload
import os


class FileUploadSerializer(serializers.ModelSerializer):
    name = serializers.SerializerMethodField(read_only=True)

    class Meta:
        model = FileUpload
        fields = ("id", "file", "name")

    def get_name(self, obj):
        return os.path.basename(obj.file.name)
