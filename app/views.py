from .models import FileUpload
from .api.serializers import FileUploadSerializer
from django.shortcuts import render


def home_view(request):
    file_uploads = FileUpload.objects.all()
    serializer = FileUploadSerializer(file_uploads, many=True)
    context = {"files": serializer.data}
    return render(request, "app/index.html", context)
