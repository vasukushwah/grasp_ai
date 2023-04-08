from django.views.generic import TemplateView
from .models import FileUpload
from .api.serializers import FileUploadSerializer


class HomeView(TemplateView):
    template_name = "app/index.html"

    def get_context_data(self, **kwargs):
        file_uploads = FileUpload.objects.all()
        serializer = FileUploadSerializer(file_uploads, many=True)

        context = super().get_context_data(**kwargs)
        context["files"] = serializer.data
        print(serializer.data)
        return context
