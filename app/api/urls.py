from django.urls import include, path
from .views import FileUploadView, UploadedFilesView, DeleteFileView, FileEmbedingView,AskQuestionView

urlpatterns = [
    path("upload/", FileUploadView.as_view(), name="file_upload"),
    path("uploaded/", UploadedFilesView.as_view(), name="uploaded_files"),
    path("delete/<int:id>", DeleteFileView.as_view(), name="delete_file"),
    path("embed/<int:id>", FileEmbedingView.as_view(), name="embed_file"),
    path("ask/",AskQuestionView.as_view(),name="ask_question")
]
