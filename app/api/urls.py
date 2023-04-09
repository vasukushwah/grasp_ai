from django.urls import include, path
from .views import (
    FilesView,
    FileEmbedingView,
    AskQuestionView,
)

urlpatterns = [
    path("files/", FilesView.as_view(), name="files"),
    path("files/<int:id>", FilesView.as_view(), name="file_delete"),
    path("embed/<int:id>", FileEmbedingView.as_view(), name="embed_file"),
    path("ask/", AskQuestionView.as_view(), name="ask_question"),
]
