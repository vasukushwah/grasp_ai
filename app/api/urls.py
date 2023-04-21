from django.urls import include, path
from .views import (
    FilesView,
    FileEmbedingView,
    ask_question_view,
)

urlpatterns = [
    path("files/", FilesView.as_view(), name="files"),
    path("files/<int:id>", FilesView.as_view(), name="file_delete"),
    path("file/embed/<int:id>", FileEmbedingView.as_view(), name="embed_file"),
    path("ask/", ask_question_view, name="ask_question"),
]
