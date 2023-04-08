from django.urls import include, path
from .views import HomeView

urlpatterns = [
    path("", HomeView.as_view()),
    path("pdf/", include("app.api.urls")),
]
