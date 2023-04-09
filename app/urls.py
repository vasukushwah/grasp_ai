from django.urls import include, path
from . import views

urlpatterns = [
    path("", views.home_view),
    path("api/", include("app.api.urls")),
]
