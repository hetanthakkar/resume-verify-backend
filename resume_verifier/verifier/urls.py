from django.urls import path
from .views import ProjectVerificationView

urlpatterns = [
    path("verify/", ProjectVerificationView.as_view(), name="verify_projects"),
]
