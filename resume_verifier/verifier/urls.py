from django.urls import path
from .views import ProjectVerificationView, ProfileMatchView, CombinedVerificationView

urlpatterns = [
    path("verify/", ProjectVerificationView.as_view(), name="verify_projects"),
    path("profile-match/", ProfileMatchView.as_view(), name="profile-match"),
    path(
        "combined-verify/",
        CombinedVerificationView.as_view(),
        name="combined-verification",
    ),
]
