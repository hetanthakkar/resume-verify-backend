from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    ProjectVerificationView,
    ProfileMatchView,
    CombinedVerificationView,
    LoginView,
    RegisterView,
    GetProfileView,
    JobViewSet,
    ShortlistViewSet,
    CandidateDetailView,
    CandidateResumesView,
)

from rest_framework_simplejwt.views import TokenRefreshView

router = DefaultRouter()
# Only register the JobViewSet - ShortlistViewSet will be included in its URLs
router.register(r"jobs", JobViewSet, basename="job")
# Create explicit URL patterns for shortlist operations
shortlist_remove = ShortlistViewSet.as_view(
    {
        "delete": "remove_from_shortlist",
    }
)

shortlist_add = ShortlistViewSet.as_view(
    {
        "post": "add_to_shortlist",
    }
)

shortlist_list = ShortlistViewSet.as_view(
    {
        "get": "get_shortlisted",
    }
)

urlpatterns = [
    path("verify/", ProjectVerificationView.as_view(), name="verify_projects"),
    path("profile-match/", ProfileMatchView.as_view(), name="profile-match"),
    path(
        "combined-verify/",
        CombinedVerificationView.as_view(),
        name="combined-verification",
    ),
    path("auth/register/", RegisterView.as_view(), name="register"),
    path("auth/login/", LoginView.as_view(), name="login"),
    path("auth/me/", GetProfileView.as_view(), name="me"),
    path("auth/refresh/", TokenRefreshView.as_view(), name="token_refresh"),
    path("", include(router.urls)),
    path("candidates/<int:pk>", CandidateDetailView.as_view(), name="candidate-detail"),
    path(
        "candidates/<int:pk>/resumes",
        CandidateResumesView.as_view(),
        name="candidate-resumes",
    ),
    path(
        "jobs/<int:pk>/shortlist/<str:resume_id>/",
        shortlist_remove,
        name="shortlist-remove",
    ),
    path(
        "jobs/<int:pk>/shortlist/<str:resume_id>/", shortlist_add, name="shortlist-add"
    ),
    path("jobs/<int:pk>/shortlisted/", shortlist_list, name="shortlist-list"),
]
