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
    VerifyOTPView,
    CheckEmailRegisteredView,
    RefreshTokenView,
    JobAnalysisView,
    RecentAnalysisView,
    ResumeDetailView,
    ForgotPasswordView,
    ResetPasswordView,
    RecruiterProfileUpdateView,
    RecruiterEmailUpdateRequestView,
    RecruiterEmailUpdateConfirmView,
    LogoutView,
)

from rest_framework_simplejwt.views import TokenRefreshView

router = DefaultRouter()
# Only register the JobViewSet - ShortlistViewSet will be included in its URLs
router.register(r"jobs", JobViewSet, basename="job")
# Create explicit URL patterns for shortlist operations
shortlist_actions = ShortlistViewSet.as_view(
    {
        "delete": "remove_from_shortlist",
        "post": "add_to_shortlist",
    }
)
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
        "resume-analysis/",
        CombinedVerificationView.as_view(),
        name="combined-verification",
    ),
    path("recent-analyses/", RecentAnalysisView.as_view()),
    path("job-analyses/<int:job_id>/", JobAnalysisView.as_view()),
    path("resumes/<int:id>/", ResumeDetailView.as_view()),
    path("auth/check-email/", CheckEmailRegisteredView.as_view(), name="register"),
    path("auth/register/", RegisterView.as_view(), name="register"),
    path("auth/verify-otp/", VerifyOTPView.as_view(), name="verify-otp"),
    path("auth/login/", LoginView.as_view(), name="login"),
    path("auth/me/", GetProfileView.as_view(), name="me"),
    path("auth/refresh/", TokenRefreshView.as_view(), name="token_refresh"),
    path("", include(router.urls)),
    path(
        "candidates/<int:id>/", CandidateDetailView.as_view(), name="candidate-detail"
    ),
    path(
        "candidates/<int:pk>/resumes",
        CandidateResumesView.as_view(),
        name="candidate-resumes",
    ),
    path(
        "jobs/<int:pk>/shortlist/<str:resume_id>/",
        shortlist_actions,
        name="shortlist-actions",
    ),
    path("jobs/<int:pk>/shortlisted/", shortlist_list, name="shortlist-list"),
    path(
        "auth/refresh/", TokenRefreshView.as_view(), name="token_refresh"
    ),  # Default JWT refresh view
    path(
        "auth/refresh-token/", RefreshTokenView.as_view(), name="custom_refresh"
    ),  # Custom refresh token view
    path("forgot-password/", ForgotPasswordView.as_view(), name="forgot-password"),
    path("reset-password/", ResetPasswordView.as_view(), name="reset-password"),
    path(
        "profile/update/",
        RecruiterProfileUpdateView.as_view(),
        name="recruiter-profile-update",
    ),
    path(
        "profile/update-email/request/",
        RecruiterEmailUpdateRequestView.as_view(),
        name="recruiter-email-update-request",
    ),
    path(
        "profile/update-email/confirm/",
        RecruiterEmailUpdateConfirmView.as_view(),
        name="recruiter-email-update-confirm",
    ),
    path("auth/logout/", LogoutView.as_view(), name="auth_logout"),
]
