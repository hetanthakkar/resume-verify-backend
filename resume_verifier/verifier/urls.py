from django.urls import path
from .views import (
    RegisterView,
    CheckEmailRegisteredView,
    RefreshTokenView,
    VerifyOTPView,
    LoginView,
    GetProfileView,
    ForgotPasswordView,
    ResetPasswordView,
    UpdateProfileView,
    LogoutView,
    GoogleAuthView,
    ProblemView,
    MatchView,
    ChatListView,
    ChatView,
    RequestEmailUpdateView,
    ConfirmEmailUpdateView,
)

shortlist_check = ShortlistViewSet.as_view(
    {
        "get": "check_shortlisted",
    }
)

urlpatterns = [
    # Authentication endpoints
    path("auth/register/", RegisterView.as_view(), name="register"),
    path("auth/check-email/", CheckEmailRegisteredView.as_view(), name="check-email"),
    path("auth/refresh-token/", RefreshTokenView.as_view(), name="refresh-token"),
    path("auth/verify-otp/", VerifyOTPView.as_view(), name="verify-otp"),
    path("auth/login/", LoginView.as_view(), name="login"),
<<<<<<< HEAD
    path("auth/forgot-password/", ForgotPasswordView.as_view(), name="forgot-password"),
    path("auth/reset-password/", ResetPasswordView.as_view(), name="reset-password"),
    path("auth/logout/", LogoutView.as_view(), name="logout"),
    path("auth/google/", GoogleAuthView.as_view(), name="google-auth"),
    
    # Profile endpoints
    path("profile/", GetProfileView.as_view(), name="profile"),
    path("profile/update/", UpdateProfileView.as_view(), name="update-profile"),
    path("profile/update-email/request/", RequestEmailUpdateView.as_view(), name="request-email-update"),
    path("profile/update-email/confirm/", ConfirmEmailUpdateView.as_view(), name="confirm-email-update"),
    
    # Problem and Matching endpoints
    path("problem/", ProblemView.as_view(), name="problem"),
    path("matches/", MatchView.as_view(), name="matches"),
    path("chats/", ChatListView.as_view(), name="chat-list"),
    path("chats/<int:chat_id>/", ChatView.as_view(), name="chat-detail"),
=======
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
        "check_shortlisted/<int:pk>/<str:resume_id>/",
        shortlist_check,
        name="check-shortlisted",
    ),
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
>>>>>>> a874d34 (Replace OpenAI/Claude APIs with Gemini API and fix various issues)
]
