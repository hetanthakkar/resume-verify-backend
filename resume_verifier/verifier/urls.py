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

urlpatterns = [
    # Authentication endpoints
    path("auth/register/", RegisterView.as_view(), name="register"),
    path("auth/check-email/", CheckEmailRegisteredView.as_view(), name="check-email"),
    path("auth/refresh-token/", RefreshTokenView.as_view(), name="refresh-token"),
    path("auth/verify-otp/", VerifyOTPView.as_view(), name="verify-otp"),
    path("auth/login/", LoginView.as_view(), name="login"),
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
]
