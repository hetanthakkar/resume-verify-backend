import requests as http_requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Python standard library
import re
from google.oauth2 import id_token
from google.auth.transport import requests

# Third-party imports
import openai

# Django imports
from django.conf import settings
from django.core.exceptions import ValidationError
from django.http import HttpRequest
from django.core.mail import send_mail
from django.utils import timezone
from django.db.models import Q

# Django REST Framework imports
from rest_framework import permissions, status, views
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework_simplejwt.tokens import RefreshToken, TokenError

# Local imports
from .models import (
    User,
    OTPVerification,
    Problem,
    Match,
    Chat,
)
from .serializers import (
    UserSerializer,
    ProfileUpdateSerializer,
    LoginSerializer,
    RegisterSerializer,
    VerifyOTPSerializer,
    ForgotPasswordSerializer,
    ResetPasswordSerializer,
    ProblemSerializer,
    MatchSerializer,
    ChatSerializer,
    ChatListSerializer,
    EmailUpdateSerializer,
)

# Initialize OpenAI client
openai.api_key = "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"

def get_embedding(text):
    """Generate embedding using OpenAI's API"""
    if not openai.api_key:
        raise ValueError("OpenAI API key is not configured")
    response = openai.embeddings.create(
        input=[text],
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

class RegisterView(APIView):
    def post(self, request):
        serializer = RegisterSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(
                {"message": "Please verify your email with the OTP sent"},
                status=status.HTTP_200_OK,
            )
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class CheckEmailRegisteredView(APIView):
    def post(self, request):
        email = request.data.get("email")

        if not email:
            return Response(
                {"error": "Email field is required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if User.objects.filter(email=email).exists():
            user = User.objects.get(email=email)
            refresh = RefreshToken.for_user(user)
            return Response(
                {
                    "message": "Email is registered",
                    "access": str(refresh.access_token),
                    "refresh": str(refresh),
                },
                status=status.HTTP_200_OK,
            )

        return Response(
            {"message": "Email is not registered."},
            status=status.HTTP_404_NOT_FOUND,
        )


class RefreshTokenView(APIView):
    def post(self, request):
        refresh_token = request.data.get("refresh")

        if not refresh_token:
            return Response(
                {"error": "Refresh token is required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            refresh = RefreshToken(refresh_token)
            new_access_token = str(refresh.access_token)

            return Response(
                {"access": new_access_token},
                status=status.HTTP_200_OK,
            )
        except Exception as e:
            return Response(
                {"error": "Invalid or expired refresh token."},
                status=status.HTTP_400_BAD_REQUEST,
            )


class VerifyOTPView(APIView):
    def post(self, request):
        serializer = VerifyOTPSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        try:
            verification = OTPVerification.objects.get(
                email=serializer.validated_data["email"],
                otp=serializer.validated_data["otp"],
                is_verified=False,
            )

            if not verification.is_valid():
                return Response(
                    {"error": "OTP has expired"}, status=status.HTTP_400_BAD_REQUEST
                )

            # Get or create user with the registration data
            try:
                user = User.objects.get(email=verification.email)
            except User.DoesNotExist:
                register_serializer = RegisterSerializer(
                    data=verification.registration_data, context={"registration_data": True}
                )
                if register_serializer.is_valid():
                    user = register_serializer.save()
                else:
                    return Response(
                        register_serializer.errors, status=status.HTTP_400_BAD_REQUEST
                    )

            # Activate user and mark OTP as verified
            user.is_active = True
            user.save()
            verification.is_verified = True
            verification.save()

            # Generate tokens
            refresh = RefreshToken.for_user(user)
            return Response(
                {
                    "refresh": str(refresh),
                    "access": str(refresh.access_token),
                    "user": UserSerializer(user).data,
                },
                status=status.HTTP_201_CREATED,
            )

        except OTPVerification.DoesNotExist:
            return Response(
                {"error": "Invalid OTP"}, status=status.HTTP_400_BAD_REQUEST
            )


class LoginView(APIView):
    def post(self, request: HttpRequest) -> Response:
        serializer = LoginSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.validated_data
            refresh = RefreshToken.for_user(user)
            return Response(
                {
                    "refresh": str(refresh),
                    "access": str(refresh.access_token),
                    "user": UserSerializer(user).data,
                }
            )
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class GetProfileView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request: HttpRequest) -> Response:
        serializer = UserSerializer(request.user)
        return Response(serializer.data)


class ForgotPasswordView(APIView):
    def post(self, request):
        serializer = ForgotPasswordSerializer(data=request.data)
        if serializer.is_valid():
            email = serializer.validated_data["email"]
            try:
                user = User.objects.get(email=email)
                verification = OTPVerification.generate_otp(email)

                send_mail(
                    "Password Reset Code",
                    f"Your password reset code is: {verification.otp}",
                    settings.EMAIL_FROM_ADDRESS,
                    [email],
                    fail_silently=False,
                )

                return Response(
                    {"message": "Password reset code has been sent to your email"},
                    status=status.HTTP_200_OK,
                )
            except User.DoesNotExist:
                return Response(
                    {
                        "message": "If an account exists with this email, a password reset code has been sent"
                    },
                    status=status.HTTP_200_OK,
                )
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class ResetPasswordView(APIView):
    def post(self, request):
        serializer = ResetPasswordSerializer(data=request.data)
        if serializer.is_valid():
            email = serializer.validated_data["email"]
            new_password = serializer.validated_data["new_password"]

            try:
                verification = OTPVerification.objects.get(
                    email=email, otp=serializer.validated_data["otp"], is_verified=False
                )

                user = User.objects.get(email=email)
                user.set_password(new_password)
                user.save()

                verification.is_verified = True
                verification.save()

                return Response(
                    {"message": "Password has been reset successfully"},
                    status=status.HTTP_200_OK,
                )
            except (OTPVerification.DoesNotExist, User.DoesNotExist):
                return Response(
                    {"message": "Invalid reset attempt"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class UpdateProfileView(APIView):
    permission_classes = [IsAuthenticated]

    def put(self, request):
        serializer = ProfileUpdateSerializer(request.user, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class LogoutView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        try:
            refresh_token = request.data.get("refresh")
            if refresh_token:
                token = RefreshToken(refresh_token)
                token.blacklist()
            return Response({"message": "Successfully logged out"})
        except Exception as e:
            return Response(
                {"error": "Failed to logout"}, status=status.HTTP_400_BAD_REQUEST
            )


class GoogleAuthView(APIView):
    def post(self, request):
        access_token = request.data.get("access_token")

        if not access_token:
            return Response(
                {"error": "Access token is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            google_response = http_requests.get(
                "https://www.googleapis.com/oauth2/v3/userinfo",
                headers={"Authorization": f"Bearer {access_token}"},
            )

            if not google_response.ok:
                return Response(
                    {"error": "Failed to verify token"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            user_info = google_response.json()
            email = user_info["email"]
            is_new_user = False

            try:
                user = User.objects.get(email=email)
            except User.DoesNotExist:
                user = User.objects.create(
                    email=email,
                    first_name=user_info.get("given_name", ""),
                    last_name=user_info.get("family_name", ""),
                )
                is_new_user = True

            refresh = RefreshToken.for_user(user)

            return Response(
                {
                    "refresh": str(refresh),
                    "access": str(refresh.access_token),
                    "user": UserSerializer(user).data,
                    "is_new_user": is_new_user,
                }
            )

        except Exception as e:
            print(f"Google auth error: {str(e)}")
            return Response(
                {"error": "Authentication failed"}, status=status.HTTP_400_BAD_REQUEST
            )


class ProblemView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        try:
            problem = Problem.objects.get(user=request.user)
            return Response({
                'description': problem.description,
                'created_at': problem.created_at,
                'updated_at': problem.updated_at
            })
        except Problem.DoesNotExist:
            return Response(
                {"error": "No problem description found"},
                status=status.HTTP_404_NOT_FOUND
            )

    def post(self, request):
        try:
            # Get or create problem
            problem, created = Problem.objects.get_or_create(
                user=request.user,
                defaults={'description': request.data.get('description', '')}
            )
            
            # Update description if provided
            if 'description' in request.data:
                problem.description = request.data['description']
                problem.save()
            
            return Response({
                'description': problem.description,
                'created_at': problem.created_at,
                'updated_at': problem.updated_at
            }, status=status.HTTP_201_CREATED if created else status.HTTP_200_OK)
            
        except Exception as e:
            print(f"Error in problem view: {str(e)}")  # For debugging
            return Response(
                {"error": "An error occurred while saving your problem"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class MatchView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        """Get existing matches for the user"""
        matches = Match.get_matches_for_user(request.user)
        serializer = MatchSerializer(matches, many=True)
        return Response(serializer.data)

    def post(self, request):
        """Find new matches using cosine similarity"""
        try:
            # Get current user's problem
            current_problem = Problem.objects.get(user=request.user)
            current_embedding = get_embedding(current_problem.description)
            
            # Get all other users with problems
            other_problems = Problem.objects.exclude(user=request.user)
            
            # Calculate similarities
            matches = []
            for other_problem in other_problems:
                other_embedding = get_embedding(other_problem.description)
                
                # Calculate cosine similarity
                similarity = cosine_similarity(
                    [current_embedding],
                    [other_embedding]
                )[0][0]
                
                # Create or update match
                match_obj, created = Match.objects.update_or_create(
                    user=request.user,
                    matched_user=other_problem.user,
                    defaults={
                        'similarity_score': float(similarity),
                        'last_interaction': timezone.now()
                    }
                )
                matches.append(match_obj)
            
            # Sort matches by similarity score
            matches.sort(key=lambda x: x.similarity_score, reverse=True)
            
            serializer = MatchSerializer(matches, many=True)
            return Response(serializer.data)
            
        except Problem.DoesNotExist:
            return Response(
                {"error": "Please add a problem description first"},
                status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            print(f"Error in match view: {str(e)}")  # For debugging
            return Response(
                {"error": "An error occurred while finding matches"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class ChatView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, chat_id):
        """Get chat messages between two users"""
        try:
            chat = Chat.objects.get(id=chat_id)
            if chat.from_user != request.user and chat.to_user != request.user:
                return Response(
                    {"error": "You don't have permission to view this chat"},
                    status=status.HTTP_403_FORBIDDEN
                )
            
            # Mark messages as read
            Chat.objects.filter(
                from_user=chat.from_user if chat.from_user != request.user else chat.to_user,
                to_user=request.user,
                is_read=False
            ).update(is_read=True)

            messages = Chat.objects.filter(
                Q(from_user=chat.from_user, to_user=chat.to_user) |
                Q(from_user=chat.to_user, to_user=chat.from_user)
            ).order_by('created_at')

            serializer = ChatSerializer(messages, many=True)
            return Response(serializer.data)
        except Chat.DoesNotExist:
            return Response(
                {"error": "Chat not found"},
                status=status.HTTP_404_NOT_FOUND
            )

    def post(self, request, chat_id):
        """Send a message in a chat"""
        try:
            chat = Chat.objects.get(id=chat_id)
            if chat.from_user != request.user and chat.to_user != request.user:
                return Response(
                    {"error": "You don't have permission to send messages in this chat"},
                    status=status.HTTP_403_FORBIDDEN
                )

            content = request.data.get('content')
            if not content:
                return Response(
                    {"error": "Message content is required"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Determine the recipient
            recipient = chat.to_user if chat.from_user == request.user else chat.from_user

            # Create new message
            new_message = Chat.objects.create(
                from_user=request.user,
                to_user=recipient,
                content=content
            )

            # Update last interaction in Match model
            Match.objects.filter(
                Q(user=request.user, matched_user=recipient) |
                Q(user=recipient, matched_user=request.user)
            ).update(last_interaction=timezone.now())

            serializer = ChatSerializer(new_message)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        except Chat.DoesNotExist:
            return Response(
                {"error": "Chat not found"},
                status=status.HTTP_404_NOT_FOUND
            )

class ChatListView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        """Get list of all chats for the current user"""
        # Get all unique conversations by getting the latest message for each conversation
        chats = Chat.objects.filter(
            Q(from_user=request.user) | Q(to_user=request.user)
        ).order_by(
            'from_user', 'to_user', '-created_at'
        ).distinct('from_user', 'to_user')

        serializer = ChatListSerializer(chats, many=True, context={'request': request})
        return Response(serializer.data)

class RequestEmailUpdateView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        try:
            email = request.data.get('email')
            if not email:
                return Response(
                    {"error": "Email is required"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            if User.objects.filter(email=email).exists():
                return Response(
                    {"error": "This email is already registered"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Generate OTP
            verification = OTPVerification.generate_otp(email)
            verification.registration_data = {
                'current_email': request.user.email,
                'new_email': email
            }
            verification.save()

            # Send verification email
            try:
                send_mail(
                    "Email Update Verification",
                    f"Your verification code is: {verification.otp}",
                    settings.EMAIL_FROM_ADDRESS,
                    [email],
                    fail_silently=False,
                )
            except Exception as e:
                print(f"Error sending email: {str(e)}")
                return Response(
                    {"error": "Failed to send verification email. Please try again."},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

            return Response(
                {"message": "Verification code sent to your new email"},
                status=status.HTTP_200_OK
            )
        except Exception as e:
            print(f"Error in RequestEmailUpdateView: {str(e)}")
            return Response(
                {"error": "An unexpected error occurred"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class ConfirmEmailUpdateView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        try:
            serializer = EmailUpdateSerializer(data=request.data)
            if not serializer.is_valid():
                return Response(
                    {"error": serializer.errors},
                    status=status.HTTP_400_BAD_REQUEST
                )

            try:
                verification = OTPVerification.objects.get(
                    email=serializer.validated_data['email'],
                    otp=serializer.validated_data['otp'],
                    is_verified=False
                )

                if not verification.is_valid():
                    return Response(
                        {"error": "OTP has expired"},
                        status=status.HTTP_400_BAD_REQUEST
                    )

                # Update user's email
                user = request.user
                user.email = serializer.validated_data['email']
                user.save()

                # Mark verification as complete
                verification.is_verified = True
                verification.save()

                return Response(
                    {"message": "Email updated successfully"},
                    status=status.HTTP_200_OK
                )

            except OTPVerification.DoesNotExist:
                return Response(
                    {"error": "Invalid verification code"},
                    status=status.HTTP_400_BAD_REQUEST
                )
        except Exception as e:
            print(f"Error in ConfirmEmailUpdateView: {str(e)}")
            return Response(
                {"error": "An unexpected error occurred"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
