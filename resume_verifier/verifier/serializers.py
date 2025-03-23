from rest_framework import serializers
from django.contrib.auth import get_user_model, authenticate
from .models import (
    User,
    OTPVerification,
    Problem,
    Match,
    Chat,
)
from django.core.mail import send_mail
from django.conf import settings
import random
from django.utils import timezone

User = get_user_model()


class RegisterSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True)
    problem_category = serializers.CharField(required=False)
    problem_description = serializers.CharField(required=False)

    class Meta:
        model = User
        fields = ("id", "email", "username", "password", "problem_category", "problem_description")
        read_only_fields = ("id",)

    def create(self, validated_data):
        # Generate OTP
        otp = "".join([str(random.randint(0, 9)) for _ in range(6)])
        
        # Create user as inactive
        user = User.objects.create_user(
            email=validated_data["email"],
            username=validated_data["username"],
            password=validated_data["password"],
            is_active=False,  # User starts as inactive
            problem_category=validated_data.get("problem_category")  # Set problem category
        )
        
        # Create Problem instance if description is provided
        if validated_data.get("problem_description"):
            Problem.objects.create(
                user=user,
                description=validated_data["problem_description"]
            )
        
        # Store registration data and OTP
        registration_data = {
            "email": validated_data["email"],
            "username": validated_data["username"],
            "password": validated_data["password"],
            "problem_category": validated_data.get("problem_category", ""),
            "problem_description": validated_data.get("problem_description", ""),
        }
        
        OTPVerification.objects.create(
            email=validated_data["email"],
            otp=otp,
            registration_data=registration_data
        )
        
        # Send verification email
        subject = "Email Verification"
        message = f"Your verification code is: {otp}"
        from_email = settings.EMAIL_FROM_ADDRESS
        recipient_list = [validated_data["email"]]
        
        try:
            send_mail(subject, message, from_email, recipient_list)
        except Exception as e:
            # If email sending fails, delete the user and OTP verification
            user.delete()
            raise serializers.ValidationError({"email": "Failed to send verification email"})
        
        return user


class LoginSerializer(serializers.Serializer):
    email = serializers.EmailField()
    password = serializers.CharField()

    def validate(self, data):
        user = authenticate(**data)
        if user and user.is_active:
            return user
        raise serializers.ValidationError("Incorrect credentials")


class UserSerializer(serializers.ModelSerializer):
    problem = serializers.SerializerMethodField()

    class Meta:
        model = User
        fields = ('id', 'email', 'username', 'date_of_birth', 
                 'latitude', 'longitude', 'city', 'company', 'problem_category', 
                 'problem')
        read_only_fields = ('id',)

    def get_problem(self, obj):
        try:
            problem = obj.problem
            return {
                'description': problem.description,
                'created_at': problem.created_at,
                'updated_at': problem.updated_at
            }
        except Problem.DoesNotExist:
            return None


class ProblemSerializer(serializers.ModelSerializer):
    class Meta:
        model = Problem
        fields = ('id', 'description', 'created_at', 'updated_at')
        read_only_fields = ('id', 'created_at', 'updated_at')


class MatchSerializer(serializers.ModelSerializer):
    matched_user = UserSerializer(read_only=True)

    class Meta:
        model = Match
        fields = ('id', 'matched_user', 'similarity_score', 
                 'created_at', 'last_interaction')
        read_only_fields = ('id', 'created_at', 'last_interaction')


class VerifyOTPSerializer(serializers.Serializer):
    email = serializers.EmailField()
    otp = serializers.CharField(max_length=6)


class ForgotPasswordSerializer(serializers.Serializer):
    email = serializers.EmailField()


class ResetPasswordSerializer(serializers.Serializer):
    email = serializers.EmailField()
    otp = serializers.CharField(max_length=6)
    new_password = serializers.CharField(min_length=8)

    def validate(self, data):
        try:
            verification = OTPVerification.objects.get(
                email=data["email"], otp=data["otp"], is_verified=False
            )
            if not verification.is_valid():
                raise serializers.ValidationError("OTP has expired")
            return data
        except OTPVerification.DoesNotExist:
            raise serializers.ValidationError("Invalid OTP")


class ProfileUpdateSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['username', 'date_of_birth', 'latitude', 'longitude', 'city', 'company', 'problem_category']


class ChatSerializer(serializers.ModelSerializer):
    from_user = UserSerializer(read_only=True)
    to_user = UserSerializer(read_only=True)

    class Meta:
        model = Chat
        fields = ('id', 'from_user', 'to_user', 'content', 'created_at', 'is_read')
        read_only_fields = ('id', 'created_at')


class ChatListSerializer(serializers.ModelSerializer):
    other_user = serializers.SerializerMethodField()
    last_message = serializers.SerializerMethodField()
    unread_count = serializers.SerializerMethodField()

    class Meta:
        model = Chat
        fields = ('id', 'other_user', 'last_message', 'created_at', 'unread_count')

    def get_other_user(self, obj):
        request = self.context.get('request')
        if obj.from_user == request.user:
            return UserSerializer(obj.to_user).data
        return UserSerializer(obj.from_user).data

    def get_last_message(self, obj):
        return obj.content

    def get_unread_count(self, obj):
        request = self.context.get('request')
        if obj.to_user == request.user:
            return Chat.objects.filter(
                from_user=obj.from_user,
                to_user=request.user,
                is_read=False
            ).count()
        return 0


class EmailUpdateSerializer(serializers.Serializer):
    email = serializers.EmailField()
    otp = serializers.CharField(max_length=6)

    def validate_email(self, value):
        if User.objects.filter(email=value).exists():
            raise serializers.ValidationError("This email is already registered.")
        return value
