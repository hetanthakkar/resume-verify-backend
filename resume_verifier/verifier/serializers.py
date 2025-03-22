from rest_framework import serializers
from django.contrib.auth import authenticate
from .models import (
    Recruiter,
    Job,
    JobRecruiter,
    Shortlist,
    Resume,
    Candidate,
    OTPVerification,
)


from rest_framework import serializers
from django.core.mail import send_mail
from django.conf import settings


class RegisterSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True)

    class Meta:
        model = Recruiter
        fields = ("email", "password", "name", "company")

    def create(self, validated_data):
        recruiter = Recruiter.objects.create_user(
            email=validated_data["email"],
            password=validated_data["password"],
            name=validated_data["name"],
            company=validated_data["company"],
        )
        return recruiter


class LoginSerializer(serializers.Serializer):
    email = serializers.EmailField()
    password = serializers.CharField()

    def validate(self, data):
        user = authenticate(**data)
        if user and user.is_active:
            return user
        raise serializers.ValidationError("Incorrect credentials")


class RecruiterSerializer(serializers.ModelSerializer):
    class Meta:
        model = Recruiter
        fields = ["id", "name", "email", "company"]


from rest_framework import serializers


class JobSerializer(serializers.ModelSerializer):
    class Meta:
        model = Job
        fields = [
            "id",
            "title",
            "company_name",
            "description",
            "location",
            "employment_type",
            "source_url",
            "required_skills",
            "preferred_skills",
            "years_of_experience",
            "education",
            "created_by",
            "created_at",
            "updated_at",
        ]
        read_only_fields = ["created_by", "created_at", "updated_at"]


class JobRecruiterSerializer(serializers.ModelSerializer):
    class Meta:
        model = JobRecruiter
        fields = ["id", "job", "recruiter", "status", "joined_at", "updated_at"]
        read_only_fields = ["joined_at", "updated_at"]


class CandidateSerializer(serializers.ModelSerializer):
    class Meta:
        model = Candidate
        fields = ["id", "name", "email"]


class ResumeSerializer(serializers.ModelSerializer):
    candidate = CandidateSerializer()

    class Meta:
        model = Resume
        fields = ["id", "candidate", "version", "upload_date"]


class ShortlistSerializer(serializers.ModelSerializer):
    class Meta:
        model = Shortlist
        fields = [
            "id",
            "resume_id",
            "job_id",
            "shortlisted_by",
            "shortlisted_at",
            "candidate_name",  # Added this field
            "analysis_data",  # Added analysis_data
        ]


class CandidateSerializer(serializers.ModelSerializer):
    class Meta:
        model = Candidate
        fields = ["id", "name", "email", "created_at"]


class ResumeVersionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Resume
        fields = ["id", "version", "pdf_file", "upload_date", "uploaded_by"]


class RegisterSerializer(serializers.ModelSerializer):
    class Meta:
        model = Recruiter
        fields = ("email", "password", "name", "company")
        extra_kwargs = {"password": {"write_only": True}}

    def create(self, validated_data):
        if "registration_data" in self.context:
            user = Recruiter.objects.create_user(**validated_data)
            return user
        verification = OTPVerification.generate_otp(validated_data["email"])
        verification.registration_data = validated_data
        verification.save()

        send_mail(
            "Email Verification Code",
            f"Your verification code is: {verification.otp}",
            settings.EMAIL_FROM_ADDRESS,
            [validated_data["email"]],
            fail_silently=False,
        )
        return verification


class VerifyOTPSerializer(serializers.Serializer):
    email = serializers.EmailField()
    otp = serializers.CharField()


class ForgotPasswordSerializer(serializers.Serializer):
    email = serializers.EmailField()


class ResetPasswordSerializer(serializers.Serializer):
    email = serializers.EmailField()
    otp = serializers.CharField()
    new_password = serializers.CharField(write_only=True)

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


class RecruiterProfileUpdateSerializer(serializers.ModelSerializer):
    class Meta:
        model = Recruiter
        fields = ["name", "company"]


class RecruiterEmailUpdateRequestSerializer(serializers.Serializer):
    new_email = serializers.EmailField()

    def validate_new_email(self, value):
        if Recruiter.objects.filter(email=value).exists():
            raise serializers.ValidationError("This email is already in use.")
        return value


class RecruiterEmailUpdateConfirmSerializer(serializers.Serializer):
    new_email = serializers.EmailField()
    otp = serializers.CharField(max_length=6)

    def validate(self, data):
        try:
            verification = OTPVerification.objects.get(
                email=data["new_email"], otp=data["otp"], is_verified=False
            )
            if not verification.is_valid():
                raise serializers.ValidationError("OTP has expired")
            return data
        except OTPVerification.DoesNotExist:
            raise serializers.ValidationError("Invalid OTP")
