from rest_framework import serializers
from django.contrib.auth import authenticate
from .models import (
    Recruiter,
    Job,
    JobRecruiter,
    Shortlist,
    Resume,
    Candidate,
)


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
        fields = ["id", "resume_id", "job_id", "shortlisted_by", "shortlisted_at"]


class CandidateSerializer(serializers.ModelSerializer):
    class Meta:
        model = Candidate
        fields = ["id", "name", "email", "created_at"]


class ResumeVersionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Resume
        fields = ["id", "version", "pdf_file", "upload_date", "uploaded_by"]
