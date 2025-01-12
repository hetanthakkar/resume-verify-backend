from django.db import models

from django.core.validators import FileExtensionValidator

import uuid
from django.contrib.auth.models import AbstractUser, BaseUserManager
from django.core.exceptions import ValidationError

from django.conf import settings
from django.contrib.postgres.fields import ArrayField
import json


class RecruiterManager(BaseUserManager):
    def create_user(self, email, password=None, **extra_fields):
        if not email:
            raise ValueError("The Email field must be set")
        email = self.normalize_email(email)
        user = self.model(email=email, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, email, password=None, **extra_fields):
        extra_fields.setdefault("is_staff", True)
        extra_fields.setdefault("is_superuser", True)

        return self.create_user(email, password, **extra_fields)


class Recruiter(AbstractUser):
    username = None
    email = models.EmailField(unique=True)
    name = models.CharField(max_length=255)
    company = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)
    groups = None
    user_permissions = None
    objects = RecruiterManager()

    USERNAME_FIELD = "email"
    REQUIRED_FIELDS = ["name", "company"]

    def __str__(self):
        return f"{self.name} - {self.company}"


class Job(models.Model):
    title = models.CharField(max_length=255)
    company_name = models.CharField(max_length=255)
    description = models.TextField()
    location = models.CharField(max_length=255)
    employment_type = models.CharField(max_length=50)
    source_url = models.URLField(max_length=500)
    required_skills = models.JSONField(default=list)  # Store as JSON string
    preferred_skills = models.JSONField(default=list)  # Store as JSON string
    years_of_experience = models.IntegerField(null=True, blank=True)
    education = models.TextField(null=True, blank=True)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, related_name="created_jobs", on_delete=models.CASCADE
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def save(self, *args, **kwargs):
        # Ensure required_skills and preferred_skills are lists
        if self.required_skills is None:
            self.required_skills = []
        if self.preferred_skills is None:
            self.preferred_skills = []

        # Convert years_of_experience to int if it's a string
        if (
            isinstance(self.years_of_experience, str)
            and self.years_of_experience.isdigit()
        ):
            self.years_of_experience = int(self.years_of_experience)

        super().save(*args, **kwargs)


class JobRecruiter(models.Model):
    STATUS_CHOICES = (
        ("ACTIVE", "Active"),
        ("INACTIVE", "Inactive"),
        ("REMOVED", "Removed"),
    )

    job = models.ForeignKey(
        "Job", on_delete=models.CASCADE, related_name="job_recruiters"
    )
    recruiter = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="recruited_jobs",
    )
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="ACTIVE")
    joined_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ("job", "recruiter")
        indexes = [
            models.Index(fields=["job", "recruiter", "status"]),
        ]

    def save(self, *args, **kwargs):
        self.full_clean()
        super().save(*args, **kwargs)


class Candidate(models.Model):
    name = models.CharField(max_length=255)
    email = models.EmailField(unique=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name


class Resume(models.Model):
    candidate = models.ForeignKey(
        Candidate, on_delete=models.CASCADE, related_name="resumes"
    )
    pdf_file = models.FileField(
        upload_to="resumes/",
        validators=[FileExtensionValidator(allowed_extensions=["pdf"])],
    )
    version = models.IntegerField()
    upload_date = models.DateTimeField(auto_now_add=True)
    uploaded_by = models.ForeignKey(
        Recruiter, on_delete=models.SET_NULL, null=True, related_name="uploaded_resumes"
    )

    class Meta:
        unique_together = ["candidate", "version"]

    def __str__(self):
        return f"{self.candidate.name}'s Resume v{self.version}"


class ResumeAnalysis(models.Model):
    resume = models.ForeignKey(
        Resume, on_delete=models.CASCADE, related_name="analyses"
    )
    job = models.ForeignKey(
        Job, on_delete=models.CASCADE, related_name="resume_analyses"
    )
    linkedin_url = models.URLField(blank=True, null=True)
    analysis_data = models.JSONField()
    analyzed_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ["resume", "job"]

    def __str__(self):
        return f"Analysis for {self.resume.candidate.name} - {self.job.title}"


class Shortlist(models.Model):
    id = models.AutoField(primary_key=True)
    resume = models.ForeignKey(
        "Resume", on_delete=models.CASCADE, db_column="resume_id"
    )  # Changed from resume_id
    job = models.ForeignKey(
        "Job", on_delete=models.CASCADE, db_column="job_id"
    )  # Changed from job_id
    shortlisted_by = models.ForeignKey(
        "Recruiter", on_delete=models.CASCADE, db_column="shortlisted_by"
    )
    shortlisted_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "shortlist"
