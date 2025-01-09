from django.db import models


import uuid


class Project(models.Model):
    name = str
    description = str
    technologies = list
    url = str
    context = str
    confidence = str

    class Meta:
        managed = False


class VerificationResult(models.Model):
    timestamp = models.DateTimeField(auto_now_add=True)
    pdf_file = models.FileField(upload_to="pdfs/")
    results = models.JSONField()

    def __str__(self):
        return f"Verification Result {self.timestamp}"


# verifier/serializers.py
from rest_framework import serializers
from .models import VerificationResult


class VerificationResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = VerificationResult
        fields = "__all__"


class ProfileMatchResult(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    pdf_url = models.URLField()
    linkedin_url = models.URLField()
    results = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"Profile Match {self.id}"
