from django.db import models


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
