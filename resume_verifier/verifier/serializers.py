from rest_framework import serializers
from .models import VerificationResult


class VerificationResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = VerificationResult
        fields = "__all__"
