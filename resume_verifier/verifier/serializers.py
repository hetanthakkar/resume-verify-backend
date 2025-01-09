from rest_framework import serializers
from .models import VerificationResult
from .models import ProfileMatchResult


class VerificationResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = VerificationResult
        fields = "__all__"


class ProfileMatchResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = ProfileMatchResult
        fields = "__all__"
