from django.contrib.auth.models import AbstractUser, BaseUserManager
from django.db import models
from django.utils import timezone
from django.core.validators import MinValueValidator, MaxValueValidator
from django.contrib.postgres.fields import ArrayField
from django.core.exceptions import ValidationError
from django.utils.translation import gettext_lazy as _
from django.conf import settings
import random
from datetime import datetime, timedelta
import numpy as np
from django.db.models import F
from django.db.models.functions import Ln, Exp
from django.db.models import FloatField
from django.db.models.functions import Sqrt
from django.db.models import ExpressionWrapper
from django.db.models import Q


class UserManager(BaseUserManager):
    def create_user(self, email, password=None, **extra_fields):
        if not email:
            raise ValueError(_("The Email field must be set"))
        email = self.normalize_email(email)
        user = self.model(email=email, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, email, password=None, **extra_fields):
        extra_fields.setdefault("is_staff", True)
        extra_fields.setdefault("is_superuser", True)
        return self.create_user(email, password, **extra_fields)


class User(AbstractUser):
    email = models.EmailField(unique=True)
    username = models.CharField(max_length=150, unique=True)
    date_of_birth = models.DateField(null=True, blank=True)
    latitude = models.FloatField(
        validators=[MinValueValidator(-90), MaxValueValidator(90)],
        null=True,
        blank=True
    )
    longitude = models.FloatField(
        validators=[MinValueValidator(-180), MaxValueValidator(180)],
        null=True,
        blank=True
    )
    city = models.CharField(max_length=100, null=True, blank=True)
    company = models.CharField(max_length=100, null=True, blank=True)
    problem_category = models.CharField(
        max_length=50,
        choices=[
            ('anxiety', 'Anxiety'),
            ('depression', 'Depression'),
            ('stress', 'Stress Management'),
            ('trauma', 'Trauma & PTSD'),
            ('addiction', 'Addiction & Recovery'),
            ('selfesteem', 'Self-Esteem'),
            ('relationships', 'Relationship Issues'),
            ('grief', 'Grief & Loss'),
            ('eating', 'Eating & Body Image'),
            ('sleep', 'Sleep Issues'),
            ('focus', 'Attention & Focus'),
            ('isolation', 'Loneliness & Isolation'),
            ('identity', 'Identity & Purpose'),
            ('other', 'Other'),
        ],
        null=True,
        blank=True
    )
    pinecone_id = models.CharField(max_length=255, null=True, blank=True, unique=True)
    objects = UserManager()

    USERNAME_FIELD = "email"
    REQUIRED_FIELDS = ["username"]

    def __str__(self):
        return self.email


class Problem(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='problem')
    description = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Problem for {self.user.email}"


class Chat(models.Model):
    from_user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='sent_chats')
    to_user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='received_chats')
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    is_read = models.BooleanField(default=False)

    class Meta:
        ordering = ['created_at']

    def __str__(self):
        return f"Chat from {self.from_user.email} to {self.to_user.email}"


class Match(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='matches')
    matched_user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='matched_by')
    similarity_score = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )
    created_at = models.DateTimeField(auto_now_add=True)
    last_interaction = models.DateTimeField(null=True, blank=True)

    class Meta:
        unique_together = ['user', 'matched_user']
        ordering = ['-similarity_score']

    def __str__(self):
        return f"Match between {self.user.email} and {self.matched_user.email}"

    def clean(self):
        if self.user == self.matched_user:
            raise ValidationError("A user cannot match with themselves")

    @classmethod
    def get_matches_for_user(cls, user, limit=10):
        """Get top matches for a user, ordered by similarity score."""
        return cls.objects.filter(
            user=user
        ).select_related('matched_user').order_by('-similarity_score')[:limit]

    @classmethod
    def get_mutual_matches(cls, user):
        """Get users who have matched with each other."""
        return cls.objects.filter(
            user=user,
            matched_user__matches__user=F('matched_user'),
            matched_user__matches__matched_user=user
        ).select_related('matched_user')


class OTPVerification(models.Model):
    email = models.EmailField()
    otp = models.CharField(max_length=6)
    created_at = models.DateTimeField(auto_now_add=True)
    is_verified = models.BooleanField(default=False)
    registration_data = models.JSONField(default=dict)

    @classmethod
    def generate_otp(cls, email):
        cls.objects.filter(email=email).delete()
        otp = "".join([str(random.randint(0, 9)) for _ in range(6)])
        return cls.objects.create(email=email, otp=otp)

    def is_valid(self):
        return datetime.now() - timedelta(minutes=10) <= self.created_at.replace(
            tzinfo=None
        )
