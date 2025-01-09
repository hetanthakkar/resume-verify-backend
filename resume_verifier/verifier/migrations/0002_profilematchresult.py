# Generated by Django 5.1.4 on 2025-01-09 01:17

import uuid
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('verifier', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='ProfileMatchResult',
            fields=[
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('pdf_url', models.URLField()),
                ('linkedin_url', models.URLField()),
                ('results', models.JSONField()),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
            ],
            options={
                'ordering': ['-created_at'],
            },
        ),
    ]
