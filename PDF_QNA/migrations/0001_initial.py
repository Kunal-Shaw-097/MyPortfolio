# Generated by Django 5.0.6 on 2024-05-26 10:20

import PDF_QNA.models
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='PdfUpload',
            fields=[
                ('id', models.CharField(max_length=100, primary_key=True, serialize=False)),
                ('pdf', models.FileField(upload_to='pdfs', validators=[PDF_QNA.models.validate_file_extension])),
            ],
        ),
    ]
