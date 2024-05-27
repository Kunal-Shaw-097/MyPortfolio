from django.db import models


def validate_file_extension(value):
    import os
    from django.core.exceptions import ValidationError
    ext = os.path.splitext(value.name)[1]  # [0] returns path+filename
    valid_extensions = ['.pdf']
    if not ext.lower() in valid_extensions:
        raise ValidationError('Unsupported file extension.')
    
# Create your models here.
class PdfUpload(models.Model):
    id = models.CharField(primary_key= True ,max_length=100)
    pdf = models.FileField(upload_to='pdfs', validators=[validate_file_extension])
