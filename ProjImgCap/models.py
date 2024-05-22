from django.db import models

# Create your models here.
class imageUpload(models.Model):
    id = models.CharField(primary_key= True ,max_length=100)
    image = models.ImageField(upload_to='images')