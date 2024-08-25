from django.db import models

# Create your models here.
class ImgModel(models.Model):
    
    image = models.ImageField(upload_to = "dams")
class UserModel(models.Model):
    
    name = models.CharField(max_length=100)
    position = models.CharField(max_length=100)
    
    