from django.contrib import admin

# Register your models here.
from .models import ImgModel,UserModel

admin.site.register(ImgModel)
admin.site.register(UserModel)
