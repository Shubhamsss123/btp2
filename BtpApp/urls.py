from django.urls import path,include
from .views import img_view,user_view

urlpatterns = [
     path('img', img_view, name="img_name"),
     path('user',user_view,name="user_name")
   
]