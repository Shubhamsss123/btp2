"""
URL configuration for btp_project project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path
from .views import *
urlpatterns = [
    path("admin/", admin.site.urls),
    path("", home_page,name='home_name'),
    # path("classification", classification_view,name='cls_name'),
    path("tsa", regression_view,name='reg_name'),
    path('model_selection',model_selection_view,name='model_selection_name'),
    path("data", data_view,name='data_name'),
    path("plot", plotly_view,name='plotly_name'),
    path('train_data',plotly_train,name='train_data_name'),
    path('test_data',plotly_test,name='test_data_name'),
    path('pred_data',plotly_pred,name='pred_data_name'),
    path('temp',temp_op,name='temp_name'),
    path('scatter',plotly_scatter,name='scatter_name'),

]
