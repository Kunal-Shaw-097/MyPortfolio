from django.urls import path
from . import views

app_name = 'ProjImgCap'

urlpatterns = [
    path('', views.Image_caption, name='ImageCaptioning'),
    path('upload', views.Generate_caption, name="GenerateCaption") 
    # other paths...
]