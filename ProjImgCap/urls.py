from django.urls import path
from . import views

app_name = 'ProjImgCap'

urlpatterns = [
    path('', views.Image_caption, name='ImageCaptioning'),
    path('upload', views.Generate_caption, name="GenerateCaption"), 
    path('delete_image/', views.delete_image , name = 'deleteimage'),
    # other paths...
]