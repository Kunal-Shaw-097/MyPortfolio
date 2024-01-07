from django.urls import path
from . import views

urlpatterns = [
    path('', views.Image_caption, name='ImageCaptioning'),
    # other paths...
]