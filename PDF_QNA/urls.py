from django.urls import path
from . import views

urlpatterns = [
    path('', views.pdf, name='pdf'),
    # other paths...
]