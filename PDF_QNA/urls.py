from django.urls import path
from . import views

app_name = 'PDF_QNA'

urlpatterns = [
    path('', views.pdf, name='pdf'),
    path('upload', views.upload_pdf, name="upload_pdf"), 
    path('qna', views.qna, name="qna") 
    # other paths...
]