from django.shortcuts import render

# Create your views here.
def Image_caption(request):
    return render(request, "upload.html")