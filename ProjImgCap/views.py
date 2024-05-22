from django.shortcuts import render
from django.core.cache import cache

from .utils.general import resume_checkpoint
from .utils.tokenizer import Tokenizer
from .utils.process_image import letterbox
import torch
import numpy as np
import cv2

from pathlib import Path
import gdown

from .models import imageUpload
import uuid
import os
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json

def get_all():
    model = cache.get('model')  
    tokenizer = cache.get('tokenizer')
    device = cache.get('device')                                            #try to get cached object if it exists

    if not model:
        device = 'cuda' if torch.cuda.is_available() else 'cpu' 
        if Path("ProjImgCap/saved_model/best.pt").exists() == False :
            gdown.download(id='1sc1l1AWDHsKsV_N9OpQcnjGfgwSwdggA', output='ProjImgCap/saved_model/best.pt')
        tokenizer = Tokenizer("ProjImgCap/saved_model/vocab.json")                    
        model = resume_checkpoint("ProjImgCap/saved_model/best.pt", tokenizer= tokenizer, resume_weight_only= True, device= device)
        cache.set('model', model, None)  
        cache.set('tokenizer', tokenizer, None)                             #cache them all
        cache.set('device', device, None)

    return model, tokenizer, device

# Create your views here.
def Image_caption(request):
    return render(request, "upload.html")

def Generate_caption(request):
    model, tokenizer, device = get_all()
    model.eval()
    if request.method == 'POST' and request.FILES.get('image', False):
        try:
            # Get the uploaded image
            img_unique_id = uuid.uuid4()
            image = request.FILES['image']
            
            # Save the image to a model instance
            my_model_instance = imageUpload(id=img_unique_id, image=image)
            my_model_instance.save()
            # Read the image contents into memory
            image.seek(0)  # Ensure you're reading from the start of the file
            file_bytes = np.frombuffer(image.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            img = letterbox(img, (480, 480))
            img_in = torch.from_numpy(img).to(device).unsqueeze(0).permute(0, 3, 1, 2).contiguous().float() / 255
            pred = model.generate(img_in, tokenizer, device=device, greedy=True, top_k=5)
            caption = tokenizer.decode(pred)
            # Retrieve the image from db
            image_instance = imageUpload.objects.filter(id=img_unique_id).first()
            # Get the image URL
            image_url = image_instance.image.url
            context = {
                "caption": caption[0],
                "image_url": image_url,
                "image_id": str(img_unique_id)
            }
            return render(request, "output.html", context)
        except Exception as e:
            context = {
                "error": str(e),
                "image": None,
            }
            return render(request, "output.html", context)

    return render(request, "upload.html")

@csrf_exempt
def delete_image(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            image_id = data.get('image_id')
            image_instance = imageUpload.objects.filter(id=image_id).first()
            if image_instance:
                # Delete the image file from db
                image_path = image_instance.image.path
                if os.path.exists(image_path):
                    os.remove(image_path)
                
                # Delete the image instance from the database
                image_instance.delete()
                return JsonResponse({"message": "Image  sucesfully deleted"})
            else:
                return JsonResponse({"message": "Image found"}, status=404)
        except Exception as e:
            return JsonResponse({"message": str(e)}, status=500)
    return JsonResponse({"message": "Invalid request"}, status=400)