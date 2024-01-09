from django.shortcuts import render
from django.core.cache import cache

from .utils.general import resume_checkpoint
from .utils.tokenizer import Tokenizer
from .utils.process_image import letterbox
import torch
import numpy as np
import cv2

def get_all():
    model = cache.get('model')  
    tokenizer = cache.get('tokenizer')
    device = cache.get('device')                                            #try to get cached object if it exists

    if not model:
        device = 'cuda' if torch.cuda.is_available() else 'cpu' 
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
    if request.method == 'POST' and request.FILES['image']:
        # Get the uploaded image
        image = request.FILES['image']

        # Read the image contents into memory
        file_bytes = np.frombuffer(image.read(), np.uint8)
        img = cv2.imdecode(file_bytes,  cv2.IMREAD_COLOR)
        img = letterbox(img, (480,480))
        img_in = torch.from_numpy(img).to(device).unsqueeze(0).permute(0, 3, 1, 2).contiguous().float()/255
        pred = model.generate(img_in, tokenizer, device=device, greedy= True, top_k=5)
        caption = tokenizer.decode(pred)
        context = {
            "caption" : caption[0],
        }
    return render(request, "upload.html", context)