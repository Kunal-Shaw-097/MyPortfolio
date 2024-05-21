from django.shortcuts import render
from django.core.cache import cache

from .utils.general import resume_checkpoint
from .utils.tokenizer import Tokenizer
from .utils.process_image import letterbox
import torch
import numpy as np
import cv2
import uuid
import os

from pathlib import Path
import gdown

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
    if not request.session.get('session_id', False) :
        request.session['session_id'] = str(uuid.uuid4())
        images  = os.listdir('staticfiles/') 
        for img in images:
            if img.split('.')[0].endswith('_uploaded'):
                os.remove(os.path.join('staticfiles/', img))
    return render(request, "upload.html")

def Generate_caption(request):
    model, tokenizer, device = get_all()
    model.eval()
    if request.method == 'POST' and request.FILES.get('image', False):
        # Get the uploaded image
        image = request.FILES['image']
        # Read the image contents into memory
        file_bytes = np.frombuffer(image.read(), np.uint8)
        img = cv2.imdecode(file_bytes,  cv2.IMREAD_COLOR)
        id = request.session['session_id']
        save_path = f'staticfiles/temp{id}_uploaded.png'
        uploaded = True
        img = letterbox(img, (480,480)) 
        img_in = torch.from_numpy(img).to(device).unsqueeze(0).permute(0, 3, 1, 2).contiguous().float()/255
        pred = model.generate(img_in, tokenizer, device=device, greedy= True, top_k=5)
        caption = tokenizer.decode(pred)
        cv2.imwrite(save_path, img)
        context = {
            "caption" : caption[0],
            "uploaded" : uploaded,
            "path " : f'/static/temp{id}.png',
            "id" : id
        }
        return render(request, "upload.html", context)
    return render(request, "upload.html")