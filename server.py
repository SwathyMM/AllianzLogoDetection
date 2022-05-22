from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Optional

import cv2
import numpy as np

import torch
import base64
import random
from PIL import Image
import os


from yolov5 import Allianz_Logo_detect

app = FastAPI()
templates = Jinja2Templates(directory='templates')

model_selection_options = ['yolov5']
model_dict = {model_name: None for model_name in model_selection_options}  # set up model cache

colors = [tuple([random.randint(0, 255) for _ in range(3)]) for _ in range(100)]  # for bbox plotting

##############################################
#-------------GET Request Routes--------------
##############################################
@app.get("/")
def home(request: Request):
    '''
    Returns html jinja2 template render for home page form
    '''

    return templates.TemplateResponse('home.html', {
            "request": request,
            "model_selection_options": model_selection_options,
        })


##############################################
# ------------POST Request Routes--------------
##############################################
@app.post("/")
async def detect_via_web_form(request: Request,
                              file_list: str = Form(...),
                              model_name: str = Form(...),
                              img_size: int = Form(640)):

    results = Allianz_Logo_detect.main(file_list)
    print(results)

    return templates.TemplateResponse('show_results.html', {
        'request': request,
        'bbox_image_data_zipped': results,  # unzipped in jinja2 template
    })



##############################################
# --------------Helper Functions---------------
##############################################

if __name__ == '__main__':
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', default=8000)
    parser.add_argument('--precache-models', action='store_true',
                        help='Pre-cache all models in memory upon initialization, otherwise dynamically caches models')
    opt = parser.parse_args()

    if opt.precache_models:
        model_dict = {model_name: torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
                      for model_name in model_selection_options}

    app_str = 'server:app'  # make the app string equal to whatever the name of this file is
    uvicorn.run(app_str, host=opt.host, port=opt.port, reload=True)