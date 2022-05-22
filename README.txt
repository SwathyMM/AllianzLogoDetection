Allianz Logo Detection using Yolov5
-------------------------------------------

The Allianz Logo Detector is implemented in Yolov5 and deployed as web app using fast api. 

Training Data - 113 images with Allianz logo split in to val and train sets ,obtained from internet. Labelled using labelImg tool. Dataset is uploaded as a zip file - datasets.zip
Augmentation - Image augmentation is done in Yolov5using Albumentations. Blur, median blur, random gamma, random brightness contrast are specified in yolov5/utils/augmentations.py
               hsv, translation, rotation, scaling, mosaic are specified in yolov5/data/hyps/hyp.scratch_med.yaml
Training - train.py script available in yolov5
Trained model - yolov5/model_final.pt
Sample Output images - output folder

Steps
------------
1. Download and extract the repository
2. cd Allianz_Logo_Detect
3. Install dependencies - pip install -r requirements.txt 
4. augmentations - specify values in yolov5/utils/augmentations.py and yolov5/data/hyps/hyp.scratch_med.yaml
5. Unzip the dataset foler
6. Training - python yolov5/train.py --img 640 --batch 16 --epochs 60 --data AllianzLogo_1.yaml --weights yolov5s.pt
7. Fastapi webapp - python -m uvicorn server:app --reload
8. Go to http://127.0.0.1:8000/
9. Enter test image file/name in the UI. Press submit.
10. Cropped logo and output will be saved to the device

