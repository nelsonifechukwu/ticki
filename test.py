import cv2 as cv
import torch
import numpy as np
import os
import argparse
import pandas as pd
import pickle
import dlib
from deepface import DeepFace
from deepface.basemodels import VGGFace
from retinaface import RetinaFace
import matplotlib.pyplot as plt
import face_recognition
from PIL import Image

# from RealESRGAN import RealESRGAN
# from super_image import EdsrModel, ImageLoader

#testing several options, arcface, mtcnn
#use the average of few correct pictures since the setting never changes



def showwith_retina():
    location = "a2342e6d-a248-497a-ae02-5eacd85213be.JPG"
    faces = RetinaFace.detect_faces(img_path=location)
    r_img = cv.imread(location)
    for face in faces:
        place = faces[f"{face}"][f"facial_area"]
        pt1 = (place[0], place[1])
        pt2 = (place[2], place[3])
        cv.rectangle(r_img, pt1, pt2, color=(0, 255, 0), thickness=2)
    plt.imshow(r_img)
    plt.show()

# showwith_retina()


            
#automatically detect and align images
#upscale images with ESRGAN
def compare_pics(database:str, test_input:str):
    all_pics = os.listdir(database)
    for i, pic in enumerate(all_pics):
        location = f"{database}/{pic}"
        faces = RetinaFace.extract_faces(img_path=location, align=True, expand_face_area= 20)        
        for  face in faces:
            if face.any():
                img = Image.fromarray(face)
                img.save('f.png')
                result = DeepFace.verify('f.png',
                            test_input, model_name="ArcFace", normalization="ArcFace", distance_metric="euclidean_l2", detector_backend="mtcnn", enforce_detection=False)
                print(test_input, location, result["verified"])
                plt.imshow(face)
                plt.show(block=False)
                plt.pause(1)
                plt.close()
compare_pics('girl', 'girl/test.JPG')

