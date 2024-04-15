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

from RealESRGAN import RealESRGAN
from super_image import EdsrModel, ImageLoader

#testing several options, arcface, mtcnn
#use the average of few correct pictures since the setting never changes


# Get face, dfsd, face-recognition, deepface
class faceDetails:
    img_deets = {}

    def __init__(self, encoding, pic_name):
        self.encoding = encoding
        self.pic_name = pic_name
        faceDetails.img_deets[encoding] = pic_name


location = "faces/Screenshot 2024-04-13 at 23.47.03.png"


def showwith_retina():
    faces = RetinaFace.detect_faces(img_path=location)
    r_img = cv.imread(location)
    for face in faces:
        place = faces[f"{face}"][f"facial_area"]
        pt1 = (place[0], place[1])
        pt2 = (place[2], place[3])
        cv.rectangle(r_img, pt1, pt2, color=(0, 255, 0), thickness=2)
    plt.imshow(r_img)
    plt.show()


def find_face_encodings(image_path):
    # reading image
    image = cv.imread(image_path)
    # get face encodings from the image
    face_enc = face_recognition.face_encodings(image)
    # return face encodings
    return face_enc[0]


def faces_to_embeddings():

    faces = RetinaFace.extract_faces(img_path=location, align=True)
    for i, face in enumerate(faces):
        for f in faces[i+1:]:

            imga = Image.fromarray(face)
            imga.save('my.png')

            imgb = Image.fromarray(f)
            imgb.save('myf.png')
            result = DeepFace.verify(
                "myf.png", "my.png", model_name="DeepID", normalization="ArcFace", distance_metric="euclidean_l2", detector_backend="mtcnn", enforce_detection=False)
            print(result)
            plt.imshow(face)
            plt.imshow(f)
            plt.show()


# faces_to_embeddings()

def compare_nei():
    all_neis = os.listdir("nei")

    for i, nei in enumerate(all_neis):
        for element in all_neis[i+1:]:
            path1 = os.path.join("nei",nei) 
            path2 = os.path.join("nei",element)
            
            result = DeepFace.verify(path1,
                        path2, model_name="ArcFace", normalization="ArcFace", distance_metric="cosine", detector_backend="skip", enforce_detection=False)
            print(path1, path2,result["verified"])
            
def compare_obi():
    device = torch.device('cpu')
    model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=2)
    all_obis = os.listdir("obinna")
    model = RealESRGAN(device, scale=4)
    model.load_weights('weights/RealESRGAN_x4.pth', download=True)
    for i, obi in enumerate(all_obis):
        location = f"obinna/{obi}"
        faces = RetinaFace.extract_faces(img_path=location, align=True, expand_face_area= 30)
        for a, face in enumerate(faces):
            img = Image.fromarray(face)
            img.save('f.png')
            # Load image
            # path_to_image = 'f.png'
            # image = Image.open(path_to_image).convert('RGB')

            # Upscale image
            # sr_image = model.predict(image)

            # Save image
            # sr_image.save('f.png')
            result = DeepFace.verify('f.png',
                        'obinna/mobi.png', model_name="ArcFace", normalization="ArcFace", distance_metric="euclidean_l2", detector_backend="mtcnn", enforce_detection=False)
            print('obinna/obin.JPG', location, result["verified"])
            plt.imshow(face)
            plt.show(block=False)
            plt.pause(1)
            plt.close()
compare_obi()

