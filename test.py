import cv2 as cv
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


## Get face, dfsd, face-recognition, deepface
class faceDetails:
    img_deets = {}

    def __init__(self, encoding, pic_name):
        self.encoding = encoding
        self.pic_name = pic_name
        faceDetails.img_deets[encoding] = pic_name

location = "faces/Screenshot 2024-04-13 at 23.47.03.png"
def showwith_retina():
    faces = RetinaFace.detect_faces(img_path = location)
    r_img = cv.imread(location)
    for face in faces:
        place = faces[f"{face}"][f"facial_area"]
        pt1 = (place[0], place[1])
        pt2 = (place[2], place[3])
        cv.rectangle(r_img, pt1, pt2, color = (0, 255, 0), thickness = 2)
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

faces_to_embeddings()
