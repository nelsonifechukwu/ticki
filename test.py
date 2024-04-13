import cv2 as cv
import numpy as np
import os
import argparse
import pandas as pd
import pickle
from deepface import DeepFace
from deepface.basemodels import VGGFace
from retinaface import RetinaFace
import matplotlib.pyplot as plt
import face_recognition


## Get face, dfsd, face-recognition, deepface
class faceDetails:
    img_deets = {}

    def __init__(self, encoding, pic_name):
        self.encoding = encoding
        self.pic_name = pic_name
        faceDetails.img_deets[encoding] = pic_name

location = "faces/Screenshot 2024-04-12 at 13.51.04.png"
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

showwith_retina()

#extract the faces
def extract_faces():
    faces = RetinaFace.extract_faces(img_path=location, align=True)



# dfs = DeepFace.find(img_path="Screenshot 2024-04-12 at 15.50.37.png",
#                     db_path="faces", model_name="Dlib")

# print(dfs)
# df0 = dfs[0]
# df1=dfs[1]
# df0=pd.concat([df0, df1])
# # print(len(df0))
# for df in range(len(df0)):
#     img_path = df0.iloc[df]["identity"]
#     img = cv.imread(img_path)
#     pt1 = (df0.iloc[df]["target_x"],df0.iloc[df]["target_y"])
#     pt2 = (pt1[0]+df0.iloc[df]["target_w"],pt1[1]+df0.iloc[df]["target_h"])
#     cv.rectangle(img,pt1,pt2, color=(0,255,0), thickness=2 )
#     img = img[:,:,::-1]
#     plt.imshow(img)
#     plt.show()

#Extract all faces
def extract_faces():
    image = face_recognition.load_image_file(
        location)
    r_img = cv.imread(location)
    face_locations = face_recognition.face_locations(
        image, number_of_times_to_upsample=3, model="cnn")
    for location in face_locations:
        pt1 = (location[3], location[0])
        pt2 = (location[1], location[2])
        cv.rectangle(r_img, pt1, pt2, color = (0, 255, 0), thickness = 2)
    plt.imshow(r_img)
    plt.show()
