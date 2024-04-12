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

model = VGGFace.load_model()
## Get face, dfsd, opencv, dlib, face-recognition, deepface
class faceDetails:
    img_deets = {}

    def __init__(self, encoding, pic_name):
        self.encoding = encoding
        self.pic_name = pic_name
        faceDetails.img_deets[encoding] = pic_name

# resp = RetinaFace.detect_faces("faces/Screenshot 2024-04-12 at 13.50.32.png")
# print(resp)

# faces = RetinaFace.extract_faces(img_path = "faces/Screenshot 2024-04-12 at 13.52.14.png", align = True)
# for face in faces:
#   plt.imshow(face)
#   plt.show()

dfs = DeepFace.find(img_path = "Screenshot 2024-04-12 at 15.50.37.png", db_path = "faces", threshold=0.5)
print(dfs)
df0 = dfs[0]
df1=dfs[1]
df0=pd.concat([df0, df1])
# print(len(df0))
for df in range(len(df0)):
    img_path = df0.iloc[df]["identity"]
    img = cv.imread(img_path)
    pt1 = (df0.iloc[df]["target_x"],df0.iloc[df]["target_y"])
    pt2 = (pt1[0]+df0.iloc[df]["target_w"],pt1[1]+df0.iloc[df]["target_h"])
    cv.rectangle(img,pt1,pt2, color=(0,255,0), thickness=2 )
    img = img[:,:,::-1]
    plt.imshow(img)
    plt.show()
# img_path = "faces/Screenshot 2024-04-12 at 13.49.29.png"
# img = cv.imread(img_path)

# cv.rectangle(img,(478,719),(478+294,719+294), color=(0,255,0), thickness=2 )
# img = img[:,:,::-1]
# plt.imshow(img)
# plt.show()
#   478       719       294       294