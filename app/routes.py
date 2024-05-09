#implement several features:
"""
select only similar images
better ui
track and run embeddings code when new image is added (like git) -> threads or check if no of new list > old list
implement hash map to group similar face embeddings to improve search, If A=B & B=C, then, A=C. Wow, I thought of a hashmap w/o
knowing it was a hash map!
"""
from app import app
# Standard library imports

import sys
sys.path.append("./")
from datetime import datetime


# Third-party library imports
import cv2 as cv
import numpy as np
from PIL import Image
import pandas as pd
import pickle
import dlib
from pathlib import Path
import face_recognition

# Framework imports
from flask import request, render_template, flash

# Deep learning and computer vision imports
from .cbir import ImageProcessor
from scipy.spatial import distance


database = Path("app/static/database")
faces_directory = database / "faces"
upload_directory = database / "uploads" 

# Read image features
fe = ImageProcessor()
# process.extract_faces(database=Path.cwd() / "app" / "static" / "database")
fe.feature_extractor(faces=faces_directory)
features = []
img_paths = []
for feature_path in faces_directory.glob("*.npy"):
    features.append(np.load(feature_path))
    img_paths.append(faces_directory / (feature_path.stem + ".png"))
features = np.array(features, dtype=object).astype(float)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['query_img']  # Save query image
        uploaded_img_path = saveFile(file)
        

        # Run search
        query = fe.extract(img).astype(float)
        # L2 distances to features
        # dists = np.linalg.norm(features-query, axis=1)
        dists = list(map(
            lambda x: 1 - distance.cosine(x, query), features))
        ids = np.argsort(dists)[:30]  # Top 30 results
        scores = [(dists[id], img_paths[id]) for id in ids]
        base_path = database

        return render_template('index.html',
                               query_path=Path(uploaded_img_path).relative_to(base_path),
                               scores=scores)
    else:
        return render_template('index.html')
    
def saveFile(file):
    try:
        img = Image.open(file.stream)  # PIL image
    except:
        raise Exception(
            flash("Can't Open file. Insert another file.", "danger"))
    uploaded_img_path=str(upload_directory)+ "/" + datetime.now().isoformat().replace(":",".") + "_" + file.filename
    img.save(uploaded_img_path)
    return uploaded_img_path