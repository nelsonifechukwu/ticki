#implement several features:
"""
check if database/uploads directory is there, else create one
select only similar images
better ui
track and run embeddings code when new image is added (like git) -> threads or check if no of new list > old list
implement hash map to group similar face embeddings to improve search, If A=B & B=C, then, A=C. Wow, I thought of a hashmap w/o
knowing it was a hash map!
Processing a lot of requests -> divide the no of images in the gdrive and processes requests asyc to compare or just download all of em and group their embeddings.
"""
from app import app
# Standard library imports

import sys
sys.path.append("./")
from datetime import datetime


# Third-party library imports
import numpy as np
from PIL import Image
from pathlib import Path

# Framework imports
from flask import request, render_template, flash

# Deep learning and computer vision imports
from .cbir import ImageProcessor
from .functions import extract_all_faces_features, process_and_save_query_image
from scipy.spatial import distance

database = Path("app/static/database")
fe = ImageProcessor()
features, img_paths = extract_all_faces_features()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        raw_img = request.files['query_img']  # get query image
        img, img_path = process_and_save_query_image(raw_img)
        # Run search
        query = fe.extract_features(img_path).astype(float)
        # L2 distances to features
        # dists = np.linalg.norm(features-query, axis=1)
        dists = list(map(
            lambda x: 1 - distance.cosine(x, query), features))
        ids = np.argsort(dists)[:30]  # Top 30 results
        scores = [(dists[id], img_paths[id]) for id in ids]
        base_path = Path("app/static")
        query_path = Path(img_path).relative_to(base_path)
        return render_template('index.html',
                               query_path=query_path,
                               scores=scores)
    else:
        return render_template('index.html')
    