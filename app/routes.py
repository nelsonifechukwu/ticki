# implement several features:
"""
clean dir_initialization in class ImgProcessor -> specify directory once
check if input img contain multiple faces and reject/search for multiple faces
monitor celery processing and indicate on flask app -> wait till all images are processed
monitor end of celery processing and convert face to embeddings
select only similar images
track and run embeddings code when new image is added (like git) -> threads or check if no of new list > old list
implement hash map to group similar face embeddings to improve search, If A=B & B=C, then, A=C. Wow, I thought of a hashmap w/o knowing it was a hash map!
Processing a lot of requests -> divide the no of images in the gdrive and processes requests asyc to compare or just download all of em and group their embeddings.
Dockerize everything 
"""
from flask import request, render_template, flash, url_for
from app import app
import sys

sys.path.append("./")
from datetime import datetime
import numpy as np
from PIL import Image
from pathlib import Path
from .cbir import ImageProcessor
from scipy.spatial import distance

fe = ImageProcessor()
database = Path("app/static/database")
upload_directory = database / "uploads"
features, img_paths = fe.load_allfaces_embeddings()

img_repo = database / "img_repo"
img_repo_list = list(img_repo.iterdir())

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        img_stream = request.files.get("query-img")  # get query image
        # img_path = upload_directory  
        _, img_path = fe.save_query_image(img_stream)
        # Run search
        face_dir = Path(fe.extract_faces(img_path))
        query_feature = fe.extract_features(face_dir).astype(float)

        # L2 distances to features
        # dists = np.linalg.norm(features-query, axis=1)
        dists = list(map(lambda x: 1 - distance.cosine(x, query_feature), features))
        ids = np.argsort(dists)[:30]  # Top 30 results
        scores = [(dists[id], img_paths[id]) for id in ids]
        base_path = Path("app/static")
        query_path = Path(img_path).relative_to(base_path)
        return render_template("main.html", scores=scores) # query_path=query_path,
    else:
        return render_template("main.html")
