from flask import request, render_template, flash, url_for
from app import app
import sys

sys.path.append("./")
from datetime import datetime
import numpy as np
from PIL import Image
from pathlib import Path
from scipy.spatial import distance
from .tasks import fe

database = fe.database
upload_directory = database / "uploads"

img_repo = fe.img_repo
allowed_exts = ("jpg", "png", "jpeg")
img_repo_list = [str(img) for img in img_repo.iterdir() if str(img).lower().endswith(allowed_exts)]

all_face_embeddings, all_face_paths = fe.load_allfaces_embeddings()

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
        dists = list(map(lambda x: 1 - distance.cosine(x, query_feature), all_face_embeddings)) 
        ids = np.argsort([-x for x in dists])[:30]  # Top 30 results (minus for sorting in descending order)
        file_info = [(dists[id], all_face_paths[id]) for id in ids]
        base_path = Path("app/static")
        query_path = Path(img_path).relative_to(base_path)
        return render_template("main.html", file_info=file_info) # query_path=query_path,
    else:
        return render_template("main.html")
