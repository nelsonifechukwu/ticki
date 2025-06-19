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
from .functions import store_in_redis

all_face_embeddings, all_face_paths = fe.load_allfaces_embeddings()

@app.route("/", methods=["GET", "POST"])
def index():
    threshold = 0.67
    if request.method == "POST":
        img_stream = request.files.get("query-img")  # get query image
        # img_path = upload_directory  
        _, img_path, img_path_in_db = fe.save_query_image(img_stream)
        # Run search
        face_path = Path(fe.extract_faces(img_path))
        store_in_redis([img_path_in_db, face_path])
        query_feature = fe.extract_features(face_path).astype(float)
        # L2 distances to features
        # dists = np.linalg.norm(features-query, axis=1)
        dists = list(map(lambda x: 1 - distance.cosine(x, query_feature), all_face_embeddings)) 
        ids = np.argsort([-x for x in dists]) #[:30]  # Top 30 results (minus for sorting in descending order)
        file_info = [(dists[id], all_face_paths[id]) for id in ids if dists[id]>=threshold]
        base_path = Path("app/static")
        query_path = Path(img_path).relative_to(base_path)
        return render_template("main.html", file_info=file_info) # query_path=query_path,
    else:
        return render_template("main.html")


