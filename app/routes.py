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
    if request.method == "POST":
        return handle_post_request()
    return render_template("main.html")

def handle_post_request():
    threshold = 0.67
    img_stream = request.files.get("query-img")
    _, uploaded_img_path = fe.save_query_image(img_stream)

    face_path = extract_and_store_faces(uploaded_img_path)
    query_feature = fe.extract_features(face_path).astype(float)
    
    results = get_similar_faces(query_feature, threshold)
    return render_template("main.html", file_info=results)

def extract_and_store_faces(uploaded_img_path: Path) -> Path:
    face_path = Path(fe.extract_faces(uploaded_img_path))
    store_in_redis([uploaded_img_path, face_path])
    return face_path

def get_similar_faces(query_feature: np.ndarray, threshold: float):
    dists = [1 - distance.cosine(x, query_feature) for x in all_face_embeddings]
    ids = np.argsort([-x for x in dists])  # descending order

    base_path = Path("app/static")
    return [
        (dists[i], all_face_paths[i]) 
        for i in ids 
        if dists[i] >= threshold
    ]