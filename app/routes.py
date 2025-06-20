from flask import request, render_template
from app import app
import sys
sys.path.append("./")
import numpy as np
from pathlib import Path
from scipy.spatial import distance
from .tasks import fe
from .functions import store_in_redis

try:  
    # all_face_embeddings, all_face_paths = None
    all_face_embeddings, all_face_paths = fe.load_allfaces_embeddings(external=True)
except ValueError as e:
    print(e)
    
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        return handle_post_request()
    return render_template("main.html")

def handle_post_request():
    threshold = 0.67
    img_stream = request.files.get("query-img")
    _, query_img_path = fe.save_query_image(img_stream)

    query_face_path = extract_faces(query_img_path)
    query_feature = fe.extract_features(query_face_path).astype(float) 
    
    store_in_redis([query_img_path, query_face_path])
    add_to_embedding_store(query_img_path, query_feature)
    
    results = get_similar_faces(query_feature, threshold)
    return render_template("main.html", file_info=results) 

def extract_faces(query_img_path: Path) -> Path:
    face_path = fe.extract_faces(query_img_path)
    return Path(face_path)

def get_similar_faces(query_feature: np.ndarray, threshold: float):
    dists = [1 - distance.cosine(x, query_feature) for x in all_face_embeddings]
    ids = np.argsort([-x for x in dists])  # descending order

    base_path = Path("app/static")
    return [
        (dists[i], all_face_paths[i]) 
        for i in ids 
        if dists[i] >= threshold
    ]

def add_to_embedding_store(uploaded_img_path, query_feature): 
    base_path = Path("app/static")
    uploaded_img_path = uploaded_img_path.relative_to(base_path)    
    try:
        fe.embeddings_store.append(query_feature, uploaded_img_path)
    except ValueError as e:
        print(e) 