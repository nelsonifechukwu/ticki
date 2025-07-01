from app import app
import numpy as np
from scipy.spatial import distance
from flask import request, render_template, make_response
from flask_restful import Resource, Api

import sys
sys.path.append("./")
from pathlib import Path
import ast

from .tasks import fe 
from .embeddings import embeddings_handler

all_face_embeddings, all_face_names = embeddings_handler.load_allfaces_embeddings(external=False)

@app.context_processor 
def inject_paths():
    base_path = Path("app/static") 

    img_data_path = fe.img_data.relative_to(base_path)
    extracted_faces_path = fe.extracted_faces_path.relative_to(base_path)

    return {
        "img_data_path": str(img_data_path) + "/",
        "extracted_faces_path": str(extracted_faces_path) + "/"
    }
class HomeResource(Resource): 
    def get(self):
        return make_response(render_template("main.html"), 200)

    def post(self): 
        threshold = 0.67
        img_stream = request.files.get("query-img")
        _, query_img_path = fe.save_query_image(img_stream)
        query_img_path_str = str(query_img_path)
        
        query_face_paths_str = fe.extract_faces(query_img_path)
        query_face_paths = ast.literal_eval(query_face_paths_str)
        
        if len(query_face_paths)>1:
            query_faces_names = [Path(face_path).name for face_path in query_face_paths]
            return make_response(render_template("main.html", input_faces=query_faces_names), 200)
        #this will be used in case of multiple faces during an upload. Select only one img for now
        #store also the embeddings of the multiple faces marked as processed
        query_face_path = Path(query_face_paths[-1])

        query_feature = fe.extract_features(query_face_path).astype(float)

        results = self._get_similar_faces(query_feature, threshold)
        ###################
        embeddings_handler.mark_as_processed(query_feature, query_img_path_str, query_face_paths)
        ###################
        return make_response(render_template("main.html", file_info=results), 200)

    def _get_similar_faces(self, query_feature: np.ndarray, threshold: float):
        # if not all_face_embeddings:
        #     return None
        dists = [1 - distance.cosine(x, query_feature) for x in all_face_embeddings]
        ids = np.argsort([-x for x in dists])  # descending order
        return [
            (dists[i], all_face_names[i])
            for i in ids
            if dists[i] >= threshold
        ]
    
api = Api(app)
api.add_resource(HomeResource, "/", endpoint="index") 
