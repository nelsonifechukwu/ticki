from app import app
import numpy as np
from scipy.spatial import distance
from flask import request, render_template, make_response
from flask_restful import Resource, Api

import sys
sys.path.append("./")
from pathlib import Path
import ast

from .functions import fe
from .embeddings import embeddings_handler

all_face_embeddings, all_face_names = embeddings_handler.load_allfaces_embeddings(external=False)

api = Api(app)
class HomeResource(Resource):
    def get(self):
        return make_response(render_template("main.html"), 200)

    def post(self): 
        threshold = 0.67
        img_stream = request.files.get("query-img")
        _, query_img_path = fe.save_query_image(img_stream)

        query_face_paths_str = fe.extract_faces(query_img_path)
        query_face_paths = ast.literal_eval(query_face_paths_str)
        
        #this will be used in case of multiple faces during an upload. Select only one img for now
        query_face_path = Path(query_face_paths[-1])

        query_feature = fe.extract_features(query_face_path).astype(float)

        results = self._get_similar_faces(query_feature, threshold)
        ###################
        embeddings_handler.mark_as_processed(query_feature, query_img_path, query_face_paths)
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

api.add_resource(HomeResource, "/", endpoint="index") 
