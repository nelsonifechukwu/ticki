from flask import request, render_template, make_response
from flask_restful import Resource, Api
from app import app
import sys
sys.path.append("./")
import numpy as np
from pathlib import Path
from scipy.spatial import distance
from .tasks import fe
from .functions import store_in_redis
from .embeddings import embeddings_store

all_face_embeddings, all_face_paths = embeddings_store.load_allfaces_embeddings(external=False)

api = Api(app)
class HomeResource(Resource):
    def get(self):
        return make_response(render_template("main.html"), 200)

    def post(self): 
        threshold = 0.67
        img_stream = request.files.get("query-img")
        _, query_img_path = fe.save_query_image(img_stream)

        query_face_path = Path(fe.extract_faces(query_img_path))
        query_feature = fe.extract_features(query_face_path).astype(float)

        store_in_redis([query_img_path, query_face_path])
        self._add_to_embedding_store(query_img_path, query_feature)

        results = self._get_similar_faces(query_feature, threshold)
        return make_response(render_template("main.html", file_info=results), 200)

    def _get_similar_faces(self, query_feature: np.ndarray, threshold: float):
        # if not all_face_embeddings:
        #     return None
        dists = [1 - distance.cosine(x, query_feature) for x in all_face_embeddings]
        ids = np.argsort([-x for x in dists])  # descending order
        return [
            (dists[i], all_face_paths[i])
            for i in ids
            if dists[i] >= threshold
        ]

    def _add_to_embedding_store(self, uploaded_img_path, query_feature):
        base_path = Path("app/static")
        uploaded_img_path = uploaded_img_path.relative_to(base_path)
        try:
            embeddings_store.append(query_feature, uploaded_img_path)
        except ValueError as e:
            print(e)

api.add_resource(HomeResource, "/", endpoint="index") 