import sys
sys.path.append("./")
import ast
import numpy as np 
from app import app
from typing import List
from pathlib import Path
#from scipy.spatial import distance
from flask import request, render_template, make_response
from flask_restful import Resource, Api
from .tasks import fe 
from .cbir import logger
from .embeddings import embeddings_handler

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
        threshold = float(request.form.get('threshold', 0.67))    
        img_stream = request.files.get("query-img")
        faces = fe.extract_faces(img_stream)
        
        context = { 
            "input_faces": [],
            "similarity_info": [] 
        }    
        #handle input img w/multiple faces
        if len(faces) > 1:
            query_faces_names = [Path(face_path).name for face_path in query_face_paths]
            context["input_faces"] = query_faces_names
        else:
            query_feature = fe.extract_features(faces[0]).astype(float)
            results = self._get_similar_faces(query_feature, threshold)
            #embeddings_handler.mark_as_processed(query_feature, query_img_path_str, query_face_paths)
            context["similarity_info"] = results

        return make_response(render_template("main.html", **context), 200)    

    def _get_similar_faces(self, query_feature: np.ndarray, threshold: float):
        """Get similar faces using FAISS for fast similarity search."""
        return embeddings_handler.get_similar_faces(query_feature, threshold)
        
class MultipleInputFacesResource(HomeResource):
    def get(self):
        return make_response(render_template("main.html"), 200)
    
    def post(self):
        
        context = {   
            "input_faces": [],
            "similarity_info": [] 
        }  
        threshold = float(request.form.get('threshold', 0.67))
        selected_faces = request.form.getlist("selected_faces")

        if not selected_faces:
            return make_response(render_template("main.html", **context), 200)
        
        features: List[np.ndarray] = []
        for face_name in selected_faces:
            face_path = fe.extracted_faces_path / face_name
            query_feature = fe.extract_features(face_path).astype(float)
            features.append(query_feature)

        if not features:
            return make_response(render_template("main.html", **context), 200) 
        
        query_features = np.vstack(features).astype(np.float32)  
        results = self._get_similar_faces(query_features, threshold)
        
        #remove duplicates (choose highest similarity) from mutiple img comparisons 
        seen_names = {}
        unique_results = []
        
        for similarity, name in results: 
            if name not in seen_names or similarity > seen_names[name]:
                seen_names[name] = similarity
        
        # Convert back to list of tuples, sorted by similarity
        unique_results = [(sim, name) for name, sim in seen_names.items()]
        unique_results.sort(key=lambda x: x[0], reverse=True)

        context["similarity_info"] = unique_results
        return make_response(render_template("main.html", **context), 200)  
    
api = Api(app)
api.add_resource(HomeResource, "/", endpoint="index") 
api.add_resource(MultipleInputFacesResource, "/multiple-faces", endpoint="multiple-faces")