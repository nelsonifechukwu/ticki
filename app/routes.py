import sys
sys.path.append("./")
import time
import secrets
import numpy as np 
from app import app
from PIL import Image
from io import BytesIO
from typing import List
from pathlib import Path
#from scipy.spatial import distance
from flask_restful import Resource, Api
from flask import request, render_template, make_response, session
from .tasks import fe 
from .cbir import logger
from .embeddings import embeddings_handler

# Global variable to store current face arrays for multiple selection
current_faces = []
# Session-based face access tokens
face_sessions = {}

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
        
        try:
            # Convert FileStorage to bytes for processing
            img_bytes = img_stream.read()
            faces = fe.extract_faces(img_bytes)
        except Exception as e:
            logger.error(f"Error processing uploaded image: {e}")
            context = {"input_faces": [], "similarity_info": [], "error": "Failed to process image"}
            return make_response(render_template("main.html", **context), 200)
        
        context = { 
            "input_faces": [],
            "similarity_info": [] 
        }    
        #handle input img w/multiple faces
        if len(faces) > 1:
            global current_faces
            current_faces = faces  # Store for later use in MultipleInputFacesResource
            
            # Generate secure session token for face access
            face_token = secrets.token_urlsafe(32)
            session['face_token'] = face_token
            face_sessions[face_token] = {
                'timestamp': time.time(),
                'face_count': len(faces)
            }
            
            # Create face info for template (no PIL conversion needed here)
            face_info = []
            for i in range(len(faces)):
                face_info.append({
                    'index': i,
                    'name': f'Face {i+1}',
                    'url': f'/face_image/{i}?token={session["face_token"]}'
                })
            
            context["input_faces"] = face_info
            context["multiple_faces"] = True
        elif len(faces) == 1:
            query_feature = fe.extract_features(faces[0]).astype(float)
            results = self._get_similar_faces(query_feature, threshold)
            context["similarity_info"] = results
        else:
            context["error"] = "No faces detected in the uploaded image"

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
        global current_faces
        
        for face_selection in selected_faces:
            try:
                if face_selection.startswith('face_') and current_faces:
                    # Extract face index from selection (e.g., "face_0" -> 0)
                    face_idx = int(face_selection.split('_')[-1])
                    if face_idx < len(current_faces):
                        face_array = current_faces[face_idx]
                        query_feature = fe.extract_features(face_array).astype(float)
                        features.append(query_feature)
            except Exception as e:
                logger.error(f"Error processing face {face_selection}: {e}")

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

        # Clear current_faces and session after processing
        current_faces = []
        face_token = session.get('face_token')
        if face_token:
            face_sessions.pop(face_token, None)
            session.pop('face_token', None)
        
        context["similarity_info"] = unique_results
        return make_response(render_template("main.html", **context), 200)

class LoadImage(Resource):
    def get(self, face_idx):
        """Serve face image directly from memory as JPEG (with security validation)"""
        global current_faces
        
        try:
            # Security validation 1: Check token
            token = request.args.get('token')
            if not token or token not in face_sessions:
                logger.warning(f"Unauthorized face image access attempt for index {face_idx}")
                return "Unauthorized", 403
            
            # Security validation 2: Check session validity
            session_data = face_sessions[token]
            if time.time() - session_data['timestamp'] > 3600:  # 1 hour expiry
                face_sessions.pop(token, None)
                return "Session expired", 403
            
            # Security validation 3: Check face index bounds
            if face_idx >= session_data['face_count'] or face_idx < 0:
                logger.warning(f"Invalid face index {face_idx} for session {token[:8]}...")
                return "Invalid face index", 400
            
            # Security validation 4: Check if faces still exist in memory
            if face_idx >= len(current_faces):
                return "Face no longer available", 404
                
            face_array = current_faces[face_idx]
                
            # Convert BGR->RGB to fix bluish appearance (RetinaFace sometimes outputs BGR format)
            if len(face_array.shape) == 3 and face_array.shape[2] == 3:
                face_array = face_array[:, :, ::-1]  # BGR to RGB conversion
                
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(face_array.astype('uint8'))
            
            # Convert to bytes
            img_buffer = BytesIO()
            pil_image.save(img_buffer, format='JPEG', quality=85)
            img_buffer.seek(0)
            
            from flask import send_file
            return send_file(img_buffer, mimetype='image/jpeg')
                
        except Exception as e:
            logger.error(f"Error serving face image {face_idx}: {e}")
            return "Error serving image", 500
        
        except Exception as e:
            logger.error(f"Security error in face image serving: {e}")
            return "Access denied", 403
    
api = Api(app)
api.add_resource(HomeResource, "/", endpoint="index") 
api.add_resource(MultipleInputFacesResource, "/multiple-faces", endpoint="multiple-faces")
api.add_resource(LoadImage, "/face_image/<int:face_idx>")