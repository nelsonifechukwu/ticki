import sys
sys.path.append("./")
import ast
import numpy as np 
from app import app
from typing import List
from pathlib import Path
from flask import request, render_template, make_response, jsonify
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
        # Serve the static HTML page for SPA
        return make_response(render_template("main_spa.html"), 200)

    def post(self):
        """API endpoint that returns JSON responses"""
        try:
            threshold = float(request.form.get('threshold', 0.67))
            img_stream = request.files.get("query-img")
            
            if not img_stream:
                return {"error": "No image file provided", "code": "NO_IMAGE"}, 400
            
            _, query_img_path = fe.save_query_image(img_stream)
            query_img_path_str = str(query_img_path) 
            
            query_face_paths_str = fe.extract_faces(query_img_path)
            query_face_paths = ast.literal_eval(query_face_paths_str)
            
            if not query_face_paths:
                return {"error": "No faces detected in the uploaded image", "code": "NO_FACES_DETECTED"}, 400
            
            # Handle multiple faces
            if len(query_face_paths) > 1:
                query_faces_names = [Path(face_path).name for face_path in query_face_paths]
                base_path = Path("app/static") 
                extracted_faces_path = fe.extracted_faces_path.relative_to(base_path)
                
                face_options = []
                for face_name in query_faces_names:
                    face_options.append({
                        "name": face_name,
                        "url": f"/static/{extracted_faces_path}/{face_name}"
                    })
                
                return {
                    "status": "multiple_faces",
                    "faces": face_options,
                    "message": f"Found {len(query_face_paths)} faces. Please select which faces to search for."
                }, 200
            
            # Single face processing
            query_face_path = Path(query_face_paths[0])
            query_feature = fe.extract_features(query_face_path).astype(float)
            results = self._get_similar_faces(query_feature, threshold)
            
            return {
                "status": "success",
                "results": self._format_results(results),
                "query_image": query_img_path_str,
                "faces_found": len(query_face_paths)
            }, 200
                
        except Exception as e:
            logger.error(f"Face search error: {str(e)}")
            return {"error": "Internal server error during face search", "code": "PROCESSING_ERROR", "details": str(e)}, 500

    def _get_similar_faces(self, query_feature: np.ndarray, threshold: float):
        """Get similar faces using FAISS for fast similarity search."""
        return embeddings_handler.get_similar_faces(query_feature, threshold)
    
    def _format_results(self, results: List[tuple]) -> List[dict]:
        """Format similarity results for API response"""
        base_path = Path("app/static") 
        img_data_path = fe.img_data.relative_to(base_path)
        
        formatted_results = []
        for similarity, img_name in results:
            formatted_results.append({
                "similarity": round(float(similarity), 4),
                "image_name": img_name,
                "image_url": f"/static/{img_data_path}/{img_name}",
                "confidence": "high" if similarity > 0.8 else "medium" if similarity > 0.6 else "low"
            })
        
        return formatted_results

class MultipleInputFacesResource(HomeResource):
    def get(self):
        return make_response(render_template("main_spa.html"), 200)
    
    def post(self):
        """API endpoint for multiple face search that returns JSON"""
        try:
            threshold = float(request.form.get('threshold', 0.67))
            selected_faces = request.form.getlist("selected_faces")

            if not selected_faces:
                return {"error": "No faces selected", "code": "NO_FACES_SELECTED"}, 400
            
            features: List[np.ndarray] = []
            for face_name in selected_faces:
                face_path = fe.extracted_faces_path / face_name
                if not face_path.exists():
                    logger.warning(f"Face file not found: {face_name}")
                    continue
                    
                query_feature = fe.extract_features(face_path).astype(float)
                features.append(query_feature)

            if not features:
                return {"error": "No valid face files found", "code": "NO_VALID_FACES"}, 400
            
            query_features = np.vstack(features).astype(np.float32)  
            results = self._get_similar_faces(query_features, threshold)
            
            # Remove duplicates (choose highest similarity) from multiple img comparisons 
            seen_names = {}
            for similarity, name in results: 
                if name not in seen_names or similarity > seen_names[name]:
                    seen_names[name] = similarity
            
            # Convert back to list of tuples, sorted by similarity
            unique_results = [(sim, name) for name, sim in seen_names.items()]
            unique_results.sort(key=lambda x: x[0], reverse=True)

            return {
                "status": "success",
                "results": self._format_results(unique_results),
                "faces_searched": len(selected_faces),
                "features_processed": len(features)
            }, 200
            
        except Exception as e:
            logger.error(f"Multiple face search error: {str(e)}")
            return {"error": "Internal server error during multiple face search", "code": "PROCESSING_ERROR", "details": str(e)}, 500

class SystemStatusResource(Resource):
    """API endpoint for system status and health checks"""
    
    def get(self):
        """Get system status and statistics"""
        try:
            # Get embedding store statistics
            if embeddings_handler.index and embeddings_handler.index.ntotal > 0:
                total_embeddings = embeddings_handler.index.ntotal
                index_type = embeddings_handler.index_type
                dimension = embeddings_handler.index.d
            else:
                total_embeddings = 0
                index_type = "not_initialized"
                dimension = 0
            
            # Get image repository statistics
            img_data_count = len(list(fe.img_data.glob("*"))) if fe.img_data.exists() else 0
            faces_count = len(list(fe.extracted_faces_path.glob("*"))) if fe.extracted_faces_path.exists() else 0
            
            return {
                "status": "healthy",
                "embedding_store": {
                    "total_embeddings": total_embeddings,
                    "index_type": index_type,
                    "dimension": dimension,
                    "status": "ready" if total_embeddings > 0 else "empty"
                },
                "image_repository": {
                    "total_images": img_data_count,
                    "extracted_faces": faces_count,
                    "repository_path": str(fe.img_data)
                },
                "supported_formats": ["jpg", "jpeg", "png", "gif", "bmp", "webp"],
                "api_version": "1.0"
            }, 200
            
        except Exception as e:
            logger.error(f"System status error: {str(e)}")
            return {
                "status": "error",
                "error": "Failed to get system status",
                "details": str(e)
            }, 500

# API setup
api = Api(app)
api.add_resource(HomeResource, "/", endpoint="index") 
api.add_resource(MultipleInputFacesResource, "/multiple-faces", endpoint="multiple-faces")
api.add_resource(SystemStatusResource, "/api/status", endpoint="system-status")