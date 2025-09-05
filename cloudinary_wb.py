import os
import ast
import sys
import json
import hmac
import hashlib
import logging
import requests
import cloudinary
from pathlib import Path
from datetime import datetime
from flask_restful import Resource, Api
from flask import Flask, request
from app.tasks import fe
from app.embeddings import embeddings_handler
# Setup paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import Config

# Flask app and API
app = Flask(__name__)
api = Api(app)

# Logging config
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Cloudinary config
cloudinary.config(
    cloud_name=Config.CLOUDINARY_CLOUD_NAME,
    api_key=Config.CLOUDINARY_API_KEY,
    api_secret=Config.CLOUDINARY_API_SECRET
)

def verify_upload_signature(body: str, timestamp: str, signature: str) -> bool:
    """Manual SHA1 verification for upload notifications following Cloudinary spec"""
    try:
        signed_payload = body + timestamp + Config.CLOUDINARY_API_SECRET
        expected_signature = hashlib.sha1(signed_payload.encode("utf-8")).hexdigest()
        return hmac.compare_digest(expected_signature, signature)
    except Exception as e:
        logger.error(f"Signature verification error: {e}")
        return False

def download_img(payload: dict):
    secure_url = payload.get("secure_url")
    original_filename = payload.get("original_filename", f"download_{datetime.utcnow().timestamp()}")
    extension = payload.get("format", "jpg")

    if secure_url:
        try:
            response = requests.get(secure_url)
            response.raise_for_status()
            filepath = os.path.join("", f"{original_filename}.{extension}")
            with open(filepath, "wb") as f:
                f.write(response.content)
            logger.info(f"Downloaded file to: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to download image: {e}")
    else:
        logger.warning("No secure_url in webhook payload.")

def compare_and_return(img_path: str):
    
    try: 
        query_face_paths_str = fe.extract_faces(img_path)
        query_face_paths = ast.literal_eval(query_face_paths_str)

        #handle input img w/multiple faces
        if not query_face_paths:
            logger.warning("No face found in uploaded image.")
            return {"error": "No face detected in input image"}, 400

        if len(query_face_paths) > 1:
            logger.warning("Multiple faces found in uploaded image.")
            return {"error": "Input image should contain only one face"}, 400

        query_face_path = Path(query_face_paths[0])
        query_feature = fe.extract_features(query_face_path).astype(float)
        results = embeddings_handler.get_similar_faces(query_feature)
        
        img_name = Path(img_path).name
        
        payload = {
            "status": "success",
            "user_url": img_name,
            "found_url": [
                {"score": round(score, 4), "img_name": img_name}
                for score, img_name in results
            ]
        }
    
        try:
            resp = requests.post(
                Config.TICKI_URL,
                json=payload,
                timeout=10
            )
            resp.raise_for_status()
            logger.info("✅ Forwarded payload to discovery API.")
        except Exception as e:
                logger.error(f"❌ Failed to forward payload to discovery API: {e}")
                
        return payload, 200
    
    except Exception as e:
        logger.error(f"Error comparing face: {e}")
        return {"error": "Internal Server Error"}, 500
    
        
class CloudinaryWebhook(Resource):
    
    def post(self):
        query_param = request.args.get()
        if query_param:
            pass
            return
        
        signature = request.headers.get("X-Cld-Signature")
        timestamp = request.headers.get("X-Cld-Timestamp")

        if not signature or not timestamp:
            logger.warning("Missing signature or timestamp headers.")
            return {"error": "Missing signature or timestamp headers"}, 400

        raw_body = request.get_data(as_text=True)

        try:
            payload = json.loads(raw_body)
        except json.JSONDecodeError:
            logger.error("Invalid JSON body.")
            return {"error": "Invalid JSON body"}, 400

        if payload.get("notification_type") != "upload":
            logger.info("Ignored non-upload notification.")
            return {"status": "ignored"}, 200

        if not verify_upload_signature(raw_body, timestamp, signature):
            logger.warning("Invalid webhook signature.")
            return {"error": "Invalid signature"}, 403

        logger.info("✅ Verified Cloudinary Webhook.")
        filepath = download_img(payload)  # Return this from download_img
        if filepath:
            return compare_and_return(Path(filepath))
        else:
            return {"error": "Failed to download image"}, 500
        # return {"status": "received"}, 200

class CloudinaryUpload(Resource):
    def post(self):
        try:
            file_path = request.json.get("file_path")
            if not file_path or not os.path.exists(file_path):
                logger.warning("Invalid or missing file_path.")
                return {"error": "Invalid or missing file_path"}, 400

            response = cloudinary.uploader.upload(file_path)
            logger.info(f"Uploaded: {response['secure_url']}")
            return {"uploaded": response}, 200
        except Exception as e:
            logger.error(f"Upload error: {e}")
            return {"error": str(e)}, 500

# Register endpoints
api.add_resource(CloudinaryWebhook, "/cloudinary-webhook")
api.add_resource(CloudinaryUpload, "/upload")

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)