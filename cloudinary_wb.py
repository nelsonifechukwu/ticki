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
from typing import Tuple
from datetime import datetime
from flask_restful import Resource, Api
from flask import Flask, request
from app.tasks import fe, process_and_store_image, compare_image_query
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

def download_img(payload: dict) -> Tuple[bytes, str]:
    """Download image directly to memory without saving to disk"""
    secure_url = payload.get("secure_url")
    original_filename = payload.get("original_filename", f"download_{datetime.utcnow().timestamp()}")
    
    if secure_url:
        try:
            response = requests.get(secure_url)
            response.raise_for_status()
            logger.info(f"Downloaded image to memory: {original_filename}")
            return response.content, original_filename
        except Exception as e:
            logger.error(f"Failed to download image: {e}")
            return None, None
    else:
        logger.warning("No secure_url in webhook payload.")
        return None, None

    
class CloudinaryWebhook(Resource):        
    def post(self):
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
        image_bytes, img_name = download_img(payload)
        if image_bytes:
            # Process image and store all face embeddings in FAISS
            result = process_and_store_image.delay(image_bytes, img_name)
            return {"status": "processing", "message": f"Started processing {img_name}", "task_id": result.id}, 202
        else:
            return {"error": "Failed to download image"}, 500
        # return {"status": "received"}, 200
        
class TickiGet(Resource):
    def get(self):
        args = request.args
        if Config.CLOUDINARY_API_SECRET != args["api_key"]:
            return {"error": "Forbidden Access - Bad API KEY"}, 403
        try: 
            img_url = args.get("url")
            if not img_url:
                return {"error": "Missing 'url' parameter"}, 400
            img_name = args.get("img_name", f"query_image_{datetime.now().timestamp()}")
            response = requests.get(img_url)
            response.raise_for_status()
            logger.info(f"Downloaded image to memory")
            return compare_image_query(response.content, img_name)
        except Exception as e:
            return {"error": str(e)}, 400
        

# Register endpoints
api.add_resource(CloudinaryWebhook, "/process_image")
api.add_resource(TickiGet, "/user_image")

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)