from flask import Flask, request, jsonify
from flask_restful import Resource, Api
import requests
import os
import sys
import json
import hmac
import hashlib
from datetime import datetime
import cloudinary

# Append the config directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import Config

# Initialize Flask and Flask-RESTful
app = Flask(__name__)
api = Api(app)

# Cloudinary configuration
cloudinary.config(
    cloud_name=Config.CLOUDINARY_CLOUD_NAME,
    api_key=Config.CLOUDINARY_API_KEY,
    api_secret=Config.CLOUDINARY_API_SECRET
)

def verify_upload_signature(body: str, timestamp: str, signature: str) -> bool:
    """Manual SHA1 verification for upload notifications following Cloudinary spec"""
    try:
        signed_payload = body + timestamp + Config.CLOUDINARY_API_SECRET
        expected_signature = hashlib.sha1(signed_payload.encode('utf-8')).hexdigest()
        return hmac.compare_digest(expected_signature, signature)
    except Exception as e:
        print(f"Manual upload verification error: {e}")
        return False

def download_img(payload: dict):
    secure_url = payload.get("secure_url")
    original_filename = payload.get("original_filename", f"download_{datetime.utcnow().timestamp()}")
    extension = payload.get("format", "jpg")

    if secure_url:
        try:
            response = requests.get(secure_url)
            response.raise_for_status()
            filepath = os.path.join(".", f"{original_filename}.{extension}")
            with open(filepath, "wb") as f:
                f.write(response.content)
            print(f"✅ Downloaded file to: {filepath}")
        except Exception as e:
            print(f"❌ Failed to download image from Cloudinary: {e}")
    else:
        print("❌ No secure_url found in webhook payload.")

class CloudinaryWebhook(Resource):
    def post(self):
        # Extract signature and timestamp from headers
        signature = request.headers.get("X-Cld-Signature")
        timestamp = request.headers.get("X-Cld-Timestamp")

        if not signature or not timestamp:
            return {"error": "Missing signature or timestamp headers"}, 400

        # Get raw body and parsed JSON
        raw_body = request.get_data(as_text=True)
        try:
            payload = json.loads(raw_body)
        except json.JSONDecodeError:
            return {"error": "Invalid JSON body"}, 400

        # Ignore non-upload notifications
        if payload.get("notification_type") != "upload":
            return {"status": "ignored"}, 200

        # Verify signature
        if not verify_upload_signature(raw_body, timestamp, signature):
            return {"error": "Invalid signature"}, 403

        # Download image
        download_img(payload)

        print("✅ Verified Cloudinary Webhook:", payload)
        return {"status": "received"}, 200

class CloudinaryUpload(Resource):
    def post(self):
        try:
            file_path = request.json.get("file_path")
            if not file_path or not os.path.exists(file_path):
                return {"error": "Invalid or missing file_path"}, 400

            response = cloudinary.uploader.upload(file_path)
            return {"uploaded": response}, 200
        except Exception as e:
            return {"error": str(e)}, 500

# Register resources
api.add_resource(CloudinaryWebhook, "/cloudinary-webhook")
api.add_resource(CloudinaryUpload, "/upload")

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)