# webhook_receiver.py
from flask import Flask, request, jsonify
import requests
from datetime import datetime
import sys
import os
import hmac
import hashlib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import Config
import cloudinary
cloudinary.config(
    cloud_name=Config.CLOUDINARY_CLOUD_NAME,
    api_key=Config.CLOUDINARY_API_KEY,
    api_secret=Config.CLOUDINARY_API_SECRET
)
app = Flask(__name__)

def verify_upload_signature(body, timestamp, signature):
    """Manual SHA1 verification for upload notifications following Cloudinary spec"""
    try:
        # Cloudinary signature formula: SHA1(body + timestamp + api_secret)
        signed_payload = body + timestamp + Config.CLOUDINARY_API_SECRET
        expected_signature = hashlib.sha1(signed_payload.encode('utf-8')).hexdigest()
        return hmac.compare_digest(expected_signature, signature)
    except Exception as e:
        print(f"Manual upload verification error: {e}")
        return False

 
@app.route('/cloudinary-webhook', methods=['POST'])
def cloudinary_webhook():
    # Get headers and raw body
    signature = request.headers.get("X-Cld-Signature")
    timestamp = request.headers.get("X-Cld-Timestamp")
        # Check for required headers
    if not signature or not timestamp:
        return jsonify({"error": "Missing signature or timestamp headers"}), 400

    raw_body = request.get_data().decode("utf-8")
    payload = request.get_json()

    # Check if this is an upload notification
    notification_type = payload.get("notification_type")
    if notification_type != "upload":
        return jsonify({"status": "ignored"}), 200

    # Use manual verification for upload events
    is_valid = verify_upload_signature(raw_body, timestamp, signature)
    
    if not is_valid:
        return jsonify({"error": "Invalid signature"}), 403
    
    #download the img
    download_img(payload)
    
    print("✅ Verified Cloudinary Webhook:", payload)
    return jsonify({"status": "received"}), 200

def download_img(payload):
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
            print(f"Downloaded file to: {filepath}")
        except Exception as e:
            print(f"Failed to download image from Cloudinary: {e}")
    else:
        print("No secure_url found in webhook payload.")

def upload(name):
    response = cloudinary.uploader.upload(name)
    print("Uploaded:", response)
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
