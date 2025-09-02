import os
from dotenv import load_dotenv
from datetime import timedelta

load_dotenv() 
# For Development Alone!!!!!!!!!!!!!!!!
class Config:
    POSTGRES_USER = os.environ.get('POSTGRES_USER')
    POSTGRES_PASS = os.environ.get('POSTGRES_PASS')
    POSTGRES_URL = os.environ.get('POSTGRES_URL')
    SECRET_KEY = os.environ.get('SECRET_KEY')

    CLIENT_ID = os.environ.get("CLIENT_ID")
    CLIENT_SECRET = os.environ.get("CLIENT_SECRET")
    EXTS = set(['png', 'jpg', 'jpeg'])
    SQLALCHEMY_DATABASE_URI = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASS}@{POSTGRES_URL}"
    # SQLALCHEMY_DATABASE_URI ="sqlite:///test"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    CACHE_TYPE = "SimpleCache"
    GOOGLE_DISCOVERY_URL = (
    "https://accounts.google.com/.well-known/openid-configuration")

    CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL")
    CELERY_RESULT_BACKEND = os.environ.get('CELERY_RESULT_BACKEND')
    
    CLOUDINARY_API_SECRET = os.environ.get('CLOUDINARY_API_SECRET')
    CLOUDINARY_CLOUD_NAME = os.environ.get('CLOUDINARY_CLOUD_NAME')
    CLOUDINARY_API_KEY = os.environ.get('CLOUDINARY_API_KEY')
    CLOUDINARY_WEBHOOK_SECRET = os.environ.get('CLOUDINARY_WEBHOOK_SECRET')
    
    TICKI_URL = os.environ.get("TICKI_URL")
    
    # SPA Mode Configuration
    # Set to True for Single Page Application mode, False for Server-Side Rendering
    APP_MODE = os.environ.get('APP_MODE')