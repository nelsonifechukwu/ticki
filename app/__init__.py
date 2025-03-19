from flask import Flask
from config import Config
from pathlib import Path
import os
from .tasks import extract_all_faces, redis_client

def create_app(config_class=Config, repeat_tasks=False):
    app = Flask(__name__)
    app.config.from_object(config_class)
    if repeat_tasks:
        redis_client.flushdb()
    
    with app.app_context():
        extract_all_faces(repeat_tasks) # Uses Celery to handle concurrency efficiently
    return app

app = create_app(repeat_tasks=True)
from app import routes
