from flask import Flask
from config import Config
from pathlib import Path
import os
import redis
redis_client = redis.Redis(host='localhost', port=6379, db=1)
__all__ = ["redis_client"]
from .tasks import extract_all_faces

def create_app(config_class=Config, repeat_tasks=False):
    app = Flask(__name__)
    app.config.from_object(config_class)
    if repeat_tasks:
        redis_client.flushdb()
    with app.app_context():
        extract_all_faces(repeat_tasks) # Uses Celery to handle concurrency efficiently
    return app

app = create_app(repeat_tasks=False)
from app import routes
