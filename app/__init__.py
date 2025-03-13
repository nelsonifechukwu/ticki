from flask import Flask
from config import Config
from pathlib import Path
import os
import redis
redis_client = redis.Redis(host='localhost', port=6379, db=1)
__all__ = ["redis_client"]
from .tasks import extract_faces_batch

# database = Path(app.root_path) / "static" / "database"
# img_repo = database / "img_repo"
# img_repo_list = [str(img) for img in img_repo.iterdir()]

# Get the absolute directory of the current script (this file)
basedir = os.path.abspath(os.path.dirname(__file__))

# Construct absolute path to the database/img_repo directory
database = os.path.join(basedir, "static", "database")
img_repo = os.path.join(database, "img_repo")
allowed_exts=("jpg", "png", "jpeg")

# List image files
img_repo_list = [os.path.join(img_repo, img) for img in os.listdir(img_repo) if img.endswith(allowed_exts)]

def create_app(config_class=Config, flush_rdb=None):
    app = Flask(__name__)
    app.config.from_object(config_class)
    if flush_rdb:
        redis_client.flushdb()
    with app.app_context():
        extract_faces_batch.delay(img_repo_list)  # Celery will handle concurrency efficiently
    
    return app

app = create_app(flush_rdb=True)
from app import routes
