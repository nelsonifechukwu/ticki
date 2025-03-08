from flask import Flask
from config import Config
from pathlib import Path
import os
from .tasks import extract_faces_batch

# database = Path(app.root_path) / "static" / "database"
# img_repo = database / "img_repo"
# img_repo_list = [str(img) for img in img_repo.iterdir()]

# Get the absolute directory of the current script (this file)
basedir = os.path.abspath(os.path.dirname(__file__))

# Construct absolute path to the database/img_repo directory
database = os.path.join(basedir, "static", "database")
img_repo = os.path.join(database, "img_repo")

# List image files
img_repo_list = [os.path.join(img_repo, img) for img in os.listdir(img_repo)]

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    with app.app_context():
        extract_faces_batch.delay(img_repo_list)  # Celery will handle concurrency efficiently
    
    return app

app = create_app()
from app import routes
