from flask import Flask
from config import Config
from pathlib import Path
from .tasks import extract_faces_batch

database = Path("app/static/database")
img_repo = database / "img-rep"
img_repo_list = list(img_repo.iterdir())
img_repo_list = [str(img) for img in img_repo_list]

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    with app.app_context():
        extract_faces_batch.delay(img_repo_list)  # Celery will handle concurrency efficiently
    
    return app

app = create_app()
from app import routes
