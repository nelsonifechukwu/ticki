from flask import Flask
from config import Config
# from .batch_img_process import convert_all_faces_to_embeddings

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    # convert_all_faces_to_embeddings()
    return app

app = create_app()
from app import routes




