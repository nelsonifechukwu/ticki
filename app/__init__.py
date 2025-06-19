from flask import Flask
from config import Config

def create_app(config_class=Config, reprocess=False):
    app = Flask(__name__)
    app.config.from_object(config_class)
    return app

app = create_app()
from app import routes
