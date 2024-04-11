from flask import Flask
from .routes import ticki
from config import Config


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    app.register_blueprint(ticki)
    return app
