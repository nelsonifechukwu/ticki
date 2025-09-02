import os
from flask import Flask
from config import Config

def create_app(config_class=Config, reprocess=False):
    app = Flask(__name__)
    app.config.from_object(config_class)
    return app

app = create_app()

# Load routes based on SPA mode configuration

if Config.APP_MODE == "SPA":
    from app import routes_spa
    print("🚀 Ticki loaded in SPA mode")
elif Config.APP_MODE == "SSR":
    from app import routes  
    print("🚀 Ticki loaded in SSR mode") 

