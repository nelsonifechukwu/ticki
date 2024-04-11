from flask import Blueprint

ticki = Blueprint('ticki', __name__)

@ticki.route('/')
def index():
	return "Hello World"