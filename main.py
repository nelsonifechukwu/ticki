from app import app
from app import routes

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=80, debug=True, use_reloader=True)
