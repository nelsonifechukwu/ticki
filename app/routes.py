from app import app

@app.route('/',  methods=['GET'])
def index():
    # Access the app context using the app object
    # Your route logic here
    return 'Hello World'
