import numpy as np
from PIL import Image
from extractor import FeatureExtractor
from datetime import datetime
from flask import Flask, request, render_template
from pathlib import Path

app = Flask(__name__)
path = Path("ycbir/static/imgycbir")
pat = "ycbir/static/imgycbir"

def image_to_embeddings(path: str):
    for img in path.iterdir():
        # Convert the image to a NumPy array
        if img.suffix != ".npy":
            image_embedding = fe.extract(Image.open(img))
            # Save the array to a .npy file
            np.save(f"{path}/{img.stem}" + ".npy", image_embedding)

# image_to_embeddings(path)

# Read image features
fe = FeatureExtractor()
features = []
img_paths = []
for feature_path in path.glob("*.npy"):
    features.append(np.load(feature_path))
    img_paths.append(path.parts[-1] + "/"+ (feature_path.stem + ".png"))
features = np.array(features, dtype=object).astype(float)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['query_img']

        # Save query image
        img = Image.open(file.stream)  # PIL image
        uploaded_img_path = "ycbir/static/uploaded/" + datetime.now().isoformat().replace(":",
                                                                                          ".") + "_" + file.filename
        img.save(uploaded_img_path)

        # Run search
        query = fe.extract(img).astype(float)
        # L2 distances to features
        dists = np.linalg.norm(features-query, axis=1)
        ids = np.argsort(dists)[:30]  # Top 30 results
        scores = [(dists[id], img_paths[id]) for id in ids]
        base_path = Path("ycbir/static")

        return render_template('index.html',
                               query_path=Path(uploaded_img_path).relative_to(base_path),
                               scores=scores)
    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run("0.0.0.0", port=8080, debug=True)
