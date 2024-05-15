from .cbir import ImageProcessor
from pathlib import Path
from glob import glob
import numpy as np
from PIL import Image

database = Path("app/static/database")
faces_directory = database / "faces"


def save_allfaces_embeddings():
    fe = ImageProcessor()
    path = database / "faces"
    files = [p for p in path.glob("*") if p.suffix.lower() in {".jpg", ".png"}]
    for img_path in files:
        # print(img_path)  # e.g., ./static/database/faces/xxx.jpg
        feature = fe.extract_features(img_path)
        # e.g., ./static/database/faces/xxx.npy
        feature_path = path / (img_path.stem + ".npy")
        np.save(feature_path, feature)


def load_allfaces_embeddings():
    features = []
    img_paths = []
    base_path = Path("app/static")
    for feature_path in faces_directory.glob("*.npy"):
        features.append(np.load(feature_path))
        img_paths.append(faces_directory.relative_to(base_path) / (feature_path.stem + ".png"))
    features = np.array(features, dtype=object).astype(float)
   
    return features, img_paths


def save_query_image(file):
    upload_directory = database / "uploads"
    try:
        img = Image.open(file.stream)  # PIL image
    except Exception as e:
        raise Exception("Can't open file: {}".format(str(e)))

    uploaded_img_path = upload_directory / file.filename

    # Save the image
    try:
        img.save(uploaded_img_path)
    except Exception as e:
        raise Exception("Failed to save image: {}".format(str(e)))

    return img, uploaded_img_path
