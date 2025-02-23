from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
from deepface import DeepFace
from deepface.basemodels import VGGFace
from retinaface import RetinaFace
import os
from PIL import Image
import warnings

database = Path("app/static/database")
upload_directory = database / "uploads"
faces_upload_directory = upload_directory / "faces"
faces_directory = database / "faces"


# See https://keras.io/api/applications/ for details

class ImageProcessor:
    
    base_model = VGG16(weights='imagenet')
    model = Model(inputs=base_model.input,
                    outputs=base_model.get_layer('fc1').output)
    @staticmethod    
    def extract_faces(database: Path):
        """
        Extract faces from all images in a directory
        Args:
            database: path to the directory containing images
        """
        if not isinstance(database, Path):
            raise TypeError("The 'database' parameter must be a Path object.")
        if not database.exists():
            raise FileNotFoundError(f"The directory '{database}' does not exist.")
        
        faces_directory = database / "faces"
        faces_directory.mkdir(parents=True, exist_ok=True)
                        
        all_pics = Path(database).iterdir()
        for pic in all_pics:
            if pic.is_file() and pic.suffix.lower() in ['.png', '.jpg']:
                # Load the image
                img = mpimg.imread(pic)
                # plt.imshow(img)
                # plt.show(block=False)
                # plt.pause(10)
                faces = RetinaFace.extract_faces(img_path=str(pic), align=True, expand_face_area=20)
                for i, face in enumerate(faces):
                    if face.any():
                        img = Image.fromarray(face)
                        face_filename = f"{pic.stem}_{i}.png"
                        face_filepath = faces_directory / face_filename
                        img.save(face_filepath)
        return faces_directory
    
    @staticmethod    
    def extract_faces_thread(img_path: Path):
        faces_directory = img_path.parent / "faces"
        faces = RetinaFace.extract_faces(img_path=str(img_path), align=True, expand_face_area=20)
        for i, face in enumerate(faces):
            if face.any():
                img = Image.fromarray(face)
                face_filename = f"{img_path.stem}_face_{i}.png"
                face_filepath = faces_directory / face_filename
                img.save(face_filepath)
    
    @staticmethod          
    def extract_features(img_path: Path):
        """
        Extract features from an Image
        Args: Path to the Image of a face
        """
            # Convert the image to a NumPy array
        img = Image.open(img_path)
        img = img.resize((224, 224))  # VGG must take a 224x224 img as an input
        img = img.convert('RGB')  # Make sure img is color
        
        # To np.array. Height x Width x Channel. dtype=float32
        x = image.img_to_array(img)
        
        # (H, W, C)->(1, H, W, C), where the first elem is the number of img
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)  # Subtracting avg values for each pixel
        
        # Get the feature vector
        feature = ImageProcessor.model.predict(x)[0]  # (1, 4096) -> (4096, )
        image_embedding = feature / np.linalg.norm(feature)  # Normalize

        return image_embedding

    @staticmethod  
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

    @staticmethod  
    def load_allfaces_embeddings():
        features = []
        img_paths = []
        base_path = Path("app/static")
        for feature_path in faces_directory.glob("*.npy"):
            features.append(np.load(feature_path))
            img_paths.append(faces_directory.relative_to(base_path) / (feature_path.stem + ".png"))
        features = np.array(features, dtype=object).astype(float)
    
        return features, img_paths
    
    @staticmethod  
    def save_query_image(file):
        upload_directory = database / "uploads"
        upload_directory.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

        try:
            img = Image.open(file.stream)  # Load image using PIL
        except Exception as e:
            raise Exception(f"Can't open file: {e}")

        uploaded_img_path = upload_directory / file.filename

        if uploaded_img_path.exists():
            warnings.warn(f"Warning: Image '{file.filename}' already exists.", UserWarning)  # Log warning
            return img, uploaded_img_path
        
        # Attempt to save the image
        try:
            img.save(uploaded_img_path)
        except Exception as e:
            raise Exception(f"Failed to save image: {e}")

        return img, uploaded_img_path