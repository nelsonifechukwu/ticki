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
from PIL import Image
import warnings
import os
import shutil
import h5py
from typing import List

database = Path("app/static/database")

#--------Tweakable GPU options-------#
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true' 
# export MPS_ENABLE_GROWTH=1 MPS_GRAPH_COMPILE_TIMEOUT=30 MPS_MEMORY_LIMIT=4096
# pkill redis-server && export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# disable mps backend which doesn't allow certain tensor ops
os.environ["TF_MPS_ENABLED"] = "0"
# tf.config.set_visible_devices([], 'GPU')

class ImageProcessor:
    base_model = VGG16(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
    
    def __init__(self, database:Path):
        #database contains img_repo, extracted_faces, extracted_face_embeddings, failed_extractions, log.txt
        self.database = database
        self.img_repo = self.database / "img_repo" 
        self.img_data = self.img_repo / "img_data"
        self.extracted_faces_path = self.img_repo / "extracted_faces"
        self.extracted_faces_embeddings_path = self.img_repo / "extracted_faces_embeddings"
        self.failed_extractions_path = self.img_repo /  "failed_face_extractions_imgs"
        
        all_paths = [self.database, self.img_repo, self.img_data, self.extracted_faces_path, self.extracted_faces_embeddings_path, self.failed_extractions_path,]
        
        self.initialize_paths(all_paths)
        
        #Logger for Initialization errors
        self.logger_path = self.failed_extractions_path / "log.txt"
        #embedding store
        self.embeddings_store = self.database /  "embeddings_info.hdf5"
    def logger_write(self, msg:str):
        with open(self.logger_path, 'a') as f:
            f.write(msg + "\n")
        
    def initialize_paths(self, all_paths):
        try:
            for path in all_paths:
                path.mkdir(parents=True, exist_ok=True)
        except:
            # self.logger_write(f"Paths initialization failed. Confirm that {self.database} is a Path")
            raise Exception("Paths initialization failed.")
    
    def _remove_existing_failed_img(self, img_path):
        destination = self.failed_extractions_path / img_path.name
        # Delete existing file if it exists
        if destination.exists():
            destination.unlink() 
            
    def _mark_as_failed(self, img_path: Path, reason: str) -> None:
        self._remove_existing_failed_img(img_path)
        shutil.move(str(img_path), self.failed_extractions_path)
        self.logger_write(f"{reason} in {img_path.name}")
    
    def extract_faces(self, img_path: str):
        """Multiprocessing-safe face extraction using RetinaFace."""
        #check if img_path is a directory
        img_path = Path(img_path)

        try:
            faces = RetinaFace.extract_faces(
                img_path=str(img_path),
                align=True,
                expand_face_area=30,
            )
            if not faces:
                self._mark_as_failed(img_path, "No faces detected")
                raise Exception("No faces detected in image")

            if len(faces) == 1 and any(x == 0 for x in faces[0].shape): #if faces array contains 0 row/column/channel, terminate since it's a wrong representation of an img
                self._mark_as_failed(img_path, "Face extraction failed")
                raise Exception("Face extraction failed - couldn't extract detected faces")
                
            faces_path=[]
            for i, face in enumerate(faces):
                if face.any():
                    img = Image.fromarray(face.astype("uint8")).convert("RGB")
                    img = img.resize((224, 224))  # Resize for consistency
                    #img upscale
                    face_filename = f"{img_path.stem}_face_{i}{img_path.suffix}" #suffix adds the leading dot of the extension
                    face_filepath = self.extracted_faces_path / face_filename
                    img.save(face_filepath)
                    faces_path.append(face_filepath)
                else:
                    print (f"Some faces in {img_path.name} couldn't be extracted")
            return str(faces_path[len(faces_path)-1])

            #this would come in handy during the extraction of multiple faces in 1 upload
            #return str: celery requires a string (which is JSON serializable) not a PosixPath

        except Exception as e:
            print(f"Error processing {img_path.name}: {str(e)}")
            raise
              # Re-raise exception for Celery task tracking
              
    def extract_features(self, face_path: str):

        """
        Extract facial embeddings using DeepFace's FaceNet model.
        Args:
            face_path (str): Path to the image of a face.
        Returns:
            np.ndarray: Normalized face embedding.
        """
        face_path = Path(face_path)
        face_img= Image.open(face_path)
        face_img = face_img.resize((224, 224)) 
        face_img = image.img_to_array(face_img) #to ndarray

        try:
            # Use DeepFace to represent the image using FaceNet
            embedding_obj = DeepFace.represent(
                img_path=face_img,
                model_name="Facenet512",
                enforce_detection=False
            )[0]  # result is a list of dicts

            embedding = np.array(embedding_obj["embedding"])
            embedding = embedding / np.linalg.norm(embedding)  # normalize

            # Save embedding to .npy
            embeddings_path = self.extracted_faces_embeddings_path / face_path.name

            feature_path = embeddings_path.with_name(embeddings_path.name + ".npy")
   
            np.save(feature_path, embedding)
            return embedding

        except Exception as e:
            raise Exception(f"Error generating embedding for {face_path}: {str(e)}")
    
    def _check_embedding_store(self):
        if not self.embeddings_store.exists():
            raise ValueError("No external embedding store available")
    def _read_embeddings_store(self):
        self._check_embedding_store()
        with h5py.File(self.embeddings_store, 'r') as file:
                features = file['embeddings'][:]
                img_paths = [path.decode('utf-8') for path in file['img_paths'][:]]
        return features, img_paths
    
    def _write_to_embeddings_store(self, features: np.ndarray, img_paths: List[Path]):
        with h5py.File(self.embeddings_store, 'w') as file:
            file.create_dataset('embeddings', data=features)
            dt = h5py.string_dtype(encoding='utf-8')
            file.create_dataset('img_paths', data=[str(p) for p in img_paths], dtype=dt)
    
    def append_to_embedding_store(self, query_feature, query_img_path):
        self._check_embedding_store()
        with h5py.File(self.embeddings_store, 'r') as file:
            features = file['embeddings'][:]
            img_paths = [path.decode('utf-8') for path in file['img_paths'][:]]
    
        query_img_path_str = str(query_img_path)
      # Skip if already in store
        if query_img_path_str in img_paths:
            print(f"{query_img_path_str} already exists in embedding store. Skipping append.")
            return
        features = np.vstack([features, query_feature])    
        img_paths.append(query_img_path_str)
        
        with h5py.File(self.embeddings_store, 'w') as file:
                file.create_dataset('embeddings', data=features)
                dt = h5py.string_dtype(encoding='utf-8')
                file.create_dataset('img_paths', data=img_paths, dtype=dt)
    
    def load_allfaces_embeddings(self, external=None): 
        #load external embeddings
        if external:
            try: 
                return self._read_embeddings_store()
            except ValueError as e:
                raise
  
        features = []
        img_paths = []
        base_path = Path("app/static")
        for feature_path in self.extracted_faces_embeddings_path.glob("*.npy"):
            features.append(np.load(feature_path))
            # From 'IMG_3011_face_0.JPG.npy' to
            img_ext = Path(feature_path.stem).suffix  # '.JPG'
            img_name = feature_path.stem.split("_face")[0]  # 'IMG_3011'
            img_paths.append(self.img_data.relative_to(base_path) / (img_name + img_ext)) #get the reference img of the face

        features = np.array(features, dtype=object).astype(float)
        self._write_to_embeddings_store(features, img_paths)
        return features, img_paths
    
    def save_query_image(self, file):
        try:
            img = Image.open(file.stream)  # Load image using PIL
        except Exception as e:
            raise Exception(f"Can't open file: {e}")

        uploaded_img_path = self.img_data / file.filename

        if uploaded_img_path.exists():
            # warnings.warn(f"Warning: Image '{file.filename}' already exists.", UserWarning)  # Log warning
            print(f"Image {file.filename} already exists.")
            
            return img, uploaded_img_path
        
        # Attempt to save the image
        try:
            img.save(uploaded_img_path)
        except Exception as e:
            raise Exception(f"Failed to save image: {e}")

        return img, uploaded_img_path
