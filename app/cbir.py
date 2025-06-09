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
        self.extracted_faces_path = self.img_repo / "extracted_faces"
        self.extracted_faces_embeddings_path = self.img_repo / "extracted_faces_embeddings"
        self.failed_extractions_path = self.img_repo /  "failed_face_extractions_imgs"
        self.upload_directory = self.database / "upload_directory"
        
        all_paths = [self.database, self.img_repo, self.extracted_faces_path, self.extracted_faces_embeddings_path, self.failed_extractions_path, self.upload_directory]
        
        self.initialize_paths(all_paths)
        
        #Logger for Initialization errors
        self.logger_path = self.failed_extractions_path / "log.txt"
    
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
    
    @staticmethod    
    def extract_faces_database(database: Path):
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
                shutil.move(str(img_path), self.failed_extractions_path)
                self.logger_write(f"No faces detected in {img_path.name}")
                raise Exception("No faces detected in image")

            if len(faces) == 1 and any(x == 0 for x in faces[0].shape): #if faces array contains 0 row/column/channel, terminate since it's a wrong representation of an img
                destination = self.failed_extractions_path / img_path.name
                # Delete existing file if it exists
                if destination.exists():
                    destination.unlink()
                shutil.move(str(img_path), str(self.failed_extractions_path))
                self.logger_write(f"Face extraction from {img_path.name} failed")
                raise Exception("Face extraction failed - couldn't extract detected faces")
                
            
            faces_path=[]
            for i, face in enumerate(faces):
                if face.any():
                    img = Image.fromarray(face.astype("uint8")).convert("RGB")
                    img = img.resize((224, 224))  # Resize for consistency
                    #img upscale
                    face_filename = f"{img_path.stem}_face_{i}.png"
                    face_filepath = self.extracted_faces_path / face_filename
                    img.save(face_filepath)
                    faces_path.append(face_filepath)
                else:
                    print (f"Some faces in {img_path.name} couldn't be extracted")
            if faces_path:
                return str(faces_path[len(faces_path)-1])
            else:
                return str(0)
            #this would come in handy during the extraction of multiple faces in 1 upload
            #return str: celery requires a string (which is JSON serializable) not a PosixPath

        except Exception as e:
            print(f"Error processing {img_path.name}: {str(e)}")
            raise
              # Re-raise exception for Celery task tracking
              
    def extract_features(self, face_path: str):
        """
        Extract features from an Image
        Args: Path to the Image of a face
        """
        face_path = Path(face_path)
            # Convert the image to a NumPy array
        img = Image.open(face_path)
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
        
        embeddings_path = self.extracted_faces_embeddings_path / face_path.stem
        
        try:
            np.save(embeddings_path.with_suffix(".npy"), image_embedding)
            return image_embedding
        except:
            raise Exception(f"Error saving {face_path} embedding")
 
    def load_allfaces_embeddings(self):
        features = []
        img_paths = []
        base_path = Path("app/static")
        for feature_path in self.extracted_faces_embeddings_path.glob("*.npy"):
            features.append(np.load(feature_path))
            img_paths.append(self.extracted_faces_path.relative_to(base_path) / (feature_path.stem + ".png"))
        features = np.array(features, dtype=object).astype(float)
    
        return features, img_paths
    
    def save_query_image(self, file):
        try:
            img = Image.open(file.stream)  # Load image using PIL
        except Exception as e:
            raise Exception(f"Can't open file: {e}")

        uploaded_img_path = self.upload_directory / file.filename

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
