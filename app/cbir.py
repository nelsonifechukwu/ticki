import os
import cv2
import shutil
import logging
import coloredlogs
import numpy as np
from PIL import Image
from typing import Tuple, Union
from pathlib import Path
from datetime import datetime
from deepface import DeepFace
from retinaface import RetinaFace
from werkzeug.datastructures import FileStorage


logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s: %(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger)

#--------Tweakable GPU options-------#
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true' 
# export MPS_ENABLE_GROWTH=1 MPS_GRAPH_COMPILE_TIMEOUT=30 MPS_MEMORY_LIMIT=4096
# pkill redis-server && export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# disable mps backend which doesn't allow certain tensor ops
os.environ["TF_MPS_ENABLED"] = "0"
# tf.config.set_visible_devices([], 'GPU')
class ImageProcessor:
    def __init__(self, database:Path):
        self.database = database
        self.img_repo = self.database / "img_repo" 
        self.img_data = self.img_repo / "img_data"
        self.extracted_faces_path = self.img_repo / "extracted_faces"
        self.extracted_faces_embeddings_path = self.img_repo / "extracted_faces_embeddings"
        self.failed_extractions_path = self.img_repo /  "failed_face_extractions_imgs"

        all_paths = [self.database, self.img_repo, self.img_data, self.extracted_faces_path, self.extracted_faces_embeddings_path, self.failed_extractions_path,]

        self._initialize_paths(all_paths)

        self.logger_path = self.failed_extractions_path / "log.txt"

    def logger_write(self, msg:str):
        with open(self.logger_path, 'a') as f:
            f.write(msg + "\n")

    def _initialize_paths(self, all_paths):
        try:
            for path in all_paths:
                path.mkdir(parents=True, exist_ok=True)
        except:
            raise Exception("Paths initialization failed.")

    def _remove_existing_failed_img(self, img_path):
        destination = self.failed_extractions_path / img_path.name
        if destination.exists():
            destination.unlink() 

    def _mark_as_failed(self, img_path: Path, reason: str) -> None:
        self._remove_existing_failed_img(img_path)
        shutil.move(str(img_path), self.failed_extractions_path)
        logger.error(f"{reason} in {img_path.name}")

    def extract_faces(self, image_input):
        """
        Extract faces from bytes, file path, or numpy array. Returns list of numpy arrays.
        """
        try:
            # Handle different input types efficiently
            if isinstance(image_input, bytes):
                # Web response bytes
                from io import BytesIO
                img_pil = Image.open(BytesIO(image_input)).convert("RGB")
                img_array = np.array(img_pil)
            elif isinstance(image_input, str):
                # File path - direct load
                img_pil = Image.open(image_input).convert("RGB")
                img_array = np.array(img_pil)
            elif isinstance(image_input, np.ndarray):
                # Already a numpy array
                img_array = image_input
            else:
                raise ValueError("image_input must be bytes, file path string, or numpy array")

            # Extract faces using RetinaFace
            faces = RetinaFace.extract_faces(
                img_path=img_array,
                align=True,
                expand_face_area=30,
            )
            
            if not faces:
                raise Exception("No faces detected in image")

            if len(faces) == 1 and any(x == 0 for x in faces[0].shape):
                raise Exception("Face extraction failed - couldn't extract detected faces")

            # Process extracted faces - return numpy arrays only
            result = []
            for i, face in enumerate(faces):
                if face.any():
                    # Face is already a numpy array from RetinaFace
                    # Use cv2 for direct numpy array resizing (much faster)
                    face_resized = cv2.resize(face.astype("uint8"), (224, 224))
                    result.append(face_resized)
                else:
                    logger.info(f"Some faces couldn't be extracted")
            
            return result
            
        except Exception as e:
            error_msg = f"Error processing image: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)

    def extract_features(self, face_array: np.ndarray) -> np.ndarray:
        """
        Extract features from face numpy array only.
        """
        try:
            # Face array already processed (most efficient path)
            arr = face_array.astype("uint8")  # RGB, uint8, shape (224, 224, 3)

            # DeepFace expects BGR arrays; convert RGB -> BGR
            bgr = arr[:, :, ::-1]
            
            embedding_obj = DeepFace.represent(
                img_path=bgr,
                model_name="Facenet512",
                detector_backend="skip",
                enforce_detection=False
            )[0]

            embedding = np.array(embedding_obj["embedding"], dtype=np.float64)
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding

        except Exception as e:
            raise Exception(f"Error generating embedding from face array: {str(e)}")

