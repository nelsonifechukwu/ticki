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

    def extract_faces(self, img_input, save=False):
        """
        Extract faces from either file path (str) or image bytes.
        Returns list of PIL Images (if save=False) or list of file paths (if save=True)
        """
        try:
            # Handle different input types
            if isinstance(img_input, bytes):
                from io import BytesIO
                img_pil = Image.open(BytesIO(img_input)).convert("RGB")
                img_array = np.array(img_pil)
                img_name = f"memory_image_{datetime.now().timestamp()}"
                img_path = None  # No path for bytes input
            elif isinstance(img_input, str):
                img_path = Path(img_input)
                img_array = str(img_path)
                img_name = img_path.name
            else:
                raise ValueError("img_input must be either string path or bytes")

            # Extract faces using RetinaFace
            faces = RetinaFace.extract_faces(
                img_path=img_array,
                align=True,
                expand_face_area=30,
            )
            
            if not faces:
                if img_path:
                    self._mark_as_failed(img_path, "No faces detected")
                raise Exception("No faces detected in image")

            if len(faces) == 1 and any(x == 0 for x in faces[0].shape):
                if img_path:
                    self._mark_as_failed(img_path, "Face extraction failed")
                raise Exception("Face extraction failed - couldn't extract detected faces")

            # Process extracted faces
            result = []
            for i, face in enumerate(faces):
                if face.any():
                    if save and img_path:
                        # Save to disk (original behavior)
                        face_img = Image.fromarray(face.astype("uint8")).convert("RGB")
                        face_img = face_img.resize((224, 224))
                        face_filename = f"{img_path.stem}_face_{i}{img_path.suffix}"
                        face_filepath = self.extracted_faces_path / face_filename
                        face_img.save(face_filepath)
                        result.append(str(face_filepath))
                    else:
                        # Face is already a numpy array from RetinaFace
                        # Use cv2 for direct numpy array resizing (much faster)
                        face_resized = cv2.resize(face.astype("uint8"), (224, 224))
                        result.append(face_resized)
                else:
                    logger.info(f"Some faces in {img_name} couldn't be extracted")
            
            return str(result) if save and img_path else result
            
        except Exception as e:
            error_msg = f"Error processing {img_name if 'img_name' in locals() else 'image'}: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)

    def extract_features(self, face_input: Union[str, Image.Image, np.ndarray, bytearray, bytes], save=False) -> np.ndarray:
        """
        Extract features from file path (str), PIL Image, numpy array, or bytearray.
        """
        try:
            # Handle different input types
            if isinstance(face_input, np.ndarray):
                # Face array already processed (most efficient path)
                arr = face_input.astype("uint8")  # RGB, uint8, shape (224, 224, 3)
                face_name = f"memory_face_array_{datetime.now().timestamp()}"
                face_path = None
            elif isinstance(face_input, (bytearray, bytes)):
                from io import BytesIO
                face_img = Image.open(BytesIO(face_input)).convert("RGB").resize((224, 224))
                arr = np.array(face_img)
                face_name = f"memory_face_bytes_{datetime.now().timestamp()}"
                face_path = None
            elif isinstance(face_input, Image.Image):
                face_img = face_input.convert("RGB").resize((224, 224))
                arr = np.array(face_img)
                face_name = f"memory_face_pil_{datetime.now().timestamp()}"
                face_path = None
            elif isinstance(face_input, str):
                face_path = Path(face_input)
                face_img = Image.open(face_path).convert("RGB").resize((224, 224))
                arr = np.array(face_img)
                face_name = face_path.name
            else:
                raise ValueError("face_input must be string path, PIL Image, numpy array, or bytearray")

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

            if save and face_path:
                embeddings_path = self.extracted_faces_embeddings_path / face_path.name
                feature_path = embeddings_path.with_name(embeddings_path.name + ".npy")
                np.save(feature_path, embedding)
            
            return embedding

        except Exception as e:
            error_msg = f"Error generating embedding for {face_name if 'face_name' in locals() else 'face'}: {str(e)}"
            raise Exception(error_msg)

    def save_query_image(self, file: FileStorage) -> Tuple[Image.Image, Path]:
        import hashlib
        from io import BytesIO

        try:
            file_bytes = file.stream.read()
        except Exception as e:
            raise Exception(f"Can't open file: {e}")

        file_hash = hashlib.sha256(file_bytes).hexdigest()
        file_ext = Path(file.filename).suffix or ".png"
        new_filename = f"{file_hash}{file_ext}"

        uploaded_img_path = self.img_data / new_filename

        img = Image.open(BytesIO(file_bytes)) 
        if uploaded_img_path.exists():
            logger.warning(f"Image {new_filename} already exists.")
            return img, uploaded_img_path

        try:
            img.save(uploaded_img_path)
        except Exception as e:
            raise Exception(f"Failed to save image: {e}")

        return img, uploaded_img_path