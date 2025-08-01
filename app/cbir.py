import os
import shutil
import logging
import coloredlogs
import numpy as np
from PIL import Image
from typing import Tuple
from pathlib import Path
from deepface import DeepFace
from retinaface import RetinaFace
from deepface.basemodels import VGGFace
from werkzeug.datastructures import FileStorage
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model

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
    base_model = VGG16(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

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

    def extract_faces(self, img_path: str)->str:
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

            if len(faces) == 1 and any(x == 0 for x in faces[0].shape):
                #if faces array contains 0 row/column/channel, terminate since it's a wrong representation of an img
                self._mark_as_failed(img_path, "Face extraction failed")
                raise Exception("Face extraction failed - couldn't extract detected faces")

            faces_path=[]
            for i, face in enumerate(faces):
                if face.any():
                    img = Image.fromarray(face.astype("uint8")).convert("RGB")
                    img = img.resize((224, 224))
                    face_filename = f"{img_path.stem}_face_{i}{img_path.suffix}"
                    face_filepath = self.extracted_faces_path / face_filename
                    img.save(face_filepath)
                    faces_path.append(str(face_filepath))
                else:
                    print (f"Some faces in {img_path.name} couldn't be extracted")
            return str(faces_path)
        except Exception as e:
            logger.error(f"Error processing {img_path.name}: {str(e)}")
            raise

    def extract_features(self, face_path: str) -> np.ndarray:
        face_path = Path(face_path)
        face_img= Image.open(face_path)
        face_img = face_img.resize((224, 224)) 
        face_img = image.img_to_array(face_img)

        try:
            embedding_obj = DeepFace.represent(
                img_path=face_img,
                model_name="Facenet512",
                enforce_detection=False
            )[0]

            embedding = np.array(embedding_obj["embedding"], dtype=np.float64)
            embedding = embedding / np.linalg.norm(embedding)

            embeddings_path = self.extracted_faces_embeddings_path / face_path.name
            feature_path = embeddings_path.with_name(embeddings_path.name + ".npy")
            np.save(feature_path, embedding)
            return embedding

        except Exception as e:
            raise Exception(f"Error generating embedding for {face_path}: {str(e)}")

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