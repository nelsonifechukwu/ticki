import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image
import numpy as np
import tensorflow as tf
from retinaface import RetinaFace

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# Ensure TensorFlow is set up for thread safety
thread_local = threading.local()

def get_session():
    """Each thread gets its own TensorFlow session to avoid conflicts."""
    if not hasattr(thread_local, "session"):
        thread_local.session = tf.compat.v1.Session()  # Create a new session per thread
    return thread_local.session

def extract_faces_thread(img_path: Path):
    """Thread-safe face extraction using RetinaFace."""
    session = get_session()  # Ensure session is thread-safe

    faces_directory = img_path.parent / "faces"
    faces_directory.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

    try:
        with session.as_default():
            faces = RetinaFace.extract_faces(
                img_path=str(img_path),
                align=True,
                expand_face_area=20,
            )
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return

    for i, face in enumerate(faces):
        if face.any():
            face = face.astype("uint8")
            img = Image.fromarray(face).convert("RGB")
            img = img.resize((224, 224))  # Resize for consistency
            face_filename = f"{img_path.stem}_face_{i}.png"
            face_filepath = faces_directory / face_filename
            img.save(face_filepath)

def convert_all_faces_to_embeddings(repository):
    """Run face extraction in a thread-safe manner."""
    with ThreadPoolExecutor(max_workers=5) as executor:  # Use fewer workers to avoid crashes
        executor.map(extract_faces_thread, repository)

# Run the process
database = Path("app/static/database")
img_repo = database / "img_repo"
img_repo_list = list(img_repo.iterdir())

start_time = time.perf_counter()
convert_all_faces_to_embeddings(img_repo_list)
duration = time.perf_counter() - start_time
print(f"Finished Img Conversion in {duration} s")
