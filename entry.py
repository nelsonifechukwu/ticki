import subprocess
import time
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from app.cbir import ImageProcessor
import threading

fe = ImageProcessor()
database = Path("app/static/database")
img_repo = database / "img_repo"
img_repo_list = list(img_repo.iterdir())

# Ensure RetinaFace runs on CPU to avoid GPU conflicts
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def convert_all_faces_to_embeddings():
    """Use multiprocessing to run face extraction in parallel."""
    with ProcessPoolExecutor(max_workers=4) as executor:  # Adjust workers based on CPU cores
        executor.map(fe.extract_faces_process, img_repo_list)

def start_face_extraction():
    """Runs face extraction in a background thread to avoid blocking Flask."""
    thread = threading.Thread(target=convert_all_faces_to_embeddings, daemon=True)
    thread.start()

if __name__ == '__main__':
    start_face_extraction()
    subprocess.run(["flask", "run"])