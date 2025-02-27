import subprocess
import time
import os
from concurrent.futures import ProcessPoolExecutor, wait
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
        subprocess.run(["flask", "run"])
        # executor.map(fe.extract_faces_process, img_repo_list)  
        for _ in executor.map(fe.extract_faces_process, img_repo_list):
            pass
        
if __name__ == '__main__':
    convert_all_faces_to_embeddings()
    # executor.submit(subprocess.run(["flask", "run"]))
    