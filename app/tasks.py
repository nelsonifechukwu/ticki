# app/tasks.py
from celery import Celery, group
from .cbir import ImageProcessor
from typing import List
from . import *
from pathlib import Path
import os
import numpy as np

#-----Todo-----#
#Initialize
#embeddings path
#img_repo_path
#faces_path
fe = ImageProcessor()
celery_app = Celery('tasks', broker='redis://localhost:6379/0')
database = Path("app/static/database")
embeddings_directory = database / "img_repo" / "embeddings"
embeddings_directory.mkdir(parents=True, exist_ok=True)
#celery_app.conf.broker_transport_options = {'visibility_timeout': 9999999}
#celery_app.conf.worker_deduplicate_successful_tasks = True
#celery_app.conf.task_acks_late=True

@celery_app.task (ignore_result=True)
def extract_faces(image_path: str):
    if redis_client.exists(image_path):
        print(f"Skipping {image_path}: Already processed.")
        return
    try:
        fe.extract_faces(image_path)
        redis_client.set(image_path, 'completed')
        print(f"✅ {image_path} processed and recorded.")
    except Exception as e:
        # Log the error properly and re-raise to mark task as failed
        # celery_app.logger.error(f"Face extraction failed for {image_path}: {str(e)}")
        raise

@celery_app.task (ignore_result=True)
def extract_faces_batch(image_paths: List[str], repeat=False):
    # Use Celery group to distribute tasks concurrently
    task_group = None
    if not repeat:
        #don't add already processed task (imgs) to the task group again
        task_group = group(extract_faces.s(path) for path in image_paths if not redis_client.exists(path)) 
    else:
        task_group = group(extract_faces.s(path) for path in image_paths)
    result = task_group.apply_async()
    #return result

def extract_all_faces(repeat_tasks=False):
    basedir = os.path.abspath(os.path.dirname(__file__))
    database = os.path.join(basedir, "static", "database")
    img_repo = os.path.join(database, "img_repo")
    allowed_exts=("jpg", "png", "jpeg")

    # List image files
    img_repo_list = [os.path.join(img_repo, img) for img in os.listdir(img_repo) if img.endswith(allowed_exts)]
    extract_faces_batch.delay(img_repo_list, repeat_tasks)
    
@celery_app.task(ignore_result=True)
def convert_faces_to_embeddings(face_path: str):
    embedding = fe.extract_features(face_path)
    embeddings_path = embeddings_directory / Path(face_path).stem
    try:
        np.save(embeddings_path.with_suffix(".npy"), embedding)
    except:
        raise Exception(f"Error saving {face_path} embedding")

@celery_app.task(ignore_result=True)
def convert_faces_to_embeddings_batch(faces_path: List[str], repeat_tasks=False):
    task_group = group(convert_faces_to_embeddings.s(path) for path in faces_path)
    result = task_group.apply_async()

def convert_all_faces_to_embeddings():
    basedir = os.path.abspath(os.path.dirname(__file__))
    database = os.path.join(basedir, "static", "database")
    faces_repo = os.path.join(database, "img_repo", "faces")
    if os.path.exists(faces_repo):
        print(f"The directory {faces_repo} exists.")
        faces_repo_list = [os.path.join(faces_repo, img) for img in os.listdir(faces_repo)]
        convert_faces_to_embeddings_batch.delay(faces_repo_list, repeat_tasks=False)  
    else:
        raise Exception(f"The directory {faces_repo} does not exist.")