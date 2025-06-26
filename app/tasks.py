# app/tasks.py
from celery import Celery, group
import redis
from .cbir import ImageProcessor
from typing import List
from pathlib import Path
import numpy as np

database = Path("app/static/database")
fe = ImageProcessor(database)
celery_app = Celery('tasks', broker='redis://localhost:6379/0')
redis_client = redis.Redis(host='localhost', port=6379, db=1)

#celery_app.conf.broker_transport_options = {'visibility_timeout': 9999999}
#celery_app.conf.worker_deduplicate_successful_tasks = True
#celery_app.conf.task_acks_late=True

@celery_app.task (ignore_result=True)
def extract_faces(image_path: str):
    # Ensure idempotency: skip processing if this image_path was already handled (e.g., due to Celery task duplication)
    img_name = Path(image_path).name
    if redis_client.exists(img_name):
        print(f"Skipping {img_name}: Already processed.")
        return
    try:
        fe.extract_faces(image_path)
        redis_client.set(img_name, 'completed')
        print(f"✅ Face extraction successful: {img_name} is processed and recorded.")
    except Exception as e:
        print(f"❌ {img_name} unprocessed.")
        redis_client.set(img_name, f"in-complete: {e}")
        raise

@celery_app.task (ignore_result=True)
def extract_faces_batch(image_paths: List[str], reprocess=False):
    # Use Celery group to distribute tasks concurrently
    task_group = None
    if not reprocess:
        # only add unprocessed task (imgs) to the task group.
        task_group = group(extract_faces.s(path) for path in image_paths if not redis_client.exists(path)) 
    else:
        task_group = group(extract_faces.s(path) for path in image_paths)
    result = task_group.apply_async()
    #return result

def extract_all_faces(reprocess=False):
    img_data = fe.img_data
    allowed_exts=("jpg", "png", "jpeg")
    # List image files
    img_data_list = [str(img) for img in img_data.iterdir() if str(img).lower().endswith(allowed_exts)]
    extract_faces_batch.delay(img_data_list, reprocess)
    
@celery_app.task(ignore_result=True)
def convert_faces_to_embeddings(face_path: str):
    # Ensure idempotency: skip processing if this face_path was already handled (e.g., due to Celery task duplication)
    face_img_name = Path(face_path).name
    if redis_client.exists(face_img_name):
        print(f"Skipping {face_img_name}: Already processed.")
        return
    try:
        fe.extract_features(face_path)
        redis_client.set(face_img_name, 'completed_f')
        print(f"✅ Feature extraction successful: {face_img_name} is processed and recorded.")
    except Exception as e:
        print(f"❌ {face_img_name} unprocessed.")
        redis_client.set(face_img_name, f"in-complete_f: {e}")
        raise

@celery_app.task(ignore_result=True)
def convert_faces_to_embeddings_batch(faces_path: List[str], reprocess=False):
    if not reprocess:
        # only add unprocessed task (faces) to the task group. 
        task_group = group(convert_faces_to_embeddings.s(path) for path in faces_path if not redis_client.exists(path))
    else:
        task_group = group(convert_faces_to_embeddings.s(path) for path in faces_path)
    result = task_group.apply_async()

def convert_all_faces_to_embeddings(reprocess=False):
    allowed_exts=("jpg", "png", "jpeg")
    faces_repo = fe.extracted_faces_path
    faces_repo_list = [str(img) for img in faces_repo.iterdir() if str(img).lower().endswith(allowed_exts)]
    convert_faces_to_embeddings_batch.delay(faces_repo_list, reprocess)  
