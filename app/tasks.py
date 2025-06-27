from celery import Celery, group
import redis
from .cbir import ImageProcessor, logger
from typing import List
from pathlib import Path
from threading import Thread 

database = Path("app/static/database")
fe = ImageProcessor(database)
celery_app = Celery('tasks', broker='redis://localhost:6379/0')
redis_client = redis.Redis(host='localhost', port=6379, db=1) 


@celery_app.task(ignore_result=True) 
def extract_faces(image_path: str):
    img_name = Path(image_path).name
    if not redis_client.setnx(img_name, "in-progress"):
        logger.info(f"Skipping {img_name}: Already processed.")
        return
    try:
        fe.extract_faces(image_path)
        redis_client.set(img_name, 'completed')
        logger.info(f"✅ Faces extracted from {img_name}.")
    except Exception as e:
        logger.error(f"❌ Face extraction from {img_name} failed: {e}")
        redis_client.set(img_name, f"in-complete: {e}")
        raise

@celery_app.task(ignore_result=True)
def extract_faces_batch(image_paths: List[str], reprocess=False):
    task_group = None
    if not reprocess:
        task_group = group(extract_faces.s(path) for path in image_paths if not redis_client.exists(Path(path).name)) 
    else:
        task_group = group(extract_faces.s(path) for path in image_paths)
    task_group.apply_async()

def extract_all_faces(reprocess=False):
    img_data = fe.img_data
    allowed_exts = ("jpg", "png", "jpeg")
    img_data_list = [str(img) for img in img_data.iterdir() if str(img).lower().endswith(allowed_exts)]
    extract_faces_batch.delay(img_data_list, reprocess)

@celery_app.task(ignore_result=True)
def convert_faces_to_embeddings(face_path: str):
    face_img_name = Path(face_path).name
    if not redis_client.setnx(face_img_name, "in-progress"):
        logger.info(f"Skipping {face_img_name}: Already processed.")
        return
    try:
        fe.extract_features(face_path)
        redis_client.set(face_img_name, 'completed_f')
        logger.info(f"✅ Feature extraction successful: {face_img_name}.")
    except Exception as e:
        logger.error(f"❌ Feature extraction from {face_img_name} failed: {e}")
        redis_client.set(face_img_name, f"in-complete_f: {e}")
        raise

@celery_app.task(ignore_result=True)
def convert_faces_to_embeddings_batch(faces_path: List[str], reprocess=False):
    if not reprocess:
        task_group = group(convert_faces_to_embeddings.s(path) for path in faces_path if not redis_client.exists(Path(path).name))
    else:
        task_group = group(convert_faces_to_embeddings.s(path) for path in faces_path)
    task_group.apply_async()

def convert_all_faces_to_embeddings(reprocess=False):
    allowed_exts = ("jpg", "png", "jpeg")
    faces_repo = fe.extracted_faces_path
    faces_repo_list = [str(img) for img in faces_repo.iterdir() if str(img).lower().endswith(allowed_exts)]
    convert_faces_to_embeddings_batch.delay(faces_repo_list, reprocess)  

@celery_app.task(ignore_result=True)
def _store_in_redis(img_path: str, faces_path: List[str]):
    try:
        redis_client.ping()
        new_upload = False
        img_name = Path(img_path).name
        new_upload = redis_client.setnx(img_name, "completed")     
        for face_path in faces_path:
            face_img_name = Path(face_path).name
            redis_client.setnx(face_img_name, "completed")

        if new_upload:
            logger.info(f"✅ {img_name} stored in Redis")
        else:
            logger.info(f"ℹ️ {img_name} already in Redis")
    except redis.exceptions.ConnectionError:
        logger.error("❌ Unable to connect to Redis. Is the server running?")
        raise RuntimeError("❌ Unable to connect to Redis. Is the server running?")
    except Exception as e:
        logger.error(f"❌ Error while storing keys in Redis: {e}")
        raise RuntimeError(f"❌ Error while storing keys in Redis: {e}") 

def store_in_redis(img_path: str, faces_path: List[str]):
    _store_in_redis.delay(img_path, faces_path)