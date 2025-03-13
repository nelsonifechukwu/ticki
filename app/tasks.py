# app/tasks.py
from celery import Celery, group
from .cbir import ImageProcessor
from typing import List
from . import *

fe = ImageProcessor()
celery_app = Celery('tasks', broker='redis://localhost:6379/0')
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
def extract_faces_batch(image_paths: List[str]):
    # Use Celery group to distribute tasks concurrently
    task_group = group(extract_faces.s(path) for path in image_paths)
    result = task_group.apply_async()
    #return result
