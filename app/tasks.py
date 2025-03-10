# app/tasks.py
from celery import Celery, group
from .cbir import ImageProcessor

fe = ImageProcessor()
celery_app = Celery('tasks', broker='redis://localhost:6379/0')

@celery_app.task (ignore_result=True)
def extract_faces(image_path):
    try:
        return fe.extract_faces(image_path)
    except Exception as e:
        # Log the error properly and re-raise to mark task as failed
        # celery_app.logger.error(f"Face extraction failed for {image_path}: {str(e)}")
        raise

@celery_app.task (ignore_result=True)
def extract_faces_batch(image_paths):
    # Use Celery group to distribute tasks concurrently
    task_group = group(extract_faces.s(path) for path in image_paths)
    result = task_group.apply_async()
    return result
