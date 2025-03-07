# app/tasks.py
from celery import Celery, group
from .cbir import ImageProcessor
fe = ImageProcessor()
celery_app = Celery('tasks', broker='redis://localhost:6379/0')

# @celery_app.task
# def extract_faces(image_path):
#     fe.extract_faces_process(image_path)

@celery_app.task
def extract_faces(image_path):
    fe.extract_faces_process(image_path)

@celery_app.task
def extract_faces_batch(image_paths):
    task_group = group(extract_faces.s(path) for path in image_paths)
    task_group.apply_async()