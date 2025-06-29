from .cbir import ImageProcessor, logger
from typing import List
from pathlib import Path
from .celery import *

database = Path("app/static/database")
fe = ImageProcessor(database)

# ---- Celery Tasks ----
@celery_app.task(ignore_result=True)
def extract_faces(image_path: str):
    # Ensure idempotency: skip processing if this image_path was already handled (e.g., due to Celery task duplication)
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
def convert_faces_to_embeddings(face_path: str):
    # Ensure idempotency: skip processing if this image_path was already handled (e.g., due to Celery task duplication)
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

@celery_app.task
def convert_all_faces_to_embeddings(reprocess=False):
    allowed_exts = ("jpg", "png", "jpeg")
    faces_repo = fe.extracted_faces_path
    faces_repo_list = [str(img) for img in faces_repo.iterdir() if str(img).lower().endswith(allowed_exts)]

    tasks = [
        convert_faces_to_embeddings.s(path)
        for path in faces_repo_list
        if reprocess or not redis_client.exists(Path(path).name)
    ]

    if not tasks:
        logger.info("No new faces to process. All faces already processed.")
        return

    logger.info("🚀 Starting feature extraction for all faces...")
    group(tasks).apply_async()

def extract_all_faces(reprocess=False):
    allowed_exts = ("jpg", "png", "jpeg")
    img_data = fe.img_data
    img_data_list = [str(img) for img in img_data.iterdir() if str(img).lower().endswith(allowed_exts)]

    face_tasks = [
        extract_faces.s(path)
        for path in img_data_list
        if reprocess or not redis_client.exists(Path(path).name)
    ]

    if not face_tasks:
        logger.info("No images to process. Skipping extraction and embedding.")
        return

    logger.info("🚀 Starting face extraction...")
    return face_tasks
# ---- Main controller function ----
def main(reprocess=False):
    #use .si in the chained task, cause the first group doesn't return anything...
    chord(extract_all_faces(reprocess))(convert_all_faces_to_embeddings.si(reprocess))

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