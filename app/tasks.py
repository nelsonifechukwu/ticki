from pathlib import Path
from .celery import *
from .cbir import ImageProcessor, logger
from .embeddings import embeddings_handler

database = Path("app/static/database")
fe = ImageProcessor(database)

# ---- Memory-Based Processing Functions ----
def process_image_in_memory(image_bytes: bytes, img_name: str) -> dict:
    """
    Process image completely in memory without saving to disk.
    Returns comparison results directly.
    """
    try:
        # Extract faces from image bytes (returns numpy arrays)
        query_faces = fe.extract_faces(image_bytes)
        
        if not query_faces:
            return {"error": "No face detected in input image"}, 400
            
        if len(query_faces) > 1:
            return {"error": "Input image should contain only one face"}, 400
        
        # Extract features from first face (numpy array)
        query_face = query_faces[0]
        query_feature = fe.extract_features(query_face).astype(float)
        
        # Get similar faces
        results = embeddings_handler.get_similar_faces(query_feature)
        
        return {
            "status": "success",
            "user_url": img_name,
            "found_url": [
                {"score": round(score, 4), "img_name": img_name}
                for score, img_name in results
            ]
        }, 200
        
    except Exception as e:
        logger.error(f"Error processing {img_name} in memory: {e}")
        return {"error": "Internal Server Error"}, 500

@celery_app.task
def process_and_store_image(image_bytes: bytes, img_name: str):
    """
    Process image in memory and store embeddings in database.
    Used for building the face database from uploaded images.
    """
    try:
        # Ensure idempotency
        if not redis_client.setnx(img_name, "in-progress"):
            logger.info(f"Skipping {img_name}: Already processed.")
            return {"processed": 0, "stored": 0}
        
        # Extract faces from image bytes (returns numpy arrays) 
        faces = fe.extract_faces(image_bytes)
        
        if not faces:
            logger.warning(f"No faces found in {img_name}")
            redis_client.set(img_name, 'no-faces')
            return {"processed": 0, "stored": 0}
        
        stored_count = 0
        for i, face_array in enumerate(faces):
            try:
                # Extract features from face (numpy array)
                embedding = fe.extract_features(face_array)
                
                # Store embedding in database
                face_id = f"{img_name}_face_{i}"
                embeddings_handler.add_feature(embedding, face_id)
                stored_count += 1
                
            except Exception as e:
                logger.error(f"Failed to process face {i} from {img_name}: {e}")
                continue
        
        redis_client.set(img_name, 'completed')
        logger.info(f"✅ Processed {len(faces)} faces from {img_name}, stored {stored_count} embeddings")
        return {"processed": len(faces), "stored": stored_count}
        
    except Exception as e:
        logger.error(f"❌ Failed to process {img_name}: {e}")
        redis_client.set(img_name, f'failed: {e}')
        raise