from pathlib import Path
import numpy as np
from .celery import *
from .cbir import ImageProcessor, logger

database = Path("app/static/database")
from .embeddings import embeddings_handler

allowed_exts = ("jpg", "png", "jpeg")

fe = ImageProcessor(database)

def process_image_query(image_input, img_name: str) -> dict:
    """Process image for face comparison and return results."""
    try:
        query_faces = fe.extract_faces(image_input)
        
        if not query_faces:
            return {"error": "No face detected in input image"}, 400
        if len(query_faces) > 1:
            return {"error": "Input image should contain only one face"}, 400
        
        query_feature = fe.extract_features(query_faces[0]).astype(float)
        results = embeddings_handler.get_similar_faces(query_feature)
        
        payload = {
            "status": "success",
            "user_url": img_name,
            "found_url": [
                {"score": round(score, 4), "img_name": img_name}
                for score, img_name in results
            ]
        }
        
        # Forward to TICKI_URL if configured
        try:
            from config import Config
            if hasattr(Config, 'TICKI_URL') and Config.TICKI_URL:
                import requests
                resp = requests.post(Config.TICKI_URL, json=payload, timeout=10)
                resp.raise_for_status()
                logger.info("✅ Forwarded payload to discovery API.")
        except Exception as e:
            logger.error(f"❌ Failed to forward payload to discovery API: {e}")
                
        return payload, 200
        
    except Exception as e:
        logger.error(f"Error processing {img_name}: {e}")
        return {"error": "Internal Server Error"}, 500

@celery_app.task
def process_and_store_image(image_input, img_name: str):
    """Process single image and store embeddings immediately (webhook use case)."""
    try:
        if not redis_client.setnx(img_name, "in-progress"):
            logger.info(f"Skipping {img_name}: Already processed.")
            return {"processed": 0, "stored": 0}
        
        faces = fe.extract_faces(image_input)
        if not faces:
            redis_client.set(img_name, 'no-faces')
            return {"processed": 0, "stored": 0}
        
        # Extract embeddings and store immediately
        embeddings = []
        face_ids = []
        for i, face_array in enumerate(faces):
            try:
                embedding = fe.extract_features(face_array)
                embeddings.append(embedding)
                face_ids.append(f"{img_name}_face_{i}")
            except Exception as e:
                logger.error(f"Failed to process face {i} from {img_name}: {e}")
        
        if embeddings:
            # Batch store all faces from this image
            embeddings_array = np.vstack(embeddings)  # Convert list to numpy array
            embeddings_handler.add_feature(embeddings_array, face_ids, sync_mode=False)
            redis_client.set(img_name, 'completed')
            logger.info(f"✅ Stored {len(embeddings)} embeddings from {img_name}")
        
        return {"processed": len(faces), "stored": len(embeddings)}
        
    except Exception as e:
        logger.error(f"❌ Failed to process {img_name}: {e}")
        redis_client.set(img_name, f'failed: {e}')
        raise

@celery_app.task
def process_single_image_to_npy(image_path: str, reprocess: bool = False):
    """Process single image, extract faces and save embeddings as .npy files."""
    try:
        img_name = Path(image_path).name
        
        # Check if already processed
        if not reprocess and redis_client.exists(img_name):
            logger.info(f"Skipping {img_name}: Already processed.")
            return {"processed": 0, "stored": 0, "image": img_name}
            
        redis_client.set(img_name, "in-progress")
        
        # Extract faces
        faces = fe.extract_faces(image_path)
        if not faces:
            redis_client.set(img_name, 'no-faces')
            return {"processed": 0, "stored": 0, "image": img_name}
        
        stored_count = 0
        # Process each face and save as .npy
        for i, face_array in enumerate(faces):
            try:
                embedding = fe.extract_features(face_array)
                
                # Save embedding to .npy file with face ID
                face_id = f"{Path(img_name).stem}_face_{i}"
                npy_path = embeddings_handler.extracted_faces_embeddings_path / f"{face_id}.npy"
                np.save(npy_path, embedding)
                
                stored_count += 1
                logger.debug(f"Saved {face_id}.npy")
                
            except Exception as e:
                logger.error(f"Failed to process face {i} from {img_name}: {e}")
        
        if stored_count > 0:
            redis_client.set(img_name, 'completed')
            logger.info(f"✅ {img_name}: Processed {len(faces)} faces, stored {stored_count} embeddings")
        else:
            redis_client.set(img_name, 'failed: no embeddings generated')
            
        return {"processed": len(faces), "stored": stored_count, "image": img_name}
        
    except Exception as e:
        logger.error(f"❌ Failed to process {image_path}: {e}")
        redis_client.set(Path(image_path).name, f'failed: {e}')
        return {"processed": 0, "stored": 0, "image": Path(image_path).name, "error": str(e)}

@celery_app.task
def load_npy_to_faiss(results):
    """Callback task to load all .npy files into FAISS after workers complete."""
    try:
        # Aggregate results from all workers
        total_processed = sum(r['processed'] for r in results)
        total_stored = sum(r['stored'] for r in results)
        
        logger.info(f"📦 All workers completed. Loading {total_stored} embeddings into FAISS...")
        
        # Load all .npy files into FAISS in a single batch operation
        embeddings_handler.load_all_embeddings_in_faiss()
        
        logger.info(f"✅ Successfully processed {total_processed} faces")
        logger.info(f"✅ Loaded {total_stored} embeddings into FAISS")
        
        return {"processed": total_processed, "stored": total_stored}
        
    except Exception as e:
        logger.error(f"❌ Failed to load embeddings to FAISS: {e}")
        raise

@celery_app.task
def process_directory_batch_parallel(reprocess=False):
    """Coordinate parallel processing using Celery chains (no blocking calls)."""
    try:
        img_data = fe.img_data
        img_data_list = [str(img) for img in img_data.iterdir() if str(img).lower().endswith(allowed_exts)]
        
        if not reprocess:
            img_data_list = [path for path in img_data_list if not redis_client.exists(Path(path).name)]
        
        if not img_data_list:
            logger.info("No new images to process.")
            return {"processed": 0, "stored": 0}
        
        len_img_data_list = len(img_data_list)
        logger.info(f"🚀 Starting parallel batch processing for {len_img_data_list} images...")
        
        # Use chord to run all workers in parallel, then execute callback
        from celery import chord
        
        # Create the parallel job
        job = chord(
            (process_single_image_to_npy.s(img_path, reprocess) for img_path in img_data_list),
            load_npy_to_faiss.s()
        )
        
        # Execute the chord (non-blocking)
        result = job.apply_async()
        
        logger.info(f"📤 Submitted {len_img_data_list} image processing tasks to workers")
        
        # Return the AsyncResult (don't wait)
        return {"task_id": result.id, "submitted_images": len_img_data_list}
        
    except Exception as e:
        logger.error(f"❌ Failed to submit parallel batch processing: {e}")
        raise

# Keep the old single-worker method as backup
@celery_app.task
def process_directory_batch_single(reprocess=False):
    """Single worker batch process (backup method)."""
    try:
        img_data = fe.img_data
        img_data_list = [str(img) for img in img_data.iterdir() if str(img).lower().endswith(allowed_exts)]
        
        if not reprocess:
            img_data_list = [path for path in img_data_list if not redis_client.exists(Path(path).name)]
        
        if not img_data_list:
            logger.info("No new images to process.")
            return {"processed": 0, "stored": 0}
        
        len_img_data_list = len(img_data_list)
        logger.info(f"🚀 Starting batch processing for {len_img_data_list} images...")
        
        # Extract all embeddings first (in memory)
        all_embeddings = []
        all_face_ids = []
        processed_count = 0
        
        for num, img_path in enumerate(img_data_list):
            img_name = Path(img_path).name
            redis_client.set(img_name, "in-progress")
            
            try:
                faces = fe.extract_faces(img_path)
                if faces:
                    for i, face_array in enumerate(faces):
                        try:
                            embedding = fe.extract_features(face_array)
                            all_embeddings.append(embedding)
                            all_face_ids.append(f"{Path(img_name).stem}_face_{i}")
                        except Exception as e:
                            logger.error(f"Failed to process face {i} from {img_name}: {e}")
                    processed_count += 1
                    redis_client.set(img_name, 'completed')
                else:
                    redis_client.set(img_name, 'no-faces')
                
            except Exception as e:
                logger.error(f"Failed to process {img_name}: {e}")
                redis_client.set(img_name, f'failed: {e}')
                
            logger.info(f"Image {num} of {len_img_data_list} processed")
        # Single batch FAISS operation for all embeddings
        if all_embeddings:
            logger.info(f"📦 Batch storing {len(all_embeddings)} embeddings...")
            embeddings_array = np.vstack(all_embeddings)  # Convert list to numpy array
            embeddings_handler.add_feature(embeddings_array, all_face_ids, sync_mode=False)
            logger.info(f"✅ Successfully stored {len(all_embeddings)} embeddings from {processed_count} images")
        
        return {"processed": processed_count, "stored": len(all_embeddings)}
        
    except Exception as e:
        logger.error(f"❌ Batch processing failed: {e}")
        raise

def main(reprocess=False, use_parallel=True):
    """Main controller for batch processing directory images."""
    if use_parallel:
        return process_directory_batch_parallel.delay(reprocess)
    else:
        return process_directory_batch_single.delay(reprocess)