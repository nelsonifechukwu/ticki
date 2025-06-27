
import subprocess
import sys
import time

from app.cbir import logger
def run_process(command):
    return subprocess.Popen(command, shell=True)

def tasks_completed():
    from app.tasks import celery_app 

    insp = celery_app.control.inspect()
    scheduled = insp.scheduled()
    active = insp.active()
    reserved = insp.reserved()

    # Check if any worker still has tasks
    for tasks in [scheduled, active, reserved]:
        if tasks and any(tasks.values()):
            return False
    return True

if __name__ == "__main__":
    try:
        logger.info("Starting Redis server...")
        redis = run_process("redis-server")
        time.sleep(2)

        logger.info("Starting Celery workers...")
        celery = run_process("celery -A app.tasks worker --loglevel=info --concurrency=5 --pool threads") 
        time.sleep(2)

        logger.info("Starting Flask application...") 
        flask = run_process("flask run")

        logger.info("🚀 All services are up and running!")

        logger.info("Face extraction Started...")
        from app.tasks import redis_client, extract_all_faces
        reprocess = False
        if reprocess:
            redis_client.flushdb()
        extract_all_faces(reprocess) 

        while True:
            logger.info("Celery is still extracting faces...")
            if tasks_completed():
                logger.info("✅ Face extraction completed.")
                logger.info("✅ Face -> Embeddings Started.")
                from app.tasks import convert_all_faces_to_embeddings
                convert_all_faces_to_embeddings(reprocess)
                logger.info("Celery is converting face to embeddings...")
                while True:
                    if tasks_completed():
                        logger.info("✅ Face -> embeddings completed.")
                        break
                break
            time.sleep(2)

        logger.info("Celery & Flask are still running... Press CTRL+C to exit.")
        flask.wait()

    except KeyboardInterrupt:
        logger.info("Stopping all services...")
        run_process("pkill redis-server")
        run_process("lsof -ti:6379 | xargs kill -9")
        run_process("ps aux | grep celery | grep -v grep | awk '{print $2}' | xargs kill -9") 
        flask.terminate()
        flask.wait()
        logger.info("All services stopped.")
        sys.exit(0)

    except:
        run_process("pkill redis-server")
        run_process("lsof -ti:6379 | xargs kill -9")
        run_process("ps aux | grep celery | grep -v grep | awk '{print $2}' | xargs kill -9")
        flask.terminate()
        flask.wait()
        raise Exception("Error launching this application")
