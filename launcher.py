
import subprocess
import sys
import time

from app.cbir import logger
def run_process(command):
    return subprocess.Popen(command, shell=True)

def tasks_completed():
    from app.celery import celery_app 

    insp = celery_app.control.inspect()
    scheduled = insp.scheduled()
    active = insp.active()
    reserved = insp.reserved()

    # Check if any worker still has tasks
    for tasks in [scheduled, active, reserved]:
        if tasks and any(tasks.values()):
            return False
    return True


def wait_until_tasks_finish(phase: str, poll_interval=2):
    logger.info(f"📌 Waiting for Celery to complete: {phase} ...")
    while True:
        if tasks_completed():
            logger.info(f"✅ {phase} completed.")
            break
        logger.info(f"⏳ {phase} in progress...")
        time.sleep(poll_interval)

if __name__ == "__main__":
    try:
        logger.info("Starting Redis server...")
        redis = run_process("redis-server")
        time.sleep(2)

        logger.info("Starting Celery workers...")
        celery = run_process("celery -A app.celery worker --loglevel=info --concurrency=5 --pool threads") 
        time.sleep(2)

        logger.info("Starting Flask application...") 
        flask = run_process("flask run")

        logger.info("🚀 All services are up and running!")

        logger.info("Face extraction Started...")
        from app.tasks import extract_all_faces, convert_all_faces_to_embeddings
        from app.celery import redis_client
        reprocess = True
        if reprocess:
            redis_client.flushdb()

        logger.info("🚀 Starting face extraction...")
        extract_all_faces(reprocess)
        wait_until_tasks_finish("Face Extraction")

        logger.info("🚀 Starting feature embedding...")
        convert_all_faces_to_embeddings(reprocess)
        wait_until_tasks_finish("Face Embedding")

        logger.info("Celery & Flask are still running... Press CTRL+C to exit.")
        flask.wait()
        # celery.terminate()
        # celery.wait()

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
