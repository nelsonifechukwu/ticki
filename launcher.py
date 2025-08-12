
import sys
import time
import subprocess
from app.cbir import logger

def run_process(command):
    return subprocess.Popen(command, shell=True)

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
        from app.tasks import main
        from app.celery import redis_client
        reprocess = False
        if reprocess:
            redis_client.flushdb()
        main(reprocess) 

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
