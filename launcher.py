import subprocess
import sys
import time
from app.tasks import convert_faces_to_embeddings_batch

# Get the absolute directory of the current script (this file)
basedir = os.path.abspath(os.path.dirname(__file__))

database = os.path.join(basedir, "static", "database")
img_repo = os.path.join(database, "img_repo")
# List image files
img_repo_list = [os.path.join(img_repo, img) for img in os.listdir(img_repo)]

def run_process(command):
    return subprocess.Popen(command, shell=True)

def are_tasks_complete():
    from app.tasks import celery_app
    
    insp = celery_app.control.inspect()
    scheduled = insp.scheduled()
    active = insp.active()
    reserved = insp.reserved()
    
    all_tasks = [scheduled, active, reserved]

    # Check if any worker still has tasks
    for tasks in all_tasks:
        if tasks:
            for worker_tasks in tasks.values():
                if worker_tasks:  # Tasks still pending or running
                    return False
    return True

if __name__ == "__main__":
    try:
        # Start Redis server
        print("Starting Redis server...")
        redis = run_process("redis-server")
        time.sleep(2)

        # Start Celery workers
        print("Starting Celery workers...")
        celery = run_process("celery -A app.tasks worker --loglevel=info --concurrency=4 --pool threads")
        time.sleep(4)

        # Start Flask server
        print("Starting Flask application...")
        flask = run_process("flask run")

        print("🚀 All services are up and running!")
        print("Waiting for Celery to finish all tasks...")

        # Poll to check if Celery has finished all tasks
        while True:
            if are_tasks_complete():
                print("✅ All Celery tasks completed.")
                convert_faces_to_embeddings_batch()
                break
            print("Celery is still processing tasks...")
            time.sleep(5)  # Wait before checking again

        print("Stopping Celery workers...")
        celery.terminate()
        celery.wait()
        print("Celery workers shut down. Flask is still running... Press CTRL+C to exit.")
        # Keep Flask running
        flask.wait()

    except KeyboardInterrupt:
        print("\nStopping all services...")
        run_process("pkill redis-server")
        run_process("lsof -ti:6379 | xargs kill -9") #kill all redis processes
        run_process("ps aux | grep celery | grep -v grep | awk '{print $2}' | xargs kill -9") #kill all celery workers
        #celery.terminate()
        flask.terminate()
        #redis.terminate()
        
        #celery.wait()
        flask.wait()
        #redis.wait()

        print("All services stopped.")
        sys.exit(0)
