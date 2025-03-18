import subprocess
import sys
import time

def run_process(command):
    return subprocess.Popen(command, shell=True)

def are_tasks_complete():
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
        # Start Redis server
        print("Starting Redis server...")
        redis = run_process("redis-server")
        time.sleep(2)

        # Start Celery workers
        print("Starting Celery workers...")
        celery = run_process("celery -A app.tasks worker --loglevel=info --concurrency=4 --pool threads")
        time.sleep(3)

        # Start Flask server
        print("Starting Flask application...")
        flask = run_process("flask run")

        print("🚀 All services are up and running!")
        print("Waiting for Celery to finish all tasks...")
        
        time.sleep(2) #wait for flask run to initiate the first celery tasks
        # Poll to check if Celery has finished all tasks
        while True:
            print("Celery is still processing tasks...")
            if are_tasks_complete(): #first round of tasks -> face extraction
                print("✅ All Celery tasks completed.")
                from app.tasks import convert_all_faces_to_embeddings
                convert_all_faces_to_embeddings()
                while True:
                    if are_tasks_complete(): #2nd round of tasks -> feature extraction
                        break
                break
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
        
    except:
        run_process("pkill redis-server")
        run_process("lsof -ti:6379 | xargs kill -9") #kill all redis processes
        run_process("ps aux | grep celery | grep -v grep | awk '{print $2}' | xargs kill -9") #kill all celery workers
        flask.terminate()
        flask.wait()
        raise Exception("Error launching this application")