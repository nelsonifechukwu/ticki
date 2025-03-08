import subprocess
import sys
import time

def run_process(command):
    return subprocess.Popen(command, shell=True)

if __name__ == "__main__":
    try:
        # Start Redis server
        print("Starting Redis server...")
        redis = run_process("redis-server")
        time.sleep(5)  # Wait briefly to ensure Redis starts

        # Start Celery workers
        print("Starting Celery workers...")
        celery = run_process("celery -A app.tasks worker --loglevel=info --concurrency=4 --pool threads")
        #celery = run_process("celery -A app.tasks worker --loglevel=info --pool=prefork --concurrency=4 --prefetch-multiplier=1 --without-gossip --without-mingle")
        time.sleep(5)  # Wait briefly to ensure Celery starts

        # Start Flask server
        print("Starting Flask application...")
        flask = run_process("flask run")

        print("🚀 All services are up and running!")
        print("Press CTRL+C to stop.")

        # Wait indefinitely while all processes run
        flask.wait()

    except KeyboardInterrupt:
        print("\nStopping all services...")
        flask.terminate()
        celery.terminate()
        redis.terminate()

        flask.wait()
        celery.wait()
        redis.wait()

        print("All services stopped.")
        sys.exit(0)
