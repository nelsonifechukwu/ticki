from .tasks import redis_client
from threading import Thread
from typing import Tuple
from .tasks import redis_client
import redis
def _store_logic(paths: Tuple[str, str]):
    #first str is the img_path, the second is the face_path
    try:
        # Check Redis connection
        redis_client.ping()

        # Set keys only if they don't exist
        for i, path in enumerate(paths):
            if not redis_client.exists(path):
                redis_client.set(str(path), 'completed_f' if i == 1 else 'completed')

    except redis.exceptions.ConnectionError:
        raise RuntimeError("❌ Unable to connect to Redis. Is the server running?")
    except Exception as e:
        raise RuntimeError(f"❌ Error while storing keys in Redis: {e}")
    
def store_in_redis(paths):
    """
    Threaded Redis setter to avoid blocking main request thread.
    Assumes Redis server is already running.
    """
    thread = Thread(target=_store_logic, args=(paths,))
    #thread.daemon = True  # Dies with the main thread
    thread.start()