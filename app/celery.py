from celery import Celery, group, chain
import redis

celery_app = Celery('app', broker='redis://localhost:6379/0', include=['app.tasks'], backend='redis://localhost:6379/0')
redis_client = redis.Redis(host='localhost', port=6379, db=1) 	

__all__ =['redis', 'celery_app', 'redis_client', 'group', 'chain']
#celery_app.conf.broker_transport_options = {'visibility_timeout': 9999999}
#celery_app.conf.worker_deduplicate_successful_tasks = True
#celery_app.conf.task_acks_late=True