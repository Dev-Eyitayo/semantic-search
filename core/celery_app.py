from celery import Celery
from core.config import settings

celery_app = Celery(
    "sheltly_tasks",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL
)

celery_config = {
    "task_serializer": "json",
    "accept_content": ["json"],
    "result_serializer": "json",
    "timezone": "UTC",
    "enable_utc": True,
}

if settings.REDIS_URL.startswith("rediss://"):
    ssl_conf = {
        'ssl_cert_reqs': None 
    }
    celery_config["broker_use_ssl"] = ssl_conf
    celery_config["redis_backend_use_ssl"] = ssl_conf

# 3. Apply the config
celery_app.conf.update(celery_config)

celery_app.autodiscover_tasks(["services"])