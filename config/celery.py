import os

from celery import Celery

# set default Django settings
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

app = Celery("config")

# Load settings from Django
app.config_from_object("django.conf:settings", namespace="CELERY")

# Discover tasks in all installed apps (look for tasks.py)
app.autodiscover_tasks()
