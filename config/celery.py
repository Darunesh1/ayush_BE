# config/celery.py

import os

from celery import Celery

# Make sure this matches your actual settings path
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

app = Celery("ayushsync")
app.config_from_object("django.conf:settings", namespace="CELERY")

# This will automatically find tasks.py files in all Django apps
app.autodiscover_tasks()


@app.task(bind=True, ignore_result=True)
def debug_task(self):
    print(f"Request: {self.request!r}")
