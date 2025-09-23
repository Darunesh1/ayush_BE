# namasthe_mapping/__init__.py

# Remove the automatic import - this is causing the error
# Let Celery's autodiscovery handle the tasks instead

default_app_config = "namasthe_mapping.apps.NamasteMappingConfig"

# Remove this function and call - it's causing AppRegistryNotReady
# def import_tasks():
#     try:
#         from . import tasks
#     except ImportError:
#         pass

# import_tasks()  # Remove this line

# Celery will automatically discover tasks, so no manual import needed
