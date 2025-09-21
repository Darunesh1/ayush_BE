# namasthe_mapping/__init__.py
# Ensure tasks are imported

default_app_config = "namasthe_mapping.apps.NamasteMappingConfig"


# Force import tasks when module is imported
def import_tasks():
    try:
        from . import tasks
    except ImportError:
        pass


import_tasks()
