from django.apps import AppConfig


class NamastheMappingConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "namasthe_mapping"

    def ready(self):
        """Import tasks when Django app is ready"""
        try:
            from . import tasks  # Import tasks to register them

            print("✅ NAMASTE mapping tasks imported")
        except ImportError:
            print("❌ Failed to import NAMASTE tasks")
