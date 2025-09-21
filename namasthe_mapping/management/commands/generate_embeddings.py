# namasthe_mapping/management/commands/generate_embeddings.py
# Fixed version that avoids result.get() issues

import logging
import time

from celery import group
from django.apps import apps
from django.core.cache import cache
from django.core.management.base import BaseCommand

from namasthe_mapping.tasks import generate_single_embedding, update_progress

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Generate BioBERT embeddings using optimized Redis + Celery"

    def add_arguments(self, parser):
        parser.add_argument(
            "--model",
            type=str,
            choices=["Ayurvedha", "Siddha", "Unani", "ICD11Term", "all"],
            default="all",
            help="Model to generate embeddings for",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=10,  # Smaller batches to avoid timeouts
            help="Batch size for Celery tasks",
        )
        parser.add_argument(
            "--workers",
            type=int,
            default=4,
            help="Number of parallel Celery workers to use",
        )
        parser.add_argument(
            "--force",
            action="store_true",
            help="Force regenerate embeddings even if they exist",
        )
        parser.add_argument(
            "--limit", type=int, help="Limit number of records to process (for testing)"
        )
        parser.add_argument(
            "--timeout",
            type=int,
            default=300,
            help="Timeout in seconds for waiting for task completion",
        )

    def handle(self, *args, **options):
        self.stdout.write(
            self.style.SUCCESS(
                "🚀 Starting optimized BioBERT embedding generation with Redis + Celery..."
            )
        )

        # Verify Celery is running
        if not self.check_celery_workers():
            self.stdout.write(
                self.style.ERROR("❌ No Celery workers detected. Start workers first:")
            )
            self.stdout.write("celery -A config worker --loglevel=info")
            return

        # Determine which models to process
        if options["model"] == "all":
            models_to_process = ["Ayurvedha", "Siddha", "Unani", "ICD11Term"]
        else:
            models_to_process = [options["model"]]

        total_processed = 0

        # Process each model
        for model_name in models_to_process:
            processed_count = self.process_model_fixed(
                model_name,
                options["batch_size"],
                options["workers"],
                options["force"],
                options.get("limit"),
                options["timeout"],
            )
            total_processed += processed_count

        self.stdout.write(
            self.style.SUCCESS(
                f"✅ Completed! Total records processed: {total_processed}"
            )
        )

    def check_celery_workers(self) -> bool:
        """Check if Celery workers are running"""
        try:
            from celery import current_app

            inspect = current_app.control.inspect()
            active_workers = inspect.active()

            if active_workers:
                worker_count = len(active_workers)
                self.stdout.write(f"✅ Found {worker_count} active Celery workers")
                return True
            else:
                return False

        except Exception as e:
            logger.error(f"Error checking Celery workers: {str(e)}")
            return False

    def process_model_fixed(
        self,
        model_name: str,
        batch_size: int,
        max_workers: int,
        force: bool,
        limit: int = None,
        timeout: int = 300,
    ) -> int:
        """Process a single model using fixed Celery approach"""

        self.stdout.write(f"\n📋 Processing {model_name} with {max_workers} workers...")

        # Get the model class
        if model_name == "ICD11Term":
            Model = apps.get_model("terminologies", "ICD11Term")
        else:
            Model = apps.get_model("terminologies", model_name)

        # Get records that need processing
        if force:
            queryset = Model.objects.all()
        else:
            queryset = Model.objects.filter(embedding__isnull=True)

        # Apply limit if specified
        if limit:
            queryset = queryset[:limit]

        # Get record IDs for processing
        record_ids = list(queryset.values_list("pk", flat=True))
        total_count = len(record_ids)

        if total_count == 0:
            self.stdout.write(f"⚪ No {model_name} records need embedding generation")
            return 0

        self.stdout.write(f"🎯 Found {total_count} {model_name} records to process")

        # Initialize progress tracking
        task_name = f"{model_name}_{int(time.time())}"
        update_progress.delay(task_name, 0, total_count, model_name)

        # Process records individually with proper async handling
        processed_count = 0
        active_tasks = []

        self.stdout.write(f"⚙️ Starting individual task processing...")

        # Submit all tasks
        for record_id in record_ids:
            task = generate_single_embedding.delay(model_name, record_id, force)
            active_tasks.append(task)

            # Process in batches to avoid overwhelming
            if len(active_tasks) >= batch_size:
                successful_count = self.wait_for_tasks(active_tasks, timeout)
                processed_count += successful_count

                # Update progress
                progress = (processed_count / total_count) * 100
                self.stdout.write(
                    f"📈 {model_name}: {processed_count}/{total_count} ({progress:.1f}%)"
                )

                update_progress.delay(
                    task_name, processed_count, total_count, model_name
                )

                # Reset for next batch
                active_tasks = []

                # Brief pause between batches
                time.sleep(2)

        # Process remaining tasks
        if active_tasks:
            successful_count = self.wait_for_tasks(active_tasks, timeout)
            processed_count += successful_count

            # Final progress update
            update_progress.delay(task_name, processed_count, total_count, model_name)

        self.stdout.write(
            self.style.SUCCESS(f"✅ {model_name}: Completed {processed_count} records")
        )

        return processed_count

    def wait_for_tasks(self, tasks, timeout=300):
        """Wait for tasks to complete and count successful ones"""

        successful_count = 0
        completed_count = 0
        start_time = time.time()

        self.stdout.write(f"⏳ Waiting for {len(tasks)} tasks to complete...")

        # Wait for tasks with timeout
        while completed_count < len(tasks) and (time.time() - start_time) < timeout:
            for i, task in enumerate(tasks):
                if task.ready():
                    try:
                        result = task.result  # Use .result instead of .get()
                        if isinstance(result, dict) and result.get("success"):
                            successful_count += 1
                        completed_count += 1
                        tasks[i] = None  # Mark as processed
                    except Exception as e:
                        logger.error(f"Task failed: {str(e)}")
                        completed_count += 1
                        tasks[i] = None

            # Remove processed tasks
            tasks = [t for t in tasks if t is not None]

            # Brief pause before checking again
            time.sleep(1)

        # Handle any remaining tasks (timeout)
        remaining_tasks = len([t for t in tasks if t is not None])
        if remaining_tasks > 0:
            self.stdout.write(
                self.style.WARNING(f"⚠️ {remaining_tasks} tasks timed out")
            )

        return successful_count
