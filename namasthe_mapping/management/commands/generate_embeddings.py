# namasthe_mapping/management/commands/generate_embeddings.py
# Optimized embedding generation with Redis + Celery

import logging
import math
import time

from celery import group
from django.apps import apps
from django.core.cache import cache
from django.core.management.base import BaseCommand

from namasthe_mapping.tasks import generate_batch_embeddings, update_progress

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
            default=20,  # Larger batches for parallel processing
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
            "--monitor",
            action="store_true",
            help="Monitor progress of running embedding tasks",
        )

    def handle(self, *args, **options):
        if options["monitor"]:
            self.monitor_progress()
            return

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
            self.stdout.write("celery -A your_project worker --loglevel=info")
            return

        # Determine which models to process
        if options["model"] == "all":
            models_to_process = ["Ayurvedha", "Siddha", "Unani", "ICD11Term"]
        else:
            models_to_process = [options["model"]]

        total_processed = 0

        # Process each model
        for model_name in models_to_process:
            processed_count = self.process_model_optimized(
                model_name,
                options["batch_size"],
                options["workers"],
                options["force"],
                options.get("limit"),
            )
            total_processed += processed_count

        self.stdout.write(
            self.style.SUCCESS(
                f"✅ Completed! Total records processed: {total_processed}"
            )
        )

        # Show monitoring info
        self.stdout.write("📊 Monitor progress with: --monitor flag")

    def check_celery_workers(self) -> bool:
        """Check if Celery workers are running"""
        try:
            from celery import current_app

            # Get active workers
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

    def process_model_optimized(
        self,
        model_name: str,
        batch_size: int,
        max_workers: int,
        force: bool,
        limit: int = None,
    ) -> int:
        """Process a single model using optimized Celery tasks"""

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

        # Create batches for parallel processing
        batches = [
            record_ids[i : i + batch_size]
            for i in range(0, len(record_ids), batch_size)
        ]

        total_batches = len(batches)
        self.stdout.write(
            f"📦 Created {total_batches} batches of ~{batch_size} records each"
        )

        # Initialize progress tracking
        task_name = f"{model_name}_{int(time.time())}"
        update_progress.delay(task_name, 0, total_count, model_name)

        # Process batches in groups of max_workers
        processed_count = 0

        for batch_group_start in range(0, total_batches, max_workers):
            batch_group_end = min(batch_group_start + max_workers, total_batches)
            current_batches = batches[batch_group_start:batch_group_end]

            self.stdout.write(
                f"⚙️  Processing batch group {batch_group_start // max_workers + 1} "
                f"({len(current_batches)} batches)"
            )

            # Create group of batch tasks
            job = group(
                generate_batch_embeddings.s(model_name, batch, force)
                for batch in current_batches
            )

            # Execute the group and wait for results
            result = job.apply_async()
            batch_results = result.get()  # This blocks until all tasks complete

            # Process results
            for batch_result in batch_results:
                processed_count += batch_result["successful"]

                if batch_result["failed"] > 0:
                    self.stdout.write(
                        self.style.WARNING(
                            f"⚠️  {batch_result['failed']} failed in batch"
                        )
                    )

            # Update progress
            update_progress.delay(task_name, processed_count, total_count, model_name)

            # Progress update
            progress = (processed_count / total_count) * 100
            self.stdout.write(
                f"📈 {model_name}: {processed_count}/{total_count} ({progress:.1f}%)"
            )

            # Brief pause between batch groups to avoid overwhelming the API
            if batch_group_end < total_batches:
                time.sleep(2)

        self.stdout.write(
            self.style.SUCCESS(f"✅ {model_name}: Completed {processed_count} records")
        )

        return processed_count

    def monitor_progress(self):
        """Monitor progress of running embedding tasks"""

        self.stdout.write("📊 Monitoring embedding generation progress...")
        self.stdout.write("Press Ctrl+C to exit monitoring\n")

        try:
            while True:
                # Get all progress keys from Redis
                progress_keys = cache.keys("embedding_progress:*")

                if not progress_keys:
                    self.stdout.write("⚪ No active embedding tasks found")
                    break

                # Display progress for each active task
                for key in progress_keys:
                    progress = cache.get(key)
                    if progress:
                        model = progress.get("model", "Unknown")
                        current = progress["current"]
                        total = progress["total"]
                        percentage = progress["percentage"]

                        # Create progress bar
                        bar_length = 40
                        filled_length = int(bar_length * percentage / 100)
                        bar = "█" * filled_length + "-" * (bar_length - filled_length)

                        self.stdout.write(
                            f"{model:12} [{bar}] {percentage:6.1f}% ({current}/{total})"
                        )

                self.stdout.write("")  # Empty line
                time.sleep(5)  # Update every 5 seconds

        except KeyboardInterrupt:
            self.stdout.write("\n👋 Monitoring stopped")
