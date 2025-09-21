# namasthe_mapping/management/commands/generate_embeddings.py

import logging

from django.apps import apps
from django.core.management.base import BaseCommand
from django.utils import timezone

from namasthe_mapping.services import HuggingFaceAPIService

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Generate BioBERT embeddings using Hugging Face API"

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
            default=5,
            help="API batch size (keep small for stability)",
        )
        parser.add_argument(
            "--force",
            action="store_true",
            help="Force regenerate embeddings even if they exist",
        )
        parser.add_argument(
            "--limit", type=int, help="Limit number of records to process (for testing)"
        )

    def handle(self, *args, **options):
        self.stdout.write(
            self.style.SUCCESS(
                "🚀 Starting BioBERT embedding generation via Hugging Face API..."
            )
        )

        # Initialize API service
        try:
            service = HuggingFaceAPIService()
            model_info = service.get_model_info()
            self.stdout.write(f"📊 Service info: {model_info}")
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f"❌ Failed to initialize API service: {str(e)}")
            )
            self.stdout.write(
                "💡 Make sure HUGGINGFACE_API_TOKEN is set in settings.py"
            )
            return

        # Determine which models to process
        if options["model"] == "all":
            models_to_process = ["Ayurvedha", "Siddha", "Unani", "ICD11Term"]
        else:
            models_to_process = [options["model"]]

        total_processed = 0

        # Process each model
        for model_name in models_to_process:
            processed_count = self.process_model(
                model_name,
                service,
                options["batch_size"],
                options["force"],
                options.get("limit"),
            )
            total_processed += processed_count

        self.stdout.write(
            self.style.SUCCESS(
                f"✅ Completed! Total records processed: {total_processed}"
            )
        )

        # API usage info
        self.stdout.write(
            "💡 Check your Hugging Face API usage at: https://huggingface.co/settings/tokens"
        )

    def process_model(
        self,
        model_name: str,
        service: HuggingFaceAPIService,
        batch_size: int,
        force: bool,
        limit: int = None,
    ) -> int:
        """Process a single model"""

        self.stdout.write(f"\n📋 Processing {model_name}...")

        # Get the model class
        if model_name == "ICD11Term":
            Model = apps.get_model("terminologies", "ICD11Term")
        else:
            Model = apps.get_model("terminologies", model_name)

        # Get records that need embedding generation
        if force:
            queryset = Model.objects.all()
        else:
            queryset = Model.objects.filter(embedding__isnull=True)

        # Apply limit if specified
        if limit:
            queryset = queryset[:limit]

        total_count = queryset.count()

        if total_count == 0:
            self.stdout.write(f"⚪ No {model_name} records need embedding generation")
            return 0

        self.stdout.write(f"🎯 Found {total_count} {model_name} records to process")

        processed_count = 0

        # Process in batches
        for offset in range(0, total_count, batch_size):
            batch = list(queryset[offset : offset + batch_size])

            # Extract texts for embedding
            texts = []
            records = []

            for record in batch:
                text = record.get_embedding_text()
                if text and text.strip():  # Only process non-empty texts
                    texts.append(text)
                    records.append(record)

            if not texts:
                continue

            # Generate embeddings via API
            self.stdout.write(f"⚙️  Calling API for batch {offset // batch_size + 1}...")
            embeddings = service.generate_batch_embeddings(texts, batch_size=len(texts))

            # Store embeddings
            now = timezone.now()
            model_version = service.model_name

            for record, embedding in zip(records, embeddings):
                record.embedding = embedding
                record.embedding_updated_at = now
                record.embedding_model_version = model_version
                record.save(
                    update_fields=[
                        "embedding",
                        "embedding_updated_at",
                        "embedding_model_version",
                    ]
                )

            processed_count += len(records)

            # Progress update
            progress = (processed_count / total_count) * 100
            self.stdout.write(
                f"📈 {model_name}: {processed_count}/{total_count} ({progress:.1f}%)"
            )

        self.stdout.write(
            self.style.SUCCESS(f"✅ {model_name}: Completed {processed_count} records")
        )

        return processed_count
