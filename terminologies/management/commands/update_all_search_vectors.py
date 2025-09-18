from django.contrib.postgres.search import SearchVector
from django.core.management.base import BaseCommand
from django.db import transaction

from terminologies.models import Ayurvedha, ICD11Term, Siddha, Unani


class Command(BaseCommand):
    help = "Update search vectors for all terminology models"

    def add_arguments(self, parser):
        parser.add_argument(
            "--batch-size",
            type=int,
            default=1000,
            help="Number of records to process in each batch (default: 1000)",
        )
        parser.add_argument(
            "--models",
            nargs="+",
            choices=["icd11", "ayurveda", "siddha", "unani", "all"],
            default=["all"],
            help="Specify which models to update (default: all)",
        )

    def handle(self, *args, **options):
        batch_size = options["batch_size"]
        selected_models = options["models"]

        if "all" in selected_models or "icd11" in selected_models:
            self.update_icd11_vectors(batch_size)

        if "all" in selected_models or "ayurveda" in selected_models:
            self.update_ayurveda_vectors(batch_size)

        if "all" in selected_models or "siddha" in selected_models:
            self.update_siddha_vectors(batch_size)

        if "all" in selected_models or "unani" in selected_models:
            self.update_unani_vectors(batch_size)

        self.stdout.write(
            self.style.SUCCESS("All search vectors updated successfully!")
        )

    def update_icd11_vectors(self, batch_size):
        self.stdout.write("=== Updating ICD11Term search vectors ===")
        self._update_model_vectors(
            ICD11Term,
            batch_size,
            (
                SearchVector("title", weight="A", config="english")
                + SearchVector("code", weight="A", config="english")
                + SearchVector("definition", weight="B", config="english")
                + SearchVector("long_definition", weight="C", config="english")
            ),
        )

    def update_ayurveda_vectors(self, batch_size):
        self.stdout.write("=== Updating Ayurveda search vectors ===")
        self._update_model_vectors(
            Ayurvedha,
            batch_size,
            (
                SearchVector("code", weight="A", config="english")
                + SearchVector("english_name", weight="A", config="english")
                + SearchVector("hindi_name", weight="B", config="simple")
                + SearchVector("diacritical_name", weight="B", config="simple")
                + SearchVector("description", weight="C", config="english")
            ),
        )

    def update_siddha_vectors(self, batch_size):
        self.stdout.write("=== Updating Siddha search vectors ===")
        self._update_model_vectors(
            Siddha,
            batch_size,
            (
                SearchVector("code", weight="A", config="english")
                + SearchVector("english_name", weight="A", config="english")
                + SearchVector("tamil_name", weight="B", config="simple")
                + SearchVector("romanized_name", weight="B", config="english")
                + SearchVector("description", weight="C", config="english")
                + SearchVector("reference", weight="D", config="english")
            ),
        )

    def update_unani_vectors(self, batch_size):
        self.stdout.write("=== Updating Unani search vectors ===")
        self._update_model_vectors(
            Unani,
            batch_size,
            (
                SearchVector("code", weight="A", config="english")
                + SearchVector("english_name", weight="A", config="english")
                + SearchVector("arabic_name", weight="B", config="simple")
                + SearchVector("romanized_name", weight="B", config="english")
                + SearchVector("description", weight="C", config="english")
                + SearchVector("reference", weight="D", config="english")
            ),
        )

    def _update_model_vectors(self, model_class, batch_size, search_vector_config):
        total_count = model_class.objects.count()
        self.stdout.write(f"Total {model_class.__name__} records: {total_count}")

        updated_count = 0

        for start in range(0, total_count, batch_size):
            end = min(start + batch_size, total_count)

            with transaction.atomic():
                batch_updated = model_class.objects.filter(
                    pk__in=model_class.objects.all()[start:end].values_list(
                        "pk", flat=True
                    )
                ).update(search_vector=search_vector_config)

                updated_count += batch_updated

                self.stdout.write(
                    f"Processed batch {start + 1}-{end} ({batch_updated} updated)"
                )

        self.stdout.write(f"Total {model_class.__name__} updated: {updated_count}\n")
