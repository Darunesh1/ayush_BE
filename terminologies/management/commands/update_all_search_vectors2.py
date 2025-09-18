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

        total_count = ICD11Term.objects.count()
        self.stdout.write(f"Total ICD11Term records to process: {total_count}")

        if total_count == 0:
            self.stdout.write("No ICD11Term records found.")
            return

        updated_count = 0

        for start in range(0, total_count, batch_size):
            end = min(start + batch_size, total_count)

            with transaction.atomic():
                batch_updated = ICD11Term.objects.filter(
                    pk__in=ICD11Term.objects.all()[start:end].values_list(
                        "pk", flat=True
                    )
                ).update(
                    search_vector=(
                        SearchVector("title", weight="A", config="english")
                        + SearchVector("code", weight="A", config="english")
                        + SearchVector("definition", weight="B", config="english")
                        + SearchVector("long_definition", weight="C", config="english")
                    )
                )

                updated_count += batch_updated

                self.stdout.write(
                    f"Processed batch {start + 1}-{end} ({batch_updated} updated)"
                )

        # Verify results
        search_vector_count = ICD11Term.objects.filter(
            search_vector__isnull=False
        ).count()

        self.stdout.write(f"Total ICD11Term updated: {updated_count}")
        self.stdout.write(f"Records with search vectors: {search_vector_count}\n")

    def update_ayurveda_vectors(self, batch_size):
        self.stdout.write("=== Updating Ayurveda search vectors ===")

        total_count = Ayurvedha.objects.count()
        self.stdout.write(f"Total Ayurveda records to process: {total_count}")

        if total_count == 0:
            self.stdout.write("No Ayurveda records found.")
            return

        updated_count = 0

        for start in range(0, total_count, batch_size):
            end = min(start + batch_size, total_count)

            with transaction.atomic():
                batch_updated = Ayurvedha.objects.filter(
                    pk__in=Ayurvedha.objects.all()[start:end].values_list(
                        "pk", flat=True
                    )
                ).update(
                    search_vector=(
                        SearchVector("code", weight="A", config="english")
                        + SearchVector("english_name", weight="A", config="english")
                        + SearchVector("hindi_name", weight="B", config="simple")
                        + SearchVector("diacritical_name", weight="B", config="simple")
                        + SearchVector("description", weight="C", config="english")
                    )
                )

                updated_count += batch_updated

                self.stdout.write(
                    f"Processed batch {start + 1}-{end} ({batch_updated} updated)"
                )

        # Verify results
        search_vector_count = Ayurvedha.objects.filter(
            search_vector__isnull=False
        ).count()

        self.stdout.write(f"Total Ayurveda updated: {updated_count}")
        self.stdout.write(f"Records with search vectors: {search_vector_count}\n")

    def update_siddha_vectors(self, batch_size):
        self.stdout.write("=== Updating Siddha search vectors ===")

        total_count = Siddha.objects.count()
        self.stdout.write(f"Total Siddha records to process: {total_count}")

        if total_count == 0:
            self.stdout.write("No Siddha records found.")
            return

        updated_count = 0

        for start in range(0, total_count, batch_size):
            end = min(start + batch_size, total_count)

            with transaction.atomic():
                batch_updated = Siddha.objects.filter(
                    pk__in=Siddha.objects.all()[start:end].values_list("pk", flat=True)
                ).update(
                    search_vector=(
                        SearchVector("code", weight="A", config="english")
                        + SearchVector("english_name", weight="A", config="english")
                        + SearchVector("tamil_name", weight="B", config="simple")
                        + SearchVector("romanized_name", weight="B", config="english")
                        + SearchVector("description", weight="C", config="english")
                        + SearchVector("reference", weight="D", config="english")
                    )
                )

                updated_count += batch_updated

                self.stdout.write(
                    f"Processed batch {start + 1}-{end} ({batch_updated} updated)"
                )

        # Verify results
        search_vector_count = Siddha.objects.filter(search_vector__isnull=False).count()

        self.stdout.write(f"Total Siddha updated: {updated_count}")
        self.stdout.write(f"Records with search vectors: {search_vector_count}\n")

    def update_unani_vectors(self, batch_size):
        self.stdout.write("=== Updating Unani search vectors ===")

        total_count = Unani.objects.count()
        self.stdout.write(f"Total Unani records to process: {total_count}")

        if total_count == 0:
            self.stdout.write("No Unani records found.")
            return

        updated_count = 0

        for start in range(0, total_count, batch_size):
            end = min(start + batch_size, total_count)

            with transaction.atomic():
                batch_updated = Unani.objects.filter(
                    pk__in=Unani.objects.all()[start:end].values_list("pk", flat=True)
                ).update(
                    search_vector=(
                        SearchVector("code", weight="A", config="english")
                        + SearchVector("english_name", weight="A", config="english")
                        + SearchVector("arabic_name", weight="B", config="simple")
                        + SearchVector("romanized_name", weight="B", config="english")
                        + SearchVector("description", weight="C", config="english")
                        + SearchVector("reference", weight="D", config="english")
                    )
                )

                updated_count += batch_updated

                self.stdout.write(
                    f"Processed batch {start + 1}-{end} ({batch_updated} updated)"
                )

        # Verify results
        search_vector_count = Unani.objects.filter(search_vector__isnull=False).count()

        self.stdout.write(f"Total Unani updated: {updated_count}")
        self.stdout.write(f"Records with search vectors: {search_vector_count}\n")
