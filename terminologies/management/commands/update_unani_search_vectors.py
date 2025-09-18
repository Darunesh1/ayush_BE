from django.contrib.postgres.search import SearchVector
from django.core.management.base import BaseCommand
from django.db import transaction

from terminologies.models import Unani


class Command(BaseCommand):
    help = "Update search vectors for all Unani terms"

    def add_arguments(self, parser):
        parser.add_argument(
            "--batch-size",
            type=int,
            default=1000,
            help="Number of records to process in each batch (default: 1000)",
        )

    def handle(self, *args, **options):
        batch_size = options["batch_size"]

        self.stdout.write("Starting search vector update for Unani terms...")

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

        self.stdout.write(
            self.style.SUCCESS(
                f"Unani search vector update complete!\n"
                f"Total updated: {updated_count}\n"
                f"Records with search vectors: {search_vector_count}"
            )
        )
