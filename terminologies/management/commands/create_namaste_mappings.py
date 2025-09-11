import logging

from django.core.management.base import BaseCommand

from terminologies.services.mapping_service import NamasteToICDMappingService


class Command(BaseCommand):
    help = "Create fuzzy mappings from NAMASTE terms to ICD-11"

    def add_arguments(self, parser):
        parser.add_argument(
            "--system",
            choices=["ayurveda", "siddha", "unani", "all"],
            default="all",
            help="Which NAMASTE system to process",
        )
        parser.add_argument(
            "--threshold", type=float, default=0.3, help="ICD similarity threshold"
        )
        parser.add_argument(
            "--cross-threshold",
            type=float,
            default=0.5,
            help="Cross-system similarity threshold",
        )

    def handle(self, *args, **options):
        # Setup logging
        logging.basicConfig(level=logging.INFO)

        service = NamasteToICDMappingService(
            similarity_threshold=options["threshold"],
            cross_system_threshold=options["cross_threshold"],
        )

        total_processed = 0
        total_created = 0

        if options["system"] == "all":
            processed, created = service.process_all_systems()
            total_processed += processed
            total_created += created
        else:
            method_name = f"process_all_{options['system']}_terms"
            if hasattr(service, method_name):
                processed, created = getattr(service, method_name)()
                total_processed += processed
                total_created += created

        # Show statistics
        stats = service.get_mapping_stats()

        self.stdout.write(
            self.style.SUCCESS(
                f"Mapping completed!\n"
                f"Total processed: {total_processed}\n"
                f"Total created: {total_created}\n"
                f"Total mappings in DB: {stats['total_mappings']}\n"
                f"High confidence mappings: {stats['confidence_distribution']['high_confidence']}"
            )
        )
