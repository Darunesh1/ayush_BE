# terminologies/management/commands/load_icd11.py

import pandas as pd
from django.core.management.base import BaseCommand
from terminologies.models import ICD11Term, ICDClassKind
from datetime import datetime

class Command(BaseCommand):
    help = "Load ICD-11 data from a CSV file into the database in batches (Option 2: fill missing Foundation URI)."

    def add_arguments(self, parser):
        parser.add_argument(
            '--file',
            type=str,
            required=True,
            help='Path to the ICD-11 CSV file.'
        )
        parser.add_argument(
            '--batch_size',
            type=int,
            default=500,
            help='Number of records to insert per batch.'
        )

    def handle(self, *args, **options):
        file_path = options['file']
        batch_size = options['batch_size']

        self.stdout.write(f"Loading ICD-11 data from {file_path} with batch size {batch_size}...")

        df = pd.read_csv(file_path, low_memory=False)

        batch = []
        inserted_count = 0

        for idx, row in df.iterrows():
            foundation_uri = row.get('Foundation URI')
            if pd.isna(foundation_uri):
                # Fill missing Foundation URI with unique placeholder
                foundation_uri = f"missing-foundation-uri-{idx}"

            # Handle ClassKind: create if not exists
            class_kind_name = row.get('ClassKind')
            if pd.isna(class_kind_name):
                class_kind_name = 'Unknown'
            class_kind, _ = ICDClassKind.objects.get_or_create(name=class_kind_name)

            # Parse boolean fields
            is_residual = str(row.get('IsResidual', 'FALSE')).strip().upper() == 'TRUE'
            is_leaf = str(row.get('isLeaf', 'FALSE')).strip().upper() == 'TRUE'

            # Parse version date from '8.0Y' if possible
            version_raw = row.get('8.0Y')
            version_date = None
            if pd.notna(version_raw):
                try:
                    version_date = datetime.strptime(version_raw, "%Y-%m-%d").date()
                except Exception:
                    version_date = None

            term = ICD11Term(
                foundation_uri=foundation_uri,
                linearization_uri=row.get('Linearization (release) URI'),
                code=row.get('8.0Y'),
                title=row.get('Title'),
                class_kind=class_kind,
                depth_in_kind=row.get('DepthInKind'),
                is_residual=is_residual,
                primary_location=row.get('PrimaryLocation'),
                chapter_no=row.get('ChapterNo'),
                browser_link=row.get('BrowserLink'),
                icat_link=row.get('iCatLink'),
                is_leaf=is_leaf,
                no_of_non_residual_children=row.get('noOfNonResidualChildren'),
                version_date=version_date
            )

            batch.append(term)

            # Bulk insert in batches
            if len(batch) >= batch_size:
                ICD11Term.objects.bulk_create(batch)
                inserted_count += len(batch)
                self.stdout.write(f"Inserted {inserted_count} records so far...")
                batch = []

        # Insert remaining records
        if batch:
            ICD11Term.objects.bulk_create(batch)
            inserted_count += len(batch)

        self.stdout.write(self.style.SUCCESS(f"ICD-11 data loaded successfully. Total records: {inserted_count}"))
