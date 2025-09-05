
import pandas as pd
from django.core.management.base import BaseCommand
from terminologies.models import SiddhaModel

class Command(BaseCommand):
    help = "Load Siddha terms from an Excel file into the database"

    def add_arguments(self, parser):
        parser.add_argument(
            '--file',
            type=str,
            required=True,
            help='Path to the Excel file containing Siddha data'
        )

    def handle(self, *args, **options):
        file_path = options['file']
        df = pd.read_excel(file_path)

        for _, row in df.iterrows():
            # Skip empty rows if 'Code' is missing
            if pd.isna(row['Code']):
                continue

            SiddhaModel.objects.update_or_create(
                code=row['Code'],
                defaults={
                    'tamil_name': row.get('Term', None),
                    'romanized_name': row.get('Word', None),
                    'english_name': row.get('Translation', None),
                    'description': row.get('Definition', None),
                    'reference': row.get('Reference', None),
                }
            )

        self.stdout.write(self.style.SUCCESS(f"Successfully loaded Siddha data from {file_path}"))
