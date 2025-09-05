from django.core.management.base import BaseCommand
import pandas as pd
from terminologies.models import AyurvedhaModel

class Command(BaseCommand):
    help = "Load Ayurveda terms from an XLSX file into the database"

    def add_arguments(self, parser):
        parser.add_argument(
            '--file',
            type=str,
            help='Path to the Ayurveda XLSX file',
            required=True
        )

    def handle(self, *args, **kwargs):
        file_path = kwargs['file']
        df = pd.read_excel(file_path)

        for _, row in df.iterrows():
            code = str(row['Code']).strip()
            devanagari_name = str(row['Name Devnagari']).strip() if not pd.isna(row['Name Devnagari']) else ""
            diacritical_name = str(row['Name Diacritical']).strip() if not pd.isna(row['Name Diacritical']) else ""
            english_name = str(row['Name English']).strip() if not pd.isna(row['Name English']) else ""
            description = str(row['Description']).strip() if not pd.isna(row['Description']) else ""

            AyurvedhaModel.objects.update_or_create(
                code=code,
                defaults={
                    'hindi_name': devanagari_name,
                    'diacritical_name': diacritical_name,
                    'english_name': english_name,
                    'description': description,
                }
            )

        self.stdout.write(self.style.SUCCESS("Ayurveda data loaded successfully!"))