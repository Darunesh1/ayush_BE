# management/commands/debug_mappings.py

from django.core.management.base import BaseCommand

from namasthe_mapping.models import TerminologyMapping
from terminologies.models import Ayurvedha, ICD11Term


class Command(BaseCommand):
    help = "Debug mapping creation issues"

    def handle(self, *args, **options):
        # Check Ayurveda terms with embeddings
        ayur_total = Ayurvedha.objects.count()
        ayur_embedded = Ayurvedha.objects.filter(embedding__isnull=False).count()

        self.stdout.write(f"🔍 Ayurveda Terms:")
        self.stdout.write(f"  • Total: {ayur_total}")
        self.stdout.write(f"  • With embeddings: {ayur_embedded}")

        if ayur_embedded > 0:
            sample = Ayurvedha.objects.filter(embedding__isnull=False).first()
            self.stdout.write(
                f"  • Sample embedding length: {len(sample.embedding) if sample.embedding else 'None'}"
            )

        # Check ICD-11 terms
        icd11_total = ICD11Term.objects.count()
        icd11_embedded = ICD11Term.objects.filter(embedding__isnull=False).count()

        self.stdout.write(f"\n🔍 ICD-11 Terms:")
        self.stdout.write(f"  • Total: {icd11_total}")
        self.stdout.write(f"  • With embeddings: {icd11_embedded}")

        if icd11_embedded > 0:
            sample = ICD11Term.objects.filter(embedding__isnull=False).first()
            self.stdout.write(
                f"  • Sample embedding length: {len(sample.embedding) if sample.embedding else 'None'}"
            )

        # Check mapping config
        mapping_configs = TerminologyMapping.objects.all()
        self.stdout.write(f"\n🔍 Mapping Configurations:")
        for config in mapping_configs:
            self.stdout.write(
                f"  • {config.name}: {config.source_system} → {config.target_system}"
            )
            self.stdout.write(f"    Threshold: {config.similarity_threshold}")
            self.stdout.write(f"    Status: {config.status}")

        # Test a simple similarity calculation
        if ayur_embedded > 0 and icd11_embedded > 0:
            from namasthe_mapping.services import HuggingFaceAPIService

            ayur_sample = Ayurvedha.objects.filter(embedding__isnull=False).first()
            icd11_samples = list(ICD11Term.objects.filter(embedding__isnull=False)[:10])

            self.stdout.write(f"\n🧪 Testing similarity calculation:")
            self.stdout.write(f"  • Ayurveda term: {ayur_sample.english_name}")

            try:
                service = HuggingFaceAPIService(model_preference="biobert")
                icd11_embeddings = [term.embedding for term in icd11_samples]

                matches = service.find_best_matches(
                    ayur_sample.embedding,
                    icd11_embeddings,
                    similarity_threshold=0.5,  # Lower threshold for testing
                    top_k=3,
                )

                self.stdout.write(
                    f"  • Found {len(matches)} matches above 0.5 similarity"
                )
                for i, match in enumerate(matches):
                    icd11_term = icd11_samples[match["index"]]
                    self.stdout.write(
                        f"    {i + 1}. {icd11_term.title} (similarity: {match['similarity']:.3f})"
                    )

            except Exception as e:
                self.stdout.write(f"  • Error: {str(e)}")
