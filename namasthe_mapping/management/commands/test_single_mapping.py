# management/commands/test_single_mapping.py

from django.contrib.contenttypes.models import ContentType
from django.core.management.base import BaseCommand
from django.db import transaction

from namasthe_mapping.models import ConceptMapping, TerminologyMapping
from namasthe_mapping.services import HuggingFaceAPIService
from terminologies.models import Ayurvedha, ICD11Term


class Command(BaseCommand):
    help = "Test single mapping creation manually"

    def handle(self, *args, **options):
        self.stdout.write("🧪 TESTING SINGLE MAPPING CREATION")

        try:
            # Get one Ayurveda term
            ayur_term = Ayurvedha.objects.filter(embedding__isnull=False).first()
            if not ayur_term:
                self.stdout.write("❌ No Ayurveda terms with embeddings")
                return

            self.stdout.write(f"📋 Source term: {ayur_term.english_name}")

            # Get mapping config
            mapping_config = TerminologyMapping.objects.filter(
                source_system="NAMASTE-Ayurveda", target_system="ICD-11"
            ).first()

            if not mapping_config:
                self.stdout.write("❌ No mapping configuration found")
                return

            self.stdout.write(
                f"⚙️ Config: {mapping_config.name} (threshold: {mapping_config.similarity_threshold})"
            )

            # Get ICD-11 terms (limited for testing)
            icd11_terms = list(
                ICD11Term.objects.filter(embedding__isnull=False).values(
                    "id", "code", "title", "embedding"
                )[:1000]  # Test with 1000 terms
            )

            self.stdout.write(
                f"🎯 Target terms: {len(icd11_terms)} ICD-11 terms loaded"
            )

            # Test similarity calculation
            service = HuggingFaceAPIService(model_preference="biobert")
            icd11_embeddings = [term["embedding"] for term in icd11_terms]

            self.stdout.write(
                f"🔧 Service: {service.model_name} (dim: {service.embedding_dim})"
            )

            # Find matches
            matches = service.find_best_matches(
                ayur_term.embedding,
                icd11_embeddings,
                similarity_threshold=0.5,  # Lower threshold for testing
                top_k=5,
            )

            self.stdout.write(f"🔍 Found {len(matches)} matches above 0.5 threshold:")

            if not matches:
                self.stdout.write(
                    "❌ No matches found - this explains why no mappings were created"
                )
                return

            # Show matches
            for i, match in enumerate(matches):
                icd11_term = icd11_terms[match["index"]]
                self.stdout.write(
                    f"  {i + 1}. {icd11_term['title'][:50]}... "
                    f"(similarity: {match['similarity']:.3f})"
                )

            # Test with your actual threshold
            threshold_matches = service.find_best_matches(
                ayur_term.embedding,
                icd11_embeddings,
                similarity_threshold=mapping_config.similarity_threshold,
                top_k=3,
            )

            self.stdout.write(
                f"\n🎯 With threshold {mapping_config.similarity_threshold}: {len(threshold_matches)} matches"
            )

            if threshold_matches:
                self.stdout.write(
                    "✅ Matches found with your threshold - mappings should be created"
                )

                # Try creating one mapping manually
                best_match = threshold_matches[0]
                icd11_term = icd11_terms[best_match["index"]]

                source_content_type = ContentType.objects.get_for_model(Ayurvedha)

                # Check if mapping already exists
                existing = ConceptMapping.objects.filter(
                    mapping=mapping_config,
                    source_content_type=source_content_type,
                    source_object_id=ayur_term.id,
                    target_concept_id=icd11_term["id"],
                ).first()

                if existing:
                    self.stdout.write(f"ℹ️ Mapping already exists: ID {existing.id}")
                else:
                    # Create test mapping
                    with transaction.atomic():
                        test_mapping = ConceptMapping.objects.create(
                            mapping=mapping_config,
                            source_content_type=source_content_type,
                            source_object_id=ayur_term.id,
                            target_concept_id=icd11_term["id"],
                            relationship="related-to",
                            similarity_score=best_match["similarity"],
                            confidence_score=min(best_match["similarity"] + 0.05, 1.0),
                            mapping_method="dmis-lab/biobert-v1.1-test",
                            is_high_confidence=(best_match["similarity"] >= 0.9),
                            needs_review=(best_match["similarity"] < 0.85),
                        )

                        self.stdout.write(
                            f"✅ Test mapping created: ID {test_mapping.id}"
                        )
                        self.stdout.write(
                            f"   {ayur_term.english_name} → {icd11_term['title']}"
                        )
                        self.stdout.write(
                            f"   Similarity: {best_match['similarity']:.3f}"
                        )

            else:
                self.stdout.write(
                    "❌ No matches with your threshold - need to lower it"
                )

        except Exception as e:
            self.stdout.write(f"❌ Error: {str(e)}")
            import traceback

            traceback.print_exc()
