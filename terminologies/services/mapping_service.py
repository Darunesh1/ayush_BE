import logging

from django.contrib.postgres.search import TrigramSimilarity
from django.db import transaction
from django.db.models import F, Q

from terminologies.models import Ayurvedha, ICD11Term, Siddha, TermMapping, Unani

logger = logging.getLogger(__name__)


class NamasteToICDMappingService:
    def __init__(self, similarity_threshold=0.3, cross_system_threshold=0.5):
        self.similarity_threshold = similarity_threshold
        self.cross_system_threshold = cross_system_threshold
        self.system_models = {"ayurveda": Ayurvedha, "siddha": Siddha, "unani": Unani}

    def find_best_icd_match(self, namaste_term):
        """Find best matching ICD-11 term for given NAMASTE term"""
        if not namaste_term.english_name:
            return []

        search_text = namaste_term.english_name

        # Enhanced search across ICD title and potentially other fields
        icd_matches = (
            ICD11Term.objects.annotate(
                title_sim=TrigramSimilarity("title", search_text)
            )
            .filter(title_sim__gt=self.similarity_threshold)
            .order_by("-title_sim")[:5]
        )

        return list(icd_matches)

    def find_cross_system_matches(self, source_term, source_system):
        """Find matching terms in other NAMASTE systems"""
        if not source_term.english_name:
            return {}

        search_text = source_term.english_name
        cross_matches = {}

        # Search in other systems (excluding source system)
        for system_name, model_class in self.system_models.items():
            if system_name != source_system:
                matches = (
                    model_class.objects.annotate(
                        similarity=TrigramSimilarity("english_name", search_text)
                    )
                    .filter(similarity__gt=self.cross_system_threshold)
                    .order_by("-similarity")[:3]
                )
                cross_matches[system_name] = list(matches)

        return cross_matches

    def _get_existing_mapping(self, namaste_term, source_system):
        """Check if mapping already exists for the given term"""
        filter_kwargs = {f"primary_{source_system}_term": namaste_term}
        return TermMapping.objects.filter(**filter_kwargs).first()

    def _add_cross_system_matches(self, mapping_data, cross_matches, source_system):
        """Add cross-system matches to mapping data"""
        for system in ["ayurveda", "siddha", "unani"]:
            if (
                system != source_system
                and system in cross_matches
                and cross_matches[system]
            ):
                best_match = cross_matches[system][0]
                if hasattr(best_match, "similarity"):
                    mapping_data[f"cross_{system}_term"] = best_match
                    mapping_data[f"cross_{system}_similarity"] = best_match.similarity

    def _calculate_confidence_score(self, mapping_data):
        """Calculate confidence score from all similarity scores"""
        similarities = [mapping_data["icd_similarity"]]

        # Add cross-system similarities
        for system in ["ayurveda", "siddha", "unani"]:
            sim_key = f"cross_{system}_similarity"
            if mapping_data.get(sim_key):
                similarities.append(mapping_data[sim_key])

        return sum(similarities) / len(similarities) if similarities else 0.0

    def create_mapping_for_namaste_term(self, namaste_term, source_system):
        """Create complete mapping for a NAMASTE term"""

        # Input validation
        if not namaste_term or not namaste_term.english_name:
            logger.warning(
                f"Skipping {source_system} term with no english_name: {namaste_term}"
            )
            return None

        if source_system not in self.system_models:
            raise ValueError(f"Invalid source_system: {source_system}")

        # Check for existing mapping to prevent duplicates
        existing_mapping = self._get_existing_mapping(namaste_term, source_system)
        if existing_mapping:
            logger.debug(
                f"Mapping already exists for {source_system} term: {namaste_term.english_name}"
            )
            return existing_mapping

        # Find best ICD match
        icd_matches = self.find_best_icd_match(namaste_term)
        if not icd_matches:
            logger.info(
                f"No ICD match found for {source_system} term: {namaste_term.english_name}"
            )
            return None

        best_icd = icd_matches[0]

        # Find cross-system matches
        cross_matches = self.find_cross_system_matches(namaste_term, source_system)

        # Prepare mapping data
        mapping_data = {
            "icd_term": best_icd,
            "icd_similarity": best_icd.title_sim,
            "source_system": source_system,
            f"primary_{source_system}_term": namaste_term,  # Dynamic field assignment
        }

        # Set cross-system matches
        self._add_cross_system_matches(mapping_data, cross_matches, source_system)

        # Calculate confidence score
        mapping_data["confidence_score"] = self._calculate_confidence_score(
            mapping_data
        )

        # Create mapping with transaction safety
        try:
            with transaction.atomic():
                mapping = TermMapping.objects.create(**mapping_data)
                logger.debug(
                    f"Created mapping for {source_system}: {namaste_term.english_name} -> {best_icd.title}"
                )
                return mapping
        except Exception as e:
            logger.error(
                f"Failed to create mapping for {source_system} term {namaste_term.english_name}: {str(e)}"
            )
            return None

    def _process_system_terms(self, system_name):
        """Generic method to process terms from any system"""
        model_class = self.system_models[system_name]
        total_count = model_class.objects.filter(english_name__isnull=False).count()
        processed = 0
        created = 0

        logger.info(f"Starting processing of {total_count} {system_name} terms")

        for term in model_class.objects.filter(english_name__isnull=False).iterator():
            try:
                mapping = self.create_mapping_for_namaste_term(term, system_name)
                if mapping:
                    created += 1
                processed += 1

                # Progress tracking every 100 items
                if processed % 100 == 0:
                    logger.info(
                        f"Processed {processed}/{total_count} {system_name} terms, created {created} mappings"
                    )

            except Exception as e:
                logger.error(
                    f"Error processing {system_name} term {term.english_name}: {str(e)}"
                )
                processed += 1  # Still count as processed even if failed

        logger.info(
            f"Completed {system_name}: {processed} processed, {created} mappings created"
        )
        return processed, created

    def process_all_ayurveda_terms(self):
        """Process all Ayurveda terms"""
        return self._process_system_terms("ayurveda")

    def process_all_siddha_terms(self):
        """Process all Siddha terms"""
        return self._process_system_terms("siddha")

    def process_all_unani_terms(self):
        """Process all Unani terms"""
        return self._process_system_terms("unani")

    def process_all_systems(self):
        """Process all NAMASTE systems"""
        total_processed = 0
        total_created = 0

        for system_name in self.system_models.keys():
            processed, created = self._process_system_terms(system_name)
            total_processed += processed
            total_created += created

        logger.info(
            f"All systems completed: {total_processed} total processed, {total_created} total created"
        )
        return total_processed, total_created

    def get_mapping_stats(self):
        """Get statistics about current mappings"""
        stats = {
            "total_mappings": TermMapping.objects.count(),
            "by_system": {},
            "confidence_distribution": {},
        }

        # Stats by system
        for system in self.system_models.keys():
            filter_kwargs = {f"primary_{system}_term__isnull": False}
            count = TermMapping.objects.filter(**filter_kwargs).count()
            stats["by_system"][system] = count

        # Confidence distribution
        high_confidence = TermMapping.objects.filter(confidence_score__gte=0.7).count()
        medium_confidence = TermMapping.objects.filter(
            confidence_score__gte=0.5, confidence_score__lt=0.7
        ).count()
        low_confidence = TermMapping.objects.filter(confidence_score__lt=0.5).count()

        stats["confidence_distribution"] = {
            "high_confidence": high_confidence,
            "medium_confidence": medium_confidence,
            "low_confidence": low_confidence,
        }

        return stats
