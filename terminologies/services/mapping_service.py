from django.contrib.postgres.search import TrigramSimilarity
from django.db.models import F, Q

from terminologies.models import Ayurvedha, ICD11Term, Siddha, TermMapping, Unani


class NamasteToICDMappingService:
    def __init__(self, similarity_threshold=0.3, cross_system_threshold=0.5):
        self.similarity_threshold = similarity_threshold
        self.cross_system_threshold = cross_system_threshold

    def find_best_icd_match(self, namaste_term):
        """Find best matching ICD-11 term for given NAMASTE term"""
        search_text = namaste_term.english_name

        # Enhanced search across ICD title and potentially other fields
        icd_matches = (
            ICD11Term.objects.annotate(
                title_sim=TrigramSimilarity("title", search_text)
            )
            .filter(title_sim__gt=self.similarity_threshold)
            .order_by("-title_sim")[:5]
        )  # Get top 5 candidates

        return list(icd_matches)

    def find_cross_system_matches(self, source_term, source_system):
        """Find matching terms in other NAMASTE systems"""
        search_text = source_term.english_name
        cross_matches = {}

        # Search in other systems (excluding source system)
        if source_system != "ayurveda":
            ayurveda_matches = (
                Ayurvedha.objects.annotate(
                    similarity=TrigramSimilarity("english_name", search_text)
                )
                .filter(similarity__gt=self.cross_system_threshold)
                .order_by("-similarity")[:3]
            )
            cross_matches["ayurveda"] = list(ayurveda_matches)

        if source_system != "siddha":
            siddha_matches = (
                Siddha.objects.annotate(
                    similarity=TrigramSimilarity("english_name", search_text)
                )
                .filter(similarity__gt=self.cross_system_threshold)
                .order_by("-similarity")[:3]
            )
            cross_matches["siddha"] = list(siddha_matches)

        if source_system != "unani":
            unani_matches = (
                Unani.objects.annotate(
                    similarity=TrigramSimilarity("english_name", search_text)
                )
                .filter(similarity__gt=self.cross_system_threshold)
                .order_by("-similarity")[:3]
            )
            cross_matches["unani"] = list(unani_matches)

        return cross_matches

    def create_mapping_for_namaste_term(self, namaste_term, source_system):
        """Create complete mapping for a NAMASTE term"""

        # Find best ICD match
        icd_matches = self.find_best_icd_match(namaste_term)
        if not icd_matches:
            return None  # No suitable ICD match found

        best_icd = icd_matches[0]  # Take the best match

        # Find cross-system matches
        cross_matches = self.find_cross_system_matches(namaste_term, source_system)

        # Prepare mapping data
        mapping_data = {
            "icd_term": best_icd,
            "icd_similarity": best_icd.title_sim,
            "source_system": source_system,
        }

        # Set primary term based on source system
        if source_system == "ayurveda":
            mapping_data["primary_ayurveda_term"] = namaste_term
        elif source_system == "siddha":
            mapping_data["primary_siddha_term"] = namaste_term
        elif source_system == "unani":
            mapping_data["primary_unani_term"] = namaste_term

        # Set cross-system matches
        if "ayurveda" in cross_matches and cross_matches["ayurveda"]:
            best_ayurveda = cross_matches["ayurveda"][0]
            mapping_data["cross_ayurveda_term"] = best_ayurveda
            mapping_data["cross_ayurveda_similarity"] = best_ayurveda.similarity

        if "siddha" in cross_matches and cross_matches["siddha"]:
            best_siddha = cross_matches["siddha"][0]
            mapping_data["cross_siddha_term"] = best_siddha
            mapping_data["cross_siddha_similarity"] = best_siddha.similarity

        if "unani" in cross_matches and cross_matches["unani"]:
            best_unani = cross_matches["unani"][0]
            mapping_data["cross_unani_term"] = best_unani
            mapping_data["cross_unani_similarity"] = best_unani.similarity

        # Calculate confidence score
        similarities = [best_icd.title_sim]
        if mapping_data.get("cross_ayurveda_similarity"):
            similarities.append(mapping_data["cross_ayurveda_similarity"])
        if mapping_data.get("cross_siddha_similarity"):
            similarities.append(mapping_data["cross_siddha_similarity"])
        if mapping_data.get("cross_unani_similarity"):
            similarities.append(mapping_data["cross_unani_similarity"])

        mapping_data["confidence_score"] = sum(similarities) / len(similarities)

        # Create mapping
        mapping = TermMapping.objects.create(**mapping_data)
        return mapping

    def process_all_ayurveda_terms(self):
        """Process all Ayurveda terms"""
        processed = 0
        created = 0

        for ayurveda_term in Ayurvedha.objects.filter(
            english_name__isnull=False
        ).iterator():
            try:
                mapping = self.create_mapping_for_namaste_term(
                    ayurveda_term, "ayurveda"
                )
                if mapping:
                    created += 1
                processed += 1
            except Exception as e:
                print(
                    f"Error processing Ayurveda term {ayurveda_term.english_name}: {str(e)}"
                )

        return processed, created

    def process_all_siddha_terms(self):
        """Process all Siddha terms"""
        processed = 0
        created = 0

        for siddha_term in Siddha.objects.filter(english_name__isnull=False).iterator():
            try:
                mapping = self.create_mapping_for_namaste_term(siddha_term, "siddha")
                if mapping:
                    created += 1
                processed += 1
            except Exception as e:
                print(
                    f"Error processing Siddha term {siddha_term.english_name}: {str(e)}"
                )

        return processed, created

    def process_all_unani_terms(self):
        """Process all Unani terms"""
        processed = 0
        created = 0

        for unani_term in Unani.objects.filter(english_name__isnull=False).iterator():
            try:
                mapping = self.create_mapping_for_namaste_term(unani_term, "unani")
                if mapping:
                    created += 1
                processed += 1
            except Exception as e:
                print(
                    f"Error processing Unani term {unani_term.english_name}: {str(e)}"
                )

        return processed, created
