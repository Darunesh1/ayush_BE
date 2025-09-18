from django.contrib.postgres.search import SearchQuery, SearchRank, TrigramSimilarity
from django.db.models import F, FloatField, Q
from django.db.models.functions import Greatest

from .models import Ayurvedha, ICD11Term, Siddha, Unani


def fuzzy_search_icd_terms(query_text, limit=20):
    """Advanced fuzzy search for ICD11 terms"""
    search_query = SearchQuery(query_text, config="english")

    return (
        ICD11Term.objects.annotate(
            # Full-text search rank
            fts_rank=SearchRank("search_vector", search_query),
            # Trigram similarity scores using TrigramSimilarity
            title_similarity=TrigramSimilarity("title", query_text),
            code_similarity=TrigramSimilarity("code", query_text),
            # Combined relevance score
            relevance=Greatest(
                F("fts_rank"), F("title_similarity"), F("code_similarity")
            ),
        )
        .filter(
            Q(search_vector=search_query)  # Full-text search
            | Q(title__trigram_similar=query_text)  # Fuzzy title match
            | Q(code__trigram_similar=query_text)  # Fuzzy code match
            | Q(title__icontains=query_text)  # Partial match fallback
            | Q(code__iexact=query_text)  # Exact code match
        )
        .distinct()
        .order_by("-relevance", "title")[:limit]
    )


def fuzzy_search_ayurveda_terms(query_text, limit=20):
    """Fuzzy search for Ayurveda terms"""
    search_query = SearchQuery(query_text, config="english")

    return (
        Ayurvedha.objects.annotate(
            fts_rank=SearchRank("search_vector", search_query),
            english_similarity=TrigramSimilarity("english_name", query_text),
            hindi_similarity=TrigramSimilarity("hindi_name", query_text),
            code_similarity=TrigramSimilarity("code", query_text),
            relevance=Greatest(
                F("fts_rank"),
                F("english_similarity"),
                F("hindi_similarity"),
                F("code_similarity"),
            ),
        )
        .filter(
            Q(search_vector=search_query)
            | Q(english_name__trigram_similar=query_text)
            | Q(hindi_name__trigram_similar=query_text)
            | Q(diacritical_name__trigram_similar=query_text)
            | Q(code__trigram_similar=query_text)
            | Q(english_name__icontains=query_text)
            | Q(code__iexact=query_text)
        )
        .distinct()
        .order_by("-relevance", "english_name")[:limit]
    )


def fuzzy_search_siddha_terms(query_text, limit=20):
    """Fuzzy search for Siddha terms"""
    search_query = SearchQuery(query_text, config="english")

    return (
        Siddha.objects.annotate(
            fts_rank=SearchRank("search_vector", search_query),
            english_similarity=TrigramSimilarity("english_name", query_text),
            tamil_similarity=TrigramSimilarity("tamil_name", query_text),
            relevance=Greatest(
                F("fts_rank"), F("english_similarity"), F("tamil_similarity")
            ),
        )
        .filter(
            Q(search_vector=search_query)
            | Q(english_name__trigram_similar=query_text)
            | Q(tamil_name__trigram_similar=query_text)
            | Q(romanized_name__trigram_similar=query_text)
            | Q(code__trigram_similar=query_text)
            | Q(english_name__icontains=query_text)
        )
        .distinct()
        .order_by("-relevance", "english_name")[:limit]
    )


def fuzzy_search_unani_terms(query_text, limit=20):
    """Fuzzy search for Unani terms"""
    search_query = SearchQuery(query_text, config="english")

    return (
        Unani.objects.annotate(
            fts_rank=SearchRank("search_vector", search_query),
            english_similarity=TrigramSimilarity("english_name", query_text),
            arabic_similarity=TrigramSimilarity("arabic_name", query_text),
            relevance=Greatest(
                F("fts_rank"), F("english_similarity"), F("arabic_similarity")
            ),
        )
        .filter(
            Q(search_vector=search_query)
            | Q(english_name__trigram_similar=query_text)
            | Q(arabic_name__trigram_similar=query_text)
            | Q(romanized_name__trigram_similar=query_text)
            | Q(code__trigram_similar=query_text)
            | Q(english_name__icontains=query_text)
        )
        .distinct()
        .order_by("-relevance", "english_name")[:limit]
    )


def unified_fuzzy_search(query_text, systems=["all"], limit=20):
    """
    Unified search across all terminology systems
    systems: list of ['icd11', 'ayurveda', 'siddha', 'unani'] or ['all']
    """
    results = []

    if "all" in systems or "icd11" in systems:
        icd_results = fuzzy_search_icd_terms(query_text, limit)
        for term in icd_results:
            results.append(
                {
                    "system": "ICD-11",
                    "code": term.code,
                    "title": term.title,
                    "description": term.definition,
                    "relevance": getattr(term, "relevance", 0),
                    "object": term,
                }
            )

    if "all" in systems or "ayurveda" in systems:
        ayurveda_results = fuzzy_search_ayurveda_terms(query_text, limit)
        for term in ayurveda_results:
            results.append(
                {
                    "system": "Ayurveda",
                    "code": term.code,
                    "title": term.english_name,
                    "description": term.description,
                    "hindi_name": term.hindi_name,
                    "relevance": getattr(term, "relevance", 0),
                    "object": term,
                }
            )

    if "all" in systems or "siddha" in systems:
        siddha_results = fuzzy_search_siddha_terms(query_text, limit)
        for term in siddha_results:
            results.append(
                {
                    "system": "Siddha",
                    "code": term.code,
                    "title": term.english_name,
                    "description": term.description,
                    "tamil_name": term.tamil_name,
                    "relevance": getattr(term, "relevance", 0),
                    "object": term,
                }
            )

    if "all" in systems or "unani" in systems:
        unani_results = fuzzy_search_unani_terms(query_text, limit)
        for term in unani_results:
            results.append(
                {
                    "system": "Unani",
                    "code": term.code,
                    "title": term.english_name,
                    "description": term.description,
                    "arabic_name": term.arabic_name,
                    "relevance": getattr(term, "relevance", 0),
                    "object": term,
                }
            )

    # Sort by relevance across all systems
    return sorted(results, key=lambda x: x["relevance"], reverse=True)[:limit]
