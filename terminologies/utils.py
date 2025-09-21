from django.contrib.postgres.search import SearchQuery, SearchRank, TrigramSimilarity
from django.db import connection
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


# terminologies/utils.py
# Smart candidate finding using fuzzy search


def find_icd11_candidates(namaste_term, similarity_threshold=0.3, max_candidates=20):
    """Find ICD-11 candidates using fuzzy search before embedding"""

    search_text = namaste_term.get_embedding_text()
    search_terms = search_text.lower().split()

    # Build fuzzy search query
    candidates = set()

    for term in search_terms:
        if len(term) >= 3:  # Skip very short terms
            # Fuzzy search on title
            title_matches = ICD11Term.objects.extra(
                where=[f"similarity(title, %s) > {similarity_threshold}"], params=[term]
            ).order_by("-id")[: max_candidates // 2]

            # Fuzzy search on definition
            def_matches = ICD11Term.objects.extra(
                where=[f"similarity(definition, %s) > {similarity_threshold}"],
                params=[term],
            ).order_by("-id")[: max_candidates // 2]

            # Add to candidates
            candidates.update(title_matches)
            candidates.update(def_matches)

    # Also search index_terms JSON field
    for term in search_terms:
        if len(term) >= 3:
            json_matches = ICD11Term.objects.extra(
                where=[
                    "EXISTS (SELECT 1 FROM jsonb_array_elements_text(index_terms) AS elem WHERE similarity(elem, %s) > %s)"
                ],
                params=[term, similarity_threshold],
            )[: max_candidates // 3]

            candidates.update(json_matches)

    return list(candidates)[:max_candidates]


def get_unique_icd11_candidates_for_all_namaste():
    """Get unique set of ICD-11 terms that match any NAMASTE term WITH PROGRESS"""

    from terminologies.models import Ayurvedha, Siddha, Unani

    all_candidates = set()
    processed_count = 0

    print("🔍 Finding ICD-11 candidates using fuzzy search...")

    # Process all NAMASTE systems
    for Model in [Ayurvedha, Siddha, Unani]:
        model_name = Model.__name__
        namaste_terms = list(
            Model.objects.exclude(embedding__isnull=True)
        )  # Convert to list for progress
        model_total = len(namaste_terms)

        print(f"\n📋 Processing {model_name}: {model_total} terms")
        print("-" * 50)

        for i, term in enumerate(namaste_terms, 1):
            # Show current term being processed
            term_name = (
                term.english_name[:40] if term.english_name else f"Term {term.pk}"
            )
            print(f"⚙️ {i:4}/{model_total:4}: {term_name}")

            # Find candidates using your existing function
            candidates = find_icd11_candidates(term, similarity_threshold=0.25)
            new_candidates = set(candidates) - all_candidates
            all_candidates.update(candidates)

            # Show new matches found
            if new_candidates:
                print(f"   ✅ Found {len(new_candidates)} new matches")
                # Show a few examples of new matches
                for candidate in list(new_candidates)[:2]:  # Show first 2 new matches
                    print(f"      → {candidate.title[:50]}")

            processed_count += 1

            # Show progress every 25 terms OR on first few terms
            if i <= 5 or i % 25 == 0 or i == model_total:
                model_progress = (i / model_total) * 100
                total_candidates = len(all_candidates)
                print(
                    f"   📈 {model_name}: {i}/{model_total} ({model_progress:5.1f}%) | Total candidates: {total_candidates}"
                )
                print()  # Empty line for readability

    print("=" * 60)
    print(f"✅ CANDIDATE DISCOVERY COMPLETE!")
    print(f"   Total NAMASTE terms processed: {processed_count}")
    print(f"   Total unique ICD-11 candidates: {len(all_candidates)}")
    print("=" * 60)

    return list(all_candidates)
