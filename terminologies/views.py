from django.contrib.postgres.search import SearchQuery, SearchRank, TrigramSimilarity
from django.core.paginator import Paginator
from django.db.models import Case, Count, FloatField, Q, Value, When
from django.shortcuts import get_object_or_404
from rest_framework.decorators import api_view
from rest_framework.pagination import PageNumberPagination
from rest_framework.response import Response

from .models import Ayurvedha, ICD11Term, Siddha, TermMapping, Unani
from .serializers import (
    AyurvedhaListSerializer,
    AyurvedhaSerializer,
    ICD11TermListSerializer,
    RecentMappingSerializer,
    SiddhaListSerializer,
    TermMappingDetailSerializer,
    TermMappingSearchSerializer,
    TopICDMatchSerializer,
    UnaniListSerializer,
)
from .services.mapping_service import NamasteToICDMappingService


@api_view(["GET"])
def ayurvedha_fuzzy_search(request):
    search_term = request.query_params.get("q", "").strip()

    if not search_term:
        queryset = Ayurvedha.objects.all().order_by("code")
    else:
        fuzzy_qs = Ayurvedha.objects.annotate(
            similarity_code=TrigramSimilarity("code", search_term),
            similarity_english=TrigramSimilarity("english_name", search_term),
            similarity_hindi=TrigramSimilarity("hindi_name", search_term),
            similarity_diacritical=TrigramSimilarity("diacritical_name", search_term),
        ).filter(
            Q(similarity_code__gt=0.1)
            | Q(similarity_english__gt=0.1)
            | Q(similarity_hindi__gt=0.1)
            | Q(similarity_diacritical__gt=0.1)
        )

        exact_qs = Ayurvedha.objects.filter(
            Q(code__iexact=search_term)
            | Q(english_name__iexact=search_term)
            | Q(hindi_name__iexact=search_term)
            | Q(diacritical_name__iexact=search_term)
        )

        queryset = (fuzzy_qs | exact_qs).distinct()

        # queryset = queryset.annotate(
        #     max_similarity=(
        #         TrigramSimilarity("code", search_term)
        #         + TrigramSimilarity("english_name", search_term)
        #         + TrigramSimilarity("hindi_name", search_term)
        #         + TrigramSimilarity("diacritical_name", search_term)
        #     )
        # ).order_by("-max_similarity", "code")
        queryset = queryset.annotate(
            weighted_score=(
                TrigramSimilarity("english_name", search_term) * 2.5
                + TrigramSimilarity("code", search_term) * 1.0
                + TrigramSimilarity("hindi_name", search_term) * 0.8
                + TrigramSimilarity("diacritical_name", search_term) * 0.8
            )
        ).order_by("-weighted_score", "code")

    paginator = PageNumberPagination()
    paginator.page_size = 20
    page = paginator.paginate_queryset(queryset, request)

    serializer = AyurvedhaListSerializer(page, many=True)
    return paginator.get_paginated_response(serializer.data)


@api_view(["GET"])
def ayurvedha_autocomplete(request):
    search_term = request.query_params.get("q", "").strip()
    limit = min(int(request.query_params.get("limit", 8)), 12)

    if not search_term or len(search_term) < 2:
        return Response({"suggestions": []})

    queryset = get_autocomplete_queryset(search_term, limit)

    suggestions = [
        {
            "id": item.id,
            "code": item.code,
            "title": item.english_name,
            "subtitle": item.hindi_name if item.hindi_name else None,
            "score": round(float(getattr(item, "autocomplete_score", 0)), 1),
        }
        for item in queryset
    ]

    return Response(
        {"suggestions": suggestions, "query": search_term, "count": len(suggestions)}
    )


def get_autocomplete_queryset(search_term, limit):
    # Exact matches first (fastest)
    exact_qs = Ayurvedha.objects.filter(
        Q(code__iexact=search_term) | Q(english_name__iexact=search_term)
    ).only("id", "code", "english_name", "hindi_name")

    # Prefix matches (fast with indexes)
    prefix_qs = Ayurvedha.objects.filter(
        Q(code__istartswith=search_term) | Q(english_name__istartswith=search_term)
    ).only("id", "code", "english_name", "hindi_name")

    # Fuzzy matches (only for 3+ characters)
    fuzzy_qs = Ayurvedha.objects.none()
    if len(search_term) >= 3:
        fuzzy_qs = (
            Ayurvedha.objects.annotate(
                eng_sim=TrigramSimilarity("english_name", search_term),
                code_sim=TrigramSimilarity("code", search_term),
            )
            .filter(Q(eng_sim__gt=0.3) | Q(code_sim__gt=0.4))
            .only("id", "code", "english_name", "hindi_name")
        )

    # Combine and score
    combined_qs = (exact_qs | prefix_qs | fuzzy_qs).distinct()

    return combined_qs.annotate(
        autocomplete_score=Case(
            When(
                Q(code__iexact=search_term) | Q(english_name__iexact=search_term),
                then=Value(100.0),
            ),
            When(Q(code__istartswith=search_term), then=Value(90.0)),
            When(Q(english_name__istartswith=search_term), then=Value(85.0)),
            default=(
                TrigramSimilarity("english_name", search_term) * 60.0
                + TrigramSimilarity("code", search_term) * 40.0
            ),
            output_field=FloatField(),
        )
    ).order_by("-autocomplete_score", "english_name")[:limit]


@api_view(["GET"])
def siddha_fuzzy_search(request):
    search_term = request.query_params.get("q", "").strip()
    if not search_term:
        queryset = Siddha.objects.all().order_by("code")
    else:
        fuzzy_qs = Siddha.objects.annotate(
            similarity_code=TrigramSimilarity("code", search_term),
            similarity_english=TrigramSimilarity("english_name", search_term),
            similarity_tamil=TrigramSimilarity("tamil_name", search_term),
            similarity_romanized=TrigramSimilarity("romanized_name", search_term),
        ).filter(
            Q(similarity_code__gt=0.1)
            | Q(similarity_english__gt=0.1)
            | Q(similarity_tamil__gt=0.1)
            | Q(similarity_romanized__gt=0.1)
        )
        exact_qs = Siddha.objects.filter(
            Q(code__iexact=search_term)
            | Q(english_name__iexact=search_term)
            | Q(tamil_name__iexact=search_term)
            | Q(romanized_name__iexact=search_term)
        )
        queryset = (fuzzy_qs | exact_qs).distinct()
        queryset = queryset.annotate(
            weighted_score=(
                TrigramSimilarity("english_name", search_term) * 2.5
                + TrigramSimilarity("code", search_term) * 1.0
                + TrigramSimilarity("tamil_name", search_term) * 0.8
                + TrigramSimilarity("romanized_name", search_term) * 0.8
            )
        ).order_by("-weighted_score", "code")

    paginator = PageNumberPagination()
    paginator.page_size = 20
    page = paginator.paginate_queryset(queryset, request)
    serializer = SiddhaListSerializer(page, many=True)
    return paginator.get_paginated_response(serializer.data)


@api_view(["GET"])
def unani_fuzzy_search(request):
    search_term = request.query_params.get("q", "").strip()
    if not search_term:
        queryset = Unani.objects.all().order_by("code")
    else:
        fuzzy_qs = Unani.objects.annotate(
            similarity_code=TrigramSimilarity("code", search_term),
            similarity_english=TrigramSimilarity("english_name", search_term),
            similarity_arabic=TrigramSimilarity("arabic_name", search_term),
            similarity_romanized=TrigramSimilarity("romanized_name", search_term),
        ).filter(
            Q(similarity_code__gt=0.1)
            | Q(similarity_english__gt=0.1)
            | Q(similarity_arabic__gt=0.1)
            | Q(similarity_romanized__gt=0.1)
        )
        exact_qs = Unani.objects.filter(
            Q(code__iexact=search_term)
            | Q(english_name__iexact=search_term)
            | Q(arabic_name__iexact=search_term)
            | Q(romanized_name__iexact=search_term)
        )
        queryset = (fuzzy_qs | exact_qs).distinct()
        queryset = queryset.annotate(
            weighted_score=(
                TrigramSimilarity("english_name", search_term) * 2.5
                + TrigramSimilarity("code", search_term) * 1.0
                + TrigramSimilarity("arabic_name", search_term) * 0.8
                + TrigramSimilarity("romanized_name", search_term) * 0.8
            )
        ).order_by("-weighted_score", "code")

    paginator = PageNumberPagination()
    paginator.page_size = 20
    page = paginator.paginate_queryset(queryset, request)
    serializer = UnaniListSerializer(page, many=True)
    return paginator.get_paginated_response(serializer.data)


@api_view(["GET"])
def icd11_advanced_search(request):
    search_term = request.query_params.get("q", "").strip()
    use_fuzzy = request.query_params.get("fuzzy", "").lower() in ["true", "1"]

    # Note: Removed filters that don't exist in new model:
    # - chapter_filter (no chapter_no field)
    # - is_leaf (no is_leaf field)
    # - is_residual (no is_residual field)
    # - is_tm2 (no chapter_no field to filter on "26")

    if not search_term:
        queryset = ICD11Term.objects.all().order_by("code")
    else:
        if use_fuzzy:
            # Use full-text search with search_vector fields
            search_query = SearchQuery(search_term)

            # Search terms and their synonyms using search vectors
            queryset = (
                ICD11Term.objects.filter(
                    Q(search_vector=search_query)
                    | Q(synonyms__search_vector=search_query)
                )
                .distinct()
                .annotate(
                    # Calculate relevance score using search rank
                    search_rank=SearchRank("search_vector", search_query)
                    + SearchRank("synonyms__search_vector", search_query)
                )
                .order_by("-search_rank", "code")
            )

        else:
            # Use traditional icontains search with trigram similarity
            term_filter = Q(code__icontains=search_term) | Q(
                title__icontains=search_term
            )

            # Search synonym labels via reverse relation
            synonym_filter = Q(synonyms__label__icontains=search_term)

            queryset = (
                ICD11Term.objects.filter(term_filter | synonym_filter)
                .distinct()
                .annotate(
                    weighted_score=Case(
                        # Exact matches get highest priority
                        When(title__iexact=search_term, then=Value(10.0)),
                        When(code__iexact=search_term, then=Value(9.0)),
                        When(synonyms__label__iexact=search_term, then=Value(8.5)),
                        # Starts with gets medium priority
                        When(title__istartswith=search_term, then=Value(8.0)),
                        When(code__istartswith=search_term, then=Value(7.0)),
                        When(synonyms__label__istartswith=search_term, then=Value(6.5)),
                        # Contains gets lower priority
                        When(title__icontains=search_term, then=Value(6.0)),
                        When(code__icontains=search_term, then=Value(5.0)),
                        When(synonyms__label__icontains=search_term, then=Value(4.5)),
                        default=Value(1.0),
                        output_field=FloatField(),
                    )
                )
                .order_by("-weighted_score", "code")
            )

    # Pagination
    paginator = PageNumberPagination()
    paginator.page_size = 20
    page = paginator.paginate_queryset(queryset, request)
    serializer = ICD11TermListSerializer(page, many=True)
    return paginator.get_paginated_response(serializer.data)


@api_view(["GET"])
def search_namaste_mappings(request):
    """Fuzzy search NAMASTE terms within a specific system and return mappings"""

    # Parse parameters safely
    system = request.GET.get("system", "").lower().strip()
    query = request.GET.get("q", "").strip()

    try:
        min_confidence = float(
            request.GET.get("min_confidence", "0.2")
        )  # Lower default threshold
    except (ValueError, TypeError):
        min_confidence = 0.2

    try:
        page = int(request.GET.get("page", "1"))
    except (ValueError, TypeError):
        page = 1

    try:
        page_size = int(request.GET.get("page_size", "20"))
    except (ValueError, TypeError):
        page_size = 20

    # Validation
    if system not in ["ayurveda", "siddha", "unani"]:
        return Response(
            {"error": "Invalid system. Use: ayurveda, siddha, or unani"}, status=400
        )

    if not query:
        return Response({"error": 'Query parameter "q" is required'}, status=400)

    # Model mapping
    model_map = {"ayurveda": Ayurvedha, "siddha": Siddha, "unani": Unani}

    source_model = model_map[system]

    # FIXED: First find fuzzy matching terms
    fuzzy_terms = (
        source_model.objects.annotate(
            similarity=TrigramSimilarity("english_name", query)
        )
        .filter(similarity__gt=min_confidence, english_name__isnull=False)
        .order_by("-similarity")
    )

    # Debug: Check if we have fuzzy matches
    if not fuzzy_terms.exists():
        return Response(
            {
                "search_params": {
                    "system": system,
                    "query": query,
                    "min_confidence": min_confidence,
                },
                "debug_info": {
                    "total_terms_in_system": source_model.objects.count(),
                    "terms_with_english_name": source_model.objects.filter(
                        english_name__isnull=False
                    ).count(),
                    "fuzzy_matches_found": 0,
                    "suggestion": "Try lowering min_confidence to 0.1 or check if data exists",
                },
                "pagination": {
                    "page": page,
                    "page_size": page_size,
                    "total_pages": 0,
                    "total_results": 0,
                    "has_next": False,
                    "has_previous": False,
                },
                "results": [],
            }
        )

    # FIXED: Get the term IDs that matched fuzzy search
    fuzzy_term_ids = list(fuzzy_terms.values_list("id", flat=True))

    # FIXED: Find mappings for these specific fuzzy-matched terms
    filter_kwargs = {f"primary_{system}_term__id__in": fuzzy_term_ids}
    mappings = TermMapping.objects.select_related(
        "icd_term",
        "icd_term__class_kind",
        "primary_ayurveda_term",
        "primary_siddha_term",
        "primary_unani_term",
        "cross_ayurveda_term",
        "cross_siddha_term",
        "cross_unani_term",
    ).filter(**filter_kwargs)

    # FIXED: If no mappings found, create a response showing the fuzzy matches without mappings
    if not mappings.exists():
        # Show fuzzy matches even if they don't have ICD mappings yet
        fuzzy_results = []
        for term in fuzzy_terms[:page_size]:
            fuzzy_results.append(
                {
                    "term_id": term.id,
                    "code": term.code,
                    "english_name": term.english_name,
                    "similarity": round(term.similarity, 3),
                    "has_mapping": False,
                    "message": "Term found but no ICD mapping exists yet",
                }
            )

        return Response(
            {
                "search_params": {
                    "system": system,
                    "query": query,
                    "min_confidence": min_confidence,
                },
                "pagination": {
                    "page": page,
                    "page_size": page_size,
                    "total_pages": 1,
                    "total_results": len(fuzzy_results),
                    "has_next": False,
                    "has_previous": False,
                },
                "fuzzy_matches_without_mappings": fuzzy_results,
                "results": [],
            }
        )

    # Order mappings by the original fuzzy similarity
    mappings = mappings.order_by("-confidence_score")

    # Pagination
    paginator = Paginator(mappings, page_size)
    page_obj = paginator.get_page(page)

    # Build results with fuzzy similarity included
    results = []
    for mapping in page_obj:
        source_term = getattr(mapping, f"primary_{system}_term")

        # Find the fuzzy similarity for this term
        fuzzy_match = fuzzy_terms.filter(id=source_term.id).first()
        fuzzy_similarity = (
            getattr(fuzzy_match, "similarity", 0.0) if fuzzy_match else 0.0
        )

        mapping_data = {
            "mapping_id": mapping.id,
            "search_system": system,
            "fuzzy_similarity": round(fuzzy_similarity, 3),
            "source_term": {
                "code": source_term.code,
                "english_name": source_term.english_name,
                "description": source_term.description,
                **_get_system_specific_fields(source_term, system),
            },
            "namaste_terms": _build_comprehensive_namaste_info(mapping),
            "icd_mapping": {
                "code": mapping.icd_term.code,
                "title": mapping.icd_term.title,
                "foundation_uri": mapping.icd_term.foundation_uri,
                "chapter_no": mapping.icd_term.chapter_no,
                "similarity_score": round(mapping.icd_similarity, 3),
            },
            "confidence_score": round(mapping.confidence_score, 3),
            "created_at": mapping.created_at.isoformat(),
        }

        results.append(mapping_data)

    return Response(
        {
            "search_params": {
                "system": system,
                "query": query,
                "min_confidence": min_confidence,
            },
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total_pages": paginator.num_pages,
                "total_results": paginator.count,
                "has_next": page_obj.has_next(),
                "has_previous": page_obj.has_previous(),
            },
            "results": results,
        }
    )


def _build_comprehensive_namaste_info(mapping):
    """Build comprehensive NAMASTE cross-system information"""
    namaste_info = {"ayurveda": None, "siddha": None, "unani": None}

    # Primary terms (the original source)
    if mapping.primary_ayurveda_term:
        namaste_info["ayurveda"] = {
            "code": mapping.primary_ayurveda_term.code,
            "english_name": mapping.primary_ayurveda_term.english_name,
            "hindi_name": mapping.primary_ayurveda_term.hindi_name,
            "diacritical_name": mapping.primary_ayurveda_term.diacritical_name,
            "is_primary": True,
            "similarity_score": None,  # This is the source, so no similarity
        }

    if mapping.primary_siddha_term:
        namaste_info["siddha"] = {
            "code": mapping.primary_siddha_term.code,
            "english_name": mapping.primary_siddha_term.english_name,
            "tamil_name": mapping.primary_siddha_term.tamil_name,
            "romanized_name": mapping.primary_siddha_term.romanized_name,
            "is_primary": True,
            "similarity_score": None,
        }

    if mapping.primary_unani_term:
        namaste_info["unani"] = {
            "code": mapping.primary_unani_term.code,
            "english_name": mapping.primary_unani_term.english_name,
            "arabic_name": mapping.primary_unani_term.arabic_name,
            "romanized_name": mapping.primary_unani_term.romanized_name,
            "is_primary": True,
            "similarity_score": None,
        }

    # Cross-system matches (related terms found via fuzzy matching)
    if mapping.cross_ayurveda_term:
        namaste_info["ayurveda"] = {
            "code": mapping.cross_ayurveda_term.code,
            "english_name": mapping.cross_ayurveda_term.english_name,
            "hindi_name": mapping.cross_ayurveda_term.hindi_name,
            "diacritical_name": mapping.cross_ayurveda_term.diacritical_name,
            "is_primary": False,
            "similarity_score": round(mapping.cross_ayurveda_similarity, 3),
        }

    if mapping.cross_siddha_term:
        namaste_info["siddha"] = {
            "code": mapping.cross_siddha_term.code,
            "english_name": mapping.cross_siddha_term.english_name,
            "tamil_name": mapping.cross_siddha_term.tamil_name,
            "romanized_name": mapping.cross_siddha_term.romanized_name,
            "is_primary": False,
            "similarity_score": round(mapping.cross_siddha_similarity, 3),
        }

    if mapping.cross_unani_term:
        namaste_info["unani"] = {
            "code": mapping.cross_unani_term.code,
            "english_name": mapping.cross_unani_term.english_name,
            "arabic_name": mapping.cross_unani_term.arabic_name,
            "romanized_name": mapping.cross_unani_term.romanized_name,
            "is_primary": False,
            "similarity_score": round(mapping.cross_unani_similarity, 3),
        }

    return namaste_info


def _get_system_specific_fields(term, system):
    """Get system-specific fields for the term"""
    if system == "ayurveda":
        return {
            "hindi_name": term.hindi_name,
            "diacritical_name": term.diacritical_name,
        }
    elif system == "siddha":
        return {
            "tamil_name": term.tamil_name,
            "romanized_name": term.romanized_name,
            "reference": getattr(term, "reference", None),
        }
    elif system == "unani":
        return {
            "arabic_name": term.arabic_name,
            "romanized_name": term.romanized_name,
            "reference": getattr(term, "reference", None),
        }
    return {}


@api_view(["GET"])
def search_mappings(request):
    """Search mappings by term name with filters"""

    # Get parameters
    query = request.GET.get("q", "").strip()
    system = request.GET.get("system", "")
    min_confidence = float(request.GET.get("min_confidence", 0.0))
    page = int(request.GET.get("page", 1))
    page_size = int(request.GET.get("page_size", 20))

    if not query:
        return Response({"error": 'Query parameter "q" is required'}, status=400)

    # Base queryset
    mappings = TermMapping.objects.select_related(
        "icd_term",
        "primary_ayurveda_term",
        "primary_siddha_term",
        "primary_unani_term",
        "cross_ayurveda_term",
        "cross_siddha_term",
        "cross_unani_term",
    ).filter(confidence_score__gte=min_confidence)

    # Apply filters
    if system and system in ["ayurveda", "siddha", "unani"]:
        filter_kwargs = {f"primary_{system}_term__english_name__icontains": query}
        mappings = mappings.filter(**filter_kwargs)
    else:
        # Search across all systems
        mappings = mappings.filter(
            Q(primary_ayurveda_term__english_name__icontains=query)
            | Q(primary_siddha_term__english_name__icontains=query)
            | Q(primary_unani_term__english_name__icontains=query)
            | Q(icd_term__title__icontains=query)
        )

    # Pagination
    paginator = Paginator(mappings.order_by("-confidence_score"), page_size)
    page_obj = paginator.get_page(page)

    # Serialize results
    serializer = TermMappingSearchSerializer(page_obj, many=True)

    return Response(
        {
            "query": query,
            "filters": {"system": system, "min_confidence": min_confidence},
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total_pages": paginator.num_pages,
                "total_results": paginator.count,
                "has_next": page_obj.has_next(),
                "has_previous": page_obj.has_previous(),
            },
            "results": serializer.data,
        }
    )


@api_view(["GET"])
def list_mappings(request):
    """List all mappings with pagination and filtering"""

    # Parameters
    system = request.GET.get("system", "")
    confidence_level = request.GET.get("confidence", "")  # high, medium, low
    page = int(request.GET.get("page", 1))
    page_size = int(request.GET.get("page_size", 50))

    # Base queryset
    mappings = TermMapping.objects.select_related(
        "icd_term",
        "primary_ayurveda_term",
        "primary_siddha_term",
        "primary_unani_term",
        "cross_ayurveda_term",
        "cross_siddha_term",
        "cross_unani_term",
    )

    # Apply filters
    if system and system in ["ayurveda", "siddha", "unani"]:
        mappings = mappings.filter(source_system=system)

    if confidence_level:
        confidence_filters = {
            "high": Q(confidence_score__gte=0.7),
            "medium": Q(confidence_score__gte=0.5, confidence_score__lt=0.7),
            "low": Q(confidence_score__lt=0.5),
        }
        if confidence_level in confidence_filters:
            mappings = mappings.filter(confidence_filters[confidence_level])

    # Order by confidence score
    mappings = mappings.order_by("-confidence_score", "-created_at")

    # Pagination
    paginator = Paginator(mappings, page_size)
    page_obj = paginator.get_page(page)

    # Serialize results
    serializer = TermMappingSearchSerializer(page_obj, many=True)

    return Response(
        {
            "filters": {"system": system, "confidence_level": confidence_level},
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total_pages": paginator.num_pages,
                "total_results": paginator.count,
            },
            "results": serializer.data,
        }
    )


@api_view(["GET"])
def mapping_details(request, mapping_id):
    """Get detailed view of a specific mapping"""

    mapping = get_object_or_404(
        TermMapping.objects.select_related(
            "icd_term",
            "icd_term__class_kind",
            "primary_ayurveda_term",
            "primary_siddha_term",
            "primary_unani_term",
            "cross_ayurveda_term",
            "cross_siddha_term",
            "cross_unani_term",
        ),
        id=mapping_id,
    )

    serializer = TermMappingDetailSerializer(mapping)
    return Response(serializer.data)


@api_view(["GET"])
def mapping_statistics(request):
    """Get comprehensive mapping statistics"""

    service = NamasteToICDMappingService()
    stats = service.get_mapping_stats()

    # Get additional statistics
    top_matches = (
        TermMapping.objects.values("icd_term__code", "icd_term__title")
        .annotate(mapping_count=Count("id"))
        .order_by("-mapping_count")[:10]
    )

    recent = TermMapping.objects.select_related(
        "icd_term", "primary_ayurveda_term", "primary_siddha_term", "primary_unani_term"
    ).order_by("-created_at")[:5]

    # Serialize additional data
    top_serializer = TopICDMatchSerializer(top_matches, many=True)
    recent_serializer = RecentMappingSerializer(
        [
            {
                "source_system": mapping.source_system,
                "source_term": (
                    mapping.primary_ayurveda_term
                    or mapping.primary_siddha_term
                    or mapping.primary_unani_term
                ).english_name,
                "icd_title": mapping.icd_term.title,
                "confidence_score": mapping.confidence_score,
                "created_at": mapping.created_at,
            }
            for mapping in recent
        ],
        many=True,
    )

    stats.update(
        {
            "top_icd_matches": top_serializer.data,
            "recent_mappings": recent_serializer.data,
        }
    )

    return Response(stats)
