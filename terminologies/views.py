from django.contrib.postgres.search import SearchQuery, SearchRank, TrigramSimilarity
from django.core.paginator import Paginator
from django.db import models
from django.db.models import Case, Count, F, FloatField, Q, Value, When
from django.db.models.functions import Cast
from django.shortcuts import get_object_or_404
from drf_spectacular.types import OpenApiTypes

# ADD THESE IMPORTS FOR drf-spectacular
from drf_spectacular.utils import (
    OpenApiExample,
    OpenApiParameter,
    OpenApiResponse,
    extend_schema,
)
from rest_framework.decorators import api_view
from rest_framework.pagination import PageNumberPagination
from rest_framework.response import Response

from .models import Ayurvedha, ICD11Term, Siddha, TermMapping, Unani
from .serializers import (
    AyurvedhaListSerializer,
    AyurvedhaSerializer,
    ICD11TermListSerializer,
    ICD11TermSearchSerializer,
    RecentMappingSerializer,
    SiddhaListSerializer,
    TermMappingDetailSerializer,
    TermMappingSearchSerializer,
    TopICDMatchSerializer,
    UnaniListSerializer,
)
from .services.mapping_service import NamasteToICDMappingService


@extend_schema(
    summary="Ayurveda Fuzzy Search",
    description="Perform fuzzy search on Ayurveda terms using PostgreSQL pg_trgm extension",
    parameters=[
        OpenApiParameter(
            name="q",
            type=OpenApiTypes.STR,
            location=OpenApiParameter.QUERY,
            description="Search term for fuzzy matching",
            required=False,
        ),
    ],
    responses={200: AyurvedhaListSerializer(many=True)},
    tags=["Ayurveda"],
)
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


@extend_schema(
    summary="Ayurveda Autocomplete",
    description="Get autocomplete suggestions for Ayurveda terms",
    parameters=[
        OpenApiParameter(
            name="q",
            type=OpenApiTypes.STR,
            location=OpenApiParameter.QUERY,
            description="Search term for autocomplete",
            required=True,
        ),
        OpenApiParameter(
            name="limit",
            type=OpenApiTypes.INT,
            location=OpenApiParameter.QUERY,
            description="Maximum number of suggestions (default: 8, max: 12)",
            required=False,
        ),
    ],
    responses={200: OpenApiTypes.OBJECT},  # Custom response format
    tags=["Ayurveda"],
)
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


@extend_schema(
    summary="Siddha Fuzzy Search",
    description="Perform fuzzy search on Siddha terms using PostgreSQL pg_trgm extension with multilingual support",
    parameters=[
        OpenApiParameter(
            name="q",
            type=OpenApiTypes.STR,
            location=OpenApiParameter.QUERY,
            description="Search term for fuzzy matching across English, Tamil, and romanized names",
            required=False,
            examples=[
                OpenApiExample("English term", value="fever"),
                OpenApiExample("Tamil term", value="காய்ச்சல்"),
                OpenApiExample("Code search", value="S001"),
                OpenApiExample("Romanized term", value="kaaychchal"),
            ],
        ),
        OpenApiParameter(
            name="threshold",
            type=OpenApiTypes.FLOAT,
            location=OpenApiParameter.QUERY,
            description="Similarity threshold for fuzzy matching, range 0.0-1.0 (default: 0.1)",
            required=False,
            examples=[
                OpenApiExample("Strict matching", value=0.3),
                OpenApiExample("Loose matching", value=0.05),
            ],
        ),
        OpenApiParameter(
            name="page",
            type=OpenApiTypes.INT,
            location=OpenApiParameter.QUERY,
            description="Page number for pagination (default: 1)",
            required=False,
        ),
    ],
    responses={
        200: OpenApiResponse(
            description="Paginated fuzzy search results with weighted scoring",
            response={
                "type": "object",
                "properties": {
                    "count": {
                        "type": "integer",
                        "description": "Total number of matching Siddha terms",
                    },
                    "next": {
                        "type": "string",
                        "nullable": True,
                        "description": "URL for next page",
                    },
                    "previous": {
                        "type": "string",
                        "nullable": True,
                        "description": "URL for previous page",
                    },
                    "results": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "integer"},
                                "code": {"type": "string"},
                                "english_name": {"type": "string", "nullable": True},
                                "tamil_name": {"type": "string", "nullable": True},
                                "romanized_name": {"type": "string", "nullable": True},
                            },
                        },
                    },
                },
            },
        ),
        400: OpenApiResponse(
            description="Bad request - invalid parameters",
            examples=[
                OpenApiExample(
                    "Invalid threshold",
                    value={"error": "Threshold must be between 0.0 and 1.0"},
                )
            ],
        ),
        500: OpenApiResponse(
            description="Internal server error",
        ),
    },
    tags=["Siddha"],
    operation_id="siddha_fuzzy_search",
)
@api_view(["GET"])
def siddha_fuzzy_search(request):
    """
    Perform fuzzy search on Siddha medicine terms with multilingual support.

    Features:
    - Searches across code, English name, Tamil name, and romanized name
    - Uses PostgreSQL trigram similarity for fuzzy matching
    - Weighted scoring with English names prioritized
    - Combines fuzzy and exact matching for comprehensive results
    - Configurable similarity threshold
    """
    search_term = request.query_params.get("q", "").strip()
    similarity_threshold = float(request.query_params.get("threshold", "0.1"))

    # Validate threshold parameter
    if not (0.0 <= similarity_threshold <= 1.0):
        return Response({"error": "Threshold must be between 0.0 and 1.0"}, status=400)

    if not search_term:
        queryset = Siddha.objects.all().order_by("code")
    else:
        # Fuzzy search using trigram similarity
        fuzzy_qs = Siddha.objects.annotate(
            similarity_code=TrigramSimilarity("code", search_term),
            similarity_english=TrigramSimilarity("english_name", search_term),
            similarity_tamil=TrigramSimilarity("tamil_name", search_term),
            similarity_romanized=TrigramSimilarity("romanized_name", search_term),
        ).filter(
            Q(similarity_code__gt=similarity_threshold)
            | Q(similarity_english__gt=similarity_threshold)
            | Q(similarity_tamil__gt=similarity_threshold)
            | Q(similarity_romanized__gt=similarity_threshold)
        )

        # Exact matches (case-insensitive)
        exact_qs = Siddha.objects.filter(
            Q(code__iexact=search_term)
            | Q(english_name__iexact=search_term)
            | Q(tamil_name__iexact=search_term)
            | Q(romanized_name__iexact=search_term)
        )

        # Combine and deduplicate results
        queryset = (fuzzy_qs | exact_qs).distinct()

        # Apply weighted scoring for relevance ranking
        queryset = queryset.annotate(
            weighted_score=(
                TrigramSimilarity("english_name", search_term)
                * 2.5  # Prioritize English
                + TrigramSimilarity("code", search_term)
                * 1.0  # Standard weight for codes
                + TrigramSimilarity("tamil_name", search_term)
                * 0.8  # Tamil language support
                + TrigramSimilarity("romanized_name", search_term)
                * 0.8  # Romanized Tamil
            )
        ).order_by("-weighted_score", "code")

    # Pagination
    paginator = PageNumberPagination()
    paginator.page_size = 20
    page = paginator.paginate_queryset(queryset, request)

    # Serialize results
    serializer = SiddhaListSerializer(page, many=True)
    return paginator.get_paginated_response(serializer.data)


@extend_schema(
    summary="Unani Fuzzy Search",
    description="Perform fuzzy search on Unani terms using PostgreSQL pg_trgm extension with multilingual support",
    parameters=[
        OpenApiParameter(
            name="q",
            type=OpenApiTypes.STR,
            location=OpenApiParameter.QUERY,
            description="Search term for fuzzy matching across English, Arabic, and romanized names",
            required=False,
            examples=[
                OpenApiExample("English term", value="fever"),
                OpenApiExample("Arabic term", value="حمى"),
                OpenApiExample("Code search", value="U001"),
            ],
        ),
        OpenApiParameter(
            name="threshold",
            type=OpenApiTypes.FLOAT,
            location=OpenApiParameter.QUERY,
            description="Similarity threshold for fuzzy matching, range 0.0-1.0 (default: 0.1)",
            required=False,
            examples=[
                OpenApiExample("Strict matching", value=0.3),
                OpenApiExample("Loose matching", value=0.05),
            ],
        ),
        OpenApiParameter(
            name="page",
            type=OpenApiTypes.INT,
            location=OpenApiParameter.QUERY,
            description="Page number for pagination (default: 1)",
            required=False,
        ),
    ],
    responses={
        200: OpenApiResponse(
            description="Paginated fuzzy search results with weighted scoring",
            response={
                "type": "object",
                "properties": {
                    "count": {
                        "type": "integer",
                        "description": "Total number of matching Unani terms",
                    },
                    "next": {
                        "type": "string",
                        "nullable": True,
                        "description": "URL for next page",
                    },
                    "previous": {
                        "type": "string",
                        "nullable": True,
                        "description": "URL for previous page",
                    },
                    "results": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "integer"},
                                "code": {"type": "string"},
                                "english_name": {"type": "string", "nullable": True},
                                "arabic_name": {"type": "string", "nullable": True},
                                "romanized_name": {"type": "string", "nullable": True},
                            },
                        },
                    },
                },
            },
        ),
        400: OpenApiResponse(
            description="Bad request - invalid parameters",
            examples=[
                OpenApiExample(
                    "Invalid threshold",
                    value={"error": "Threshold must be between 0.0 and 1.0"},
                )
            ],
        ),
        500: OpenApiResponse(
            description="Internal server error",
        ),
    },
    tags=["Unani"],
    operation_id="unani_fuzzy_search",
)
@api_view(["GET"])
def unani_fuzzy_search(request):
    """
    Perform fuzzy search on Unani medicine terms with multilingual support.

    Features:
    - Searches across code, English name, Arabic name, and romanized name
    - Uses PostgreSQL trigram similarity for fuzzy matching
    - Weighted scoring with English names prioritized
    - Combines fuzzy and exact matching for comprehensive results
    - Configurable similarity threshold
    """
    search_term = request.query_params.get("q", "").strip()
    similarity_threshold = float(request.query_params.get("threshold", "0.1"))

    # Validate threshold parameter
    if not (0.0 <= similarity_threshold <= 1.0):
        return Response({"error": "Threshold must be between 0.0 and 1.0"}, status=400)

    if not search_term:
        queryset = Unani.objects.all().order_by("code")
    else:
        # Fuzzy search using trigram similarity
        fuzzy_qs = Unani.objects.annotate(
            similarity_code=TrigramSimilarity("code", search_term),
            similarity_english=TrigramSimilarity("english_name", search_term),
            similarity_arabic=TrigramSimilarity("arabic_name", search_term),
            similarity_romanized=TrigramSimilarity("romanized_name", search_term),
        ).filter(
            Q(similarity_code__gt=similarity_threshold)
            | Q(similarity_english__gt=similarity_threshold)
            | Q(similarity_arabic__gt=similarity_threshold)
            | Q(similarity_romanized__gt=similarity_threshold)
        )

        # Exact matches (case-insensitive)
        exact_qs = Unani.objects.filter(
            Q(code__iexact=search_term)
            | Q(english_name__iexact=search_term)
            | Q(arabic_name__iexact=search_term)
            | Q(romanized_name__iexact=search_term)
        )

        # Combine and deduplicate results
        queryset = (fuzzy_qs | exact_qs).distinct()

        # Apply weighted scoring for relevance ranking
        queryset = queryset.annotate(
            weighted_score=(
                TrigramSimilarity("english_name", search_term)
                * 2.5  # Prioritize English
                + TrigramSimilarity("code", search_term)
                * 1.0  # Standard weight for codes
                + TrigramSimilarity("arabic_name", search_term)
                * 0.8  # Slightly lower for Arabic
                + TrigramSimilarity("romanized_name", search_term)
                * 0.8  # Equal to Arabic
            )
        ).order_by("-weighted_score", "code")

    # Pagination
    paginator = PageNumberPagination()
    paginator.page_size = 20
    page = paginator.paginate_queryset(queryset, request)

    # Serialize results
    serializer = UnaniListSerializer(page, many=True)
    return paginator.get_paginated_response(serializer.data)


@extend_schema(
    summary="ICD-11 Advanced Search",
    description="Advanced search through ICD-11 terms including Traditional Medicine Module 2 with fuzzy matching, full-text search, and JSON field support",
    parameters=[
        OpenApiParameter(
            name="q",
            type=OpenApiTypes.STR,
            location=OpenApiParameter.QUERY,
            description="Search term to query across codes, titles, definitions, and JSON fields",
            required=False,
            examples=[
                OpenApiExample(
                    "Basic search", value="diabetes", description="Simple text search"
                ),
                OpenApiExample(
                    "Code search", value="E10", description="Search by ICD code"
                ),
                OpenApiExample(
                    "Complex term",
                    value="blood cancer",
                    description="Multi-word medical term",
                ),
            ],
        ),
        OpenApiParameter(
            name="fuzzy",
            type=OpenApiTypes.BOOL,
            location=OpenApiParameter.QUERY,
            description="Enable fuzzy matching using trigram similarity (default: false)",
            required=False,
            examples=[
                OpenApiExample(
                    "Enable fuzzy",
                    value=True,
                    description="Use fuzzy matching for typos and similar terms",
                )
            ],
        ),
        OpenApiParameter(
            name="threshold",
            type=OpenApiTypes.FLOAT,
            location=OpenApiParameter.QUERY,
            description="Similarity threshold for fuzzy search, range 0.0-1.0 (default: 0.2)",
            required=False,
            examples=[
                OpenApiExample(
                    "Strict matching",
                    value=0.4,
                    description="Higher threshold for more precise matches",
                ),
                OpenApiExample(
                    "Loose matching",
                    value=0.1,
                    description="Lower threshold for broader matches",
                ),
            ],
        ),
        OpenApiParameter(
            name="use_fts",
            type=OpenApiTypes.BOOL,
            location=OpenApiParameter.QUERY,
            description="Enable full-text search using search_vector field (default: false)",
            required=False,
            examples=[
                OpenApiExample(
                    "Full-text search",
                    value=True,
                    description="Use PostgreSQL full-text search capabilities",
                )
            ],
        ),
        OpenApiParameter(
            name="page",
            type=OpenApiTypes.INT,
            location=OpenApiParameter.QUERY,
            description="Page number for pagination (default: 1)",
            required=False,
        ),
    ],
    responses={
        200: OpenApiResponse(
            description="Paginated search results",
            response={
                "type": "object",
                "properties": {
                    "count": {
                        "type": "integer",
                        "description": "Total number of matching terms",
                    },
                    "next": {
                        "type": "string",
                        "nullable": True,
                        "description": "URL for next page",
                    },
                    "previous": {
                        "type": "string",
                        "nullable": True,
                        "description": "URL for previous page",
                    },
                    "results": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "foundation_uri": {"type": "string"},
                                "code": {"type": "string", "nullable": True},
                                "title": {"type": "string"},
                                "browser_url": {"type": "string"},
                                "class_kind": {"type": "string"},
                            },
                        },
                    },
                },
            },
        ),
        400: OpenApiResponse(
            description="Bad request - invalid parameters",
            examples=[
                OpenApiExample(
                    "Invalid threshold",
                    value={"error": "Threshold must be between 0.0 and 1.0"},
                )
            ],
        ),
        500: OpenApiResponse(
            description="Internal server error",
        ),
    },
    tags=["ICD-11"],
    operation_id="icd11_advanced_search",
)
@api_view(["GET"])
def icd11_advanced_search(request):
    """
    Advanced search for ICD-11 terms with fuzzy search and full-text capabilities.

    Query Parameters:
    - q: Search term
    - fuzzy: Use fuzzy search (true/false)
    - threshold: Similarity threshold for fuzzy search (default: 0.2)
    - use_fts: Use full-text search with search_vector (true/false)
    """
    search_term = request.query_params.get("q", "").strip()
    use_fuzzy = request.query_params.get("fuzzy", "").lower() in ["true", "1"]
    use_fts = request.query_params.get("use_fts", "").lower() in ["true", "1"]
    similarity_threshold = float(request.query_params.get("threshold", "0.2"))

    if not search_term:
        queryset = ICD11Term.objects.all().order_by("code")
    else:
        if use_fts and hasattr(ICD11Term, "search_vector"):
            # Use full-text search with search_vector
            search_query = SearchQuery(search_term)
            queryset = (
                ICD11Term.objects.filter(search_vector=search_query)
                .annotate(rank=SearchRank(F("search_vector"), search_query))
                .order_by("-rank", "code")
            )
        elif use_fuzzy:
            # Enhanced fuzzy search with JSON field support
            queryset = (
                ICD11Term.objects.annotate(
                    code_sim=TrigramSimilarity("code", search_term),
                    title_sim=TrigramSimilarity("title", search_term),
                    definition_sim=TrigramSimilarity("definition", search_term),
                    # JSON field similarity - convert JSON array to text for similarity
                    index_terms_sim=Case(
                        When(
                            index_terms__isnull=False,
                            then=TrigramSimilarity(
                                Cast("index_terms", models.TextField()),
                                search_term,
                            ),
                        ),
                        default=Value(0.0),
                        output_field=FloatField(),
                    ),
                    inclusions_sim=Case(
                        When(
                            inclusions__isnull=False,
                            then=TrigramSimilarity(
                                Cast("inclusions", models.TextField()),
                                search_term,
                            ),
                        ),
                        default=Value(0.0),
                        output_field=FloatField(),
                    ),
                )
                .filter(
                    Q(code_sim__gte=similarity_threshold)
                    | Q(title_sim__gte=similarity_threshold)
                    | Q(definition_sim__gte=similarity_threshold)
                    | Q(index_terms_sim__gte=similarity_threshold)
                    | Q(inclusions_sim__gte=similarity_threshold)
                )
                .distinct()
                .annotate(
                    total_similarity=(
                        F("code_sim")
                        + F("title_sim")
                        + F("definition_sim")
                        + F("index_terms_sim")
                        + F("inclusions_sim")
                    )
                )
                .order_by("-total_similarity", "code")
            )
        else:
            # Traditional icontains search with JSON field support
            basic_filter = (
                Q(code__icontains=search_term)
                | Q(title__icontains=search_term)
                | Q(definition__icontains=search_term)
            )

            # JSON field searches using containment
            json_filter = (
                Q(index_terms__icontains=search_term)
                | Q(inclusions__icontains=search_term)
                | Q(exclusions__icontains=search_term)
            )

            queryset = (
                ICD11Term.objects.filter(basic_filter | json_filter)
                .distinct()
                .annotate(
                    weighted_score=Case(
                        # Exact matches get highest priority
                        When(title__iexact=search_term, then=Value(10.0)),
                        When(code__iexact=search_term, then=Value(9.0)),
                        When(definition__iexact=search_term, then=Value(8.8)),
                        # JSON field exact matches
                        When(
                            index_terms__icontains=f'"{search_term}"', then=Value(8.5)
                        ),
                        When(inclusions__icontains=f'"{search_term}"', then=Value(8.2)),
                        # Starts with gets medium priority
                        When(title__istartswith=search_term, then=Value(8.0)),
                        When(code__istartswith=search_term, then=Value(7.0)),
                        When(definition__istartswith=search_term, then=Value(6.8)),
                        # Contains gets lower priority
                        When(title__icontains=search_term, then=Value(6.0)),
                        When(code__icontains=search_term, then=Value(5.0)),
                        When(definition__icontains=search_term, then=Value(4.8)),
                        When(index_terms__icontains=search_term, then=Value(4.5)),
                        When(inclusions__icontains=search_term, then=Value(4.2)),
                        When(exclusions__icontains=search_term, then=Value(4.0)),
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

    # Use List serializer for better performance
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
