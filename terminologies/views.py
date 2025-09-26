import csv
import logging
from io import TextIOWrapper

from django.conf import settings
from django.contrib.postgres.search import SearchQuery, SearchRank, TrigramSimilarity
from django.core.paginator import Paginator
from django.db import models, transaction
from django.db.models import Case, Count, F, FloatField, Q, Value, When
from django.db.models.functions import Cast
from django.shortcuts import get_object_or_404
from django.utils import timezone
from django.views.decorators.cache import cache_page
from drf_spectacular.openapi import AutoSchema
from drf_spectacular.types import OpenApiTypes

# ADD THESE IMPORTS FOR drf-spectacular
from drf_spectacular.utils import (
    OpenApiExample,
    OpenApiParameter,
    OpenApiResponse,
    extend_schema,
)
from rest_framework import status
from rest_framework.decorators import api_view, parser_classes, renderer_classes
from rest_framework.pagination import PageNumberPagination
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.renderers import JSONRenderer
from rest_framework.response import Response

from .models import Ayurvedha, ICD11Term, Siddha, TermMapping, Unani
from .serializers import (
    AyurvedhaListSerializer,
    AyurvedhaSerializer,
    CombinedSearchResponseSerializer,
    ErrorResponseSerializer,
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

logger = logging.getLogger(__name__)


@extend_schema(
    summary="Ayurveda Fuzzy Search",
    description="Perform fuzzy search on Ayurveda terms using PostgreSQL pg_trgm extension with multilingual support",
    parameters=[
        OpenApiParameter(
            name="q",
            type=OpenApiTypes.STR,
            location=OpenApiParameter.QUERY,
            description="Search term for fuzzy matching across English, Hindi, and diacritical names",
            required=False,
            examples=[
                OpenApiExample("English term", value="fever"),
                OpenApiExample("Hindi term", value="बुखार"),
                OpenApiExample("Code search", value="A001"),
                OpenApiExample("Diacritical term", value="jvara"),
                OpenApiExample("Sanskrit term", value="pittajvara"),
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
                        "description": "Total number of matching Ayurveda terms",
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
                                "hindi_name": {"type": "string", "nullable": True},
                                "diacritical_name": {
                                    "type": "string",
                                    "nullable": True,
                                },
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
    tags=["Ayurveda"],
    operation_id="ayurveda_fuzzy_search",
)
@api_view(["GET"])
def ayurvedha_fuzzy_search(request):
    """
    Perform fuzzy search on Ayurveda medicine terms with multilingual support.

    Features:
    - Searches across code, English name, Hindi name, and diacritical name
    - Uses PostgreSQL trigram similarity for fuzzy matching
    - Weighted scoring with English names prioritized
    - Combines fuzzy and exact matching for comprehensive results
    - Configurable similarity threshold
    - Support for Sanskrit/Hindi terminology and transliteration
    """
    search_term = request.query_params.get("q", "").strip()
    similarity_threshold = float(request.query_params.get("threshold", "0.1"))

    # Validate threshold parameter
    if not (0.0 <= similarity_threshold <= 1.0):
        return Response({"error": "Threshold must be between 0.0 and 1.0"}, status=400)

    if not search_term:
        queryset = Ayurvedha.objects.all().order_by("code")
    else:
        # Fuzzy search using trigram similarity
        fuzzy_qs = Ayurvedha.objects.annotate(
            similarity_code=TrigramSimilarity("code", search_term),
            similarity_english=TrigramSimilarity("english_name", search_term),
            similarity_hindi=TrigramSimilarity("hindi_name", search_term),
            similarity_diacritical=TrigramSimilarity("diacritical_name", search_term),
        ).filter(
            Q(similarity_code__gt=similarity_threshold)
            | Q(similarity_english__gt=similarity_threshold)
            | Q(similarity_hindi__gt=similarity_threshold)
            | Q(similarity_diacritical__gt=similarity_threshold)
        )

        # Exact matches (case-insensitive)
        exact_qs = Ayurvedha.objects.filter(
            Q(code__iexact=search_term)
            | Q(english_name__iexact=search_term)
            | Q(hindi_name__iexact=search_term)
            | Q(diacritical_name__iexact=search_term)
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
                + TrigramSimilarity("hindi_name", search_term)
                * 0.8  # Hindi language support
                + TrigramSimilarity("diacritical_name", search_term)
                * 0.8  # Diacritical/Sanskrit terms
            )
        ).order_by("-weighted_score", "code")

    # Pagination
    paginator = PageNumberPagination()
    paginator.page_size = 20
    page = paginator.paginate_queryset(queryset, request)

    # Serialize results
    serializer = AyurvedhaListSerializer(page, many=True)
    return paginator.get_paginated_response(serializer.data)


@extend_schema(
    summary="Ayurveda Autocomplete",
    description="Fast autocomplete for Ayurveda terms - returns only English name titles for optimal performance",
    parameters=[
        OpenApiParameter(
            name="q",
            type=OpenApiTypes.STR,
            location=OpenApiParameter.QUERY,
            description="Search term for autocomplete (searches across English, Hindi, and diacritical names)",
            required=True,
            examples=[
                OpenApiExample(
                    "Partial English", value="fev", description="Matches 'fever'"
                ),
                OpenApiExample(
                    "Hindi term", value="बुख", description="Searches Hindi names"
                ),
                OpenApiExample(
                    "Diacritical",
                    value="jvar",
                    description="Searches transliterated terms",
                ),
            ],
        ),
        OpenApiParameter(
            name="limit",
            type=OpenApiTypes.INT,
            location=OpenApiParameter.QUERY,
            description="Maximum results to return (default: 10, max: 20)",
            required=False,
        ),
    ],
    responses={
        200: OpenApiResponse(
            description="List of matching English name titles only",
            response={
                "type": "array",
                "items": {"type": "string"},
                "example": ["fever", "fever with headache", "febrile condition"],
            },
        )
    },
    tags=["Ayurveda"],
    operation_id="ayurveda_autocomplete",
)
@api_view(["GET"])
def ayurveda_autocomplete(request):
    search_term = request.query_params.get("q", "").strip()
    limit = min(int(request.query_params.get("limit", 10)), 20)

    if (
        not search_term or len(search_term) < 1
    ):  # Reduced from 2 to 1 for better autocomplete
        return Response([])

    # Use the same pattern as your working fuzzy search
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

    # Exact matches for better autocomplete experience
    exact_qs = Ayurvedha.objects.filter(
        Q(code__icontains=search_term)
        | Q(english_name__icontains=search_term)
        | Q(hindi_name__icontains=search_term)
        | Q(diacritical_name__icontains=search_term)
    )

    # Combine queries and apply weighted scoring
    queryset = (
        (fuzzy_qs | exact_qs)
        .distinct()
        .annotate(
            weighted_score=(
                TrigramSimilarity("english_name", search_term) * 2.5
                + TrigramSimilarity("code", search_term) * 1.0
                + TrigramSimilarity("hindi_name", search_term) * 0.8
                + TrigramSimilarity("diacritical_name", search_term) * 0.8
            )
        )
        .order_by("-weighted_score", "english_name")
    )

    # Extract only english_name titles
    titles = list(queryset.values_list("english_name", flat=True)[:limit])

    return Response(titles)


@extend_schema(
    summary="Upload Ayurveda CSV",
    description="Upload CSV file to populate Ayurveda terms database. Updates existing records by code or creates new ones.",
    request={
        "multipart/form-data": {
            "type": "object",
            "properties": {
                "file": {
                    "type": "string",
                    "format": "binary",
                    "description": "CSV file with Ayurveda terms (columns: code, english_name, description, hindi_name, diacritical_name)",
                },
                "update_search_vector": {
                    "type": "boolean",
                    "description": "Whether to update search vectors after import",
                    "default": True,
                },
            },
            "required": ["file"],
        }
    },
    responses={
        200: OpenApiResponse(
            description="CSV processed successfully",
            response={
                "type": "object",
                "properties": {
                    "created": {
                        "type": "integer",
                        "description": "Number of new records created",
                    },
                    "updated": {
                        "type": "integer",
                        "description": "Number of existing records updated",
                    },
                    "skipped": {
                        "type": "integer",
                        "description": "Number of rows skipped due to errors",
                    },
                    "total_processed": {
                        "type": "integer",
                        "description": "Total rows processed",
                    },
                    "errors": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of error messages for failed rows",
                    },
                    "summary": {"type": "string", "description": "Processing summary"},
                },
            },
        ),
        400: OpenApiResponse(
            description="Bad request - invalid file or format",
            examples=[
                OpenApiExample(
                    "No file provided", value={"error": "CSV file not provided"}
                ),
                OpenApiExample(
                    "Invalid CSV format",
                    value={"error": "Invalid CSV format: missing required headers"},
                ),
            ],
        ),
        413: OpenApiResponse(description="File too large"),
    },
    tags=["Ayurveda"],
    operation_id="ayurveda_csv_upload",
)
@api_view(["POST"])
@parser_classes([MultiPartParser, FormParser])
def ayurveda_csv_upload(request):
    """
    Upload CSV file to populate Ayurveda terms database.

    CSV Format:
    - Required columns: code
    - Optional columns: english_name, description, hindi_name, diacritical_name
    - First row should contain headers
    - Encoding: UTF-8

    Processing Logic:
    - If code exists: Update the existing record
    - If code doesn't exist: Create new record
    - Ensures no duplicate codes in database
    """

    # Validate file upload
    file = request.FILES.get("file")
    if not file:
        return Response(
            {"error": "CSV file not provided"}, status=status.HTTP_400_BAD_REQUEST
        )

    # Check file size (limit to 10MB)
    if file.size > 10 * 1024 * 1024:
        return Response(
            {"error": "File size too large. Maximum 10MB allowed"},
            status=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
        )

    # Check file extension
    if not file.name.endswith(".csv"):
        return Response(
            {"error": "Invalid file format. Only CSV files allowed"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    update_search_vector = (
        request.data.get("update_search_vector", "true").lower() == "true"
    )

    try:
        # Read CSV file
        csv_file = TextIOWrapper(file.file, encoding="utf-8")
        reader = csv.DictReader(csv_file)

        # Validate required headers
        required_headers = ["code"]

        if not all(header in reader.fieldnames for header in required_headers):
            return Response(
                {"error": f"Missing required headers: {required_headers}"},
                status=status.HTTP_400_BAD_REQUEST,
            )

    except UnicodeDecodeError:
        return Response(
            {"error": "Invalid file encoding. Please use UTF-8 encoding"},
            status=status.HTTP_400_BAD_REQUEST,
        )
    except Exception as e:
        return Response(
            {"error": f"Failed to read CSV file: {str(e)}"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    # Process CSV data
    created_count = 0
    updated_count = 0
    skipped_count = 0
    total_processed = 0
    errors = []

    with transaction.atomic():
        for row_num, row in enumerate(
            reader, start=2
        ):  # Start at 2 because of header row
            total_processed += 1

            # Validate required fields
            code = row.get("code", "").strip()
            if not code:
                errors.append(f"Row {row_num}: Missing or empty code field")
                skipped_count += 1
                continue

            # Validate code format (basic validation)
            if len(code) > 50:
                errors.append(
                    f'Row {row_num}: Code "{code}" exceeds maximum length of 50 characters'
                )
                skipped_count += 1
                continue

            try:
                # Check if record exists
                ayurvedha_obj, created = Ayurvedha.objects.get_or_create(
                    code=code,
                    defaults={
                        "english_name": row.get("english_name", "").strip() or None,
                        "description": row.get("description", "").strip() or None,
                        "hindi_name": row.get("hindi_name", "").strip() or None,
                        "diacritical_name": row.get("diacritical_name", "").strip()
                        or None,
                    },
                )

                if created:
                    created_count += 1
                else:
                    # Update existing record
                    updated_fields = []

                    new_english_name = row.get("english_name", "").strip() or None
                    if (
                        new_english_name
                        and new_english_name != ayurvedha_obj.english_name
                    ):
                        ayurvedha_obj.english_name = new_english_name
                        updated_fields.append("english_name")

                    new_description = row.get("description", "").strip() or None
                    if new_description and new_description != ayurvedha_obj.description:
                        ayurvedha_obj.description = new_description
                        updated_fields.append("description")

                    new_hindi_name = row.get("hindi_name", "").strip() or None
                    if new_hindi_name and new_hindi_name != ayurvedha_obj.hindi_name:
                        ayurvedha_obj.hindi_name = new_hindi_name
                        updated_fields.append("hindi_name")

                    new_diacritical_name = (
                        row.get("diacritical_name", "").strip() or None
                    )
                    if (
                        new_diacritical_name
                        and new_diacritical_name != ayurvedha_obj.diacritical_name
                    ):
                        ayurvedha_obj.diacritical_name = new_diacritical_name
                        updated_fields.append("diacritical_name")

                    if updated_fields:
                        ayurvedha_obj.save()
                        updated_count += 1

            except Exception as e:
                errors.append(
                    f'Row {row_num}: Failed to process code "{code}" - {str(e)}'
                )
                skipped_count += 1
                continue

    # Prepare response
    result = {
        "created": created_count,
        "updated": updated_count,
        "skipped": skipped_count,
        "total_processed": total_processed,
        "errors": errors,
        "summary": f"Processed {total_processed} rows: {created_count} created, {updated_count} updated, {skipped_count} skipped",
    }

    # Determine response status
    if errors and (created_count == 0 and updated_count == 0):
        status_code = status.HTTP_400_BAD_REQUEST
    elif errors:
        status_code = status.HTTP_207_MULTI_STATUS  # Partial success
    else:
        status_code = status.HTTP_200_OK

    return Response(result, status=status_code)


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
    summary="Siddha Autocomplete",
    description="Fast autocomplete for Siddha terms - returns only English name titles for optimal performance",
    parameters=[
        OpenApiParameter(
            name="q",
            type=OpenApiTypes.STR,
            location=OpenApiParameter.QUERY,
            description="Search term for autocomplete (searches across English, Tamil, and romanized names)",
            required=True,
            examples=[
                OpenApiExample(
                    "Partial English", value="fev", description="Matches 'fever'"
                ),
                OpenApiExample(
                    "Tamil term", value="காய்ச்சல்", description="Searches Tamil names"
                ),
                OpenApiExample(
                    "Romanized",
                    value="kaaychchal",
                    description="Searches romanized Tamil terms",
                ),
                OpenApiExample(
                    "Code search", value="S001", description="Searches by Siddha codes"
                ),
            ],
        ),
        OpenApiParameter(
            name="limit",
            type=OpenApiTypes.INT,
            location=OpenApiParameter.QUERY,
            description="Maximum results to return (default: 10, max: 20)",
            required=False,
        ),
    ],
    responses={
        200: OpenApiResponse(
            description="List of matching English name titles only",
            response={
                "type": "array",
                "items": {"type": "string"},
                "example": ["fever", "fever with headache", "febrile condition"],
            },
        )
    },
    tags=["Siddha"],
    operation_id="siddha_autocomplete",
)
@api_view(["GET"])
def siddha_autocomplete(request):
    search_term = request.query_params.get("q", "").strip()
    limit = min(int(request.query_params.get("limit", 10)), 20)

    if not search_term or len(search_term) < 1:
        return Response([])

    # Use the same pattern as your working Ayurveda fuzzy search
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

    # Exact matches for better autocomplete experience
    exact_qs = Siddha.objects.filter(
        Q(code__icontains=search_term)
        | Q(english_name__icontains=search_term)
        | Q(tamil_name__icontains=search_term)
        | Q(romanized_name__icontains=search_term)
    )

    # Combine queries and apply weighted scoring
    queryset = (
        (fuzzy_qs | exact_qs)
        .distinct()
        .annotate(
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
        )
        .order_by("-weighted_score", "english_name")
    )

    # Extract only english_name titles
    titles = list(queryset.values_list("english_name", flat=True)[:limit])
    return Response(titles)


@extend_schema(
    summary="Upload Siddha CSV",
    description="Upload CSV file to populate Siddha terms database. Updates existing records by code or creates new ones.",
    request={
        "multipart/form-data": {
            "type": "object",
            "properties": {
                "file": {
                    "type": "string",
                    "format": "binary",
                    "description": "CSV file with Siddha terms (columns: code, english_name, description, tamil_name, romanized_name, reference)",
                },
                "update_search_vector": {
                    "type": "boolean",
                    "description": "Whether to update search vectors after import",
                    "default": True,
                },
            },
            "required": ["file"],
        }
    },
    responses={
        200: OpenApiResponse(
            description="CSV processed successfully",
            response={
                "type": "object",
                "properties": {
                    "created": {
                        "type": "integer",
                        "description": "Number of new records created",
                    },
                    "updated": {
                        "type": "integer",
                        "description": "Number of existing records updated",
                    },
                    "skipped": {
                        "type": "integer",
                        "description": "Number of rows skipped due to errors",
                    },
                    "total_processed": {
                        "type": "integer",
                        "description": "Total rows processed",
                    },
                    "errors": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of error messages for failed rows",
                    },
                    "summary": {"type": "string", "description": "Processing summary"},
                },
            },
        ),
        400: OpenApiResponse(
            description="Bad request - invalid file or format",
            examples=[
                OpenApiExample(
                    "No file provided", value={"error": "CSV file not provided"}
                ),
                OpenApiExample(
                    "Invalid CSV format",
                    value={"error": "Invalid CSV format: missing required headers"},
                ),
            ],
        ),
        413: OpenApiResponse(description="File too large"),
    },
    tags=["Siddha"],
    operation_id="siddha_csv_upload",
)
@api_view(["POST"])
@parser_classes([MultiPartParser, FormParser])
def siddha_csv_upload(request):
    """
    Upload CSV file to populate Siddha terms database.

    CSV Format:
    - Required columns: code
    - Optional columns: english_name, description, tamil_name, romanized_name, reference
    - First row should contain headers
    - Encoding: UTF-8

    Processing Logic:
    - If code exists: Update the existing record
    - If code doesn't exist: Create new record
    - Ensures no duplicate codes in database
    """

    # Validate file upload
    file = request.FILES.get("file")
    if not file:
        return Response(
            {"error": "CSV file not provided"}, status=status.HTTP_400_BAD_REQUEST
        )

    # Check file size (limit to 10MB)
    if file.size > 10 * 1024 * 1024:
        return Response(
            {"error": "File size too large. Maximum 10MB allowed"},
            status=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
        )

    # Check file extension
    if not file.name.endswith(".csv"):
        return Response(
            {"error": "Invalid file format. Only CSV files allowed"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    update_search_vector = (
        request.data.get("update_search_vector", "true").lower() == "true"
    )

    try:
        # Read CSV file
        csv_file = TextIOWrapper(file.file, encoding="utf-8")
        reader = csv.DictReader(csv_file)

        # Validate required headers
        required_headers = ["code"]

        if not all(header in reader.fieldnames for header in required_headers):
            return Response(
                {"error": f"Missing required headers: {required_headers}"},
                status=status.HTTP_400_BAD_REQUEST,
            )

    except UnicodeDecodeError:
        return Response(
            {"error": "Invalid file encoding. Please use UTF-8 encoding"},
            status=status.HTTP_400_BAD_REQUEST,
        )
    except Exception as e:
        return Response(
            {"error": f"Failed to read CSV file: {str(e)}"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    # Process CSV data
    created_count = 0
    updated_count = 0
    skipped_count = 0
    total_processed = 0
    errors = []

    with transaction.atomic():
        for row_num, row in enumerate(
            reader, start=2
        ):  # Start at 2 because of header row
            total_processed += 1

            # Validate required fields
            code = row.get("code", "").strip()
            if not code:
                errors.append(f"Row {row_num}: Missing or empty code field")
                skipped_count += 1
                continue

            # Validate code format (basic validation)
            if len(code) > 50:
                errors.append(
                    f'Row {row_num}: Code "{code}" exceeds maximum length of 50 characters'
                )
                skipped_count += 1
                continue

            try:
                # Check if record exists
                siddha_obj, created = Siddha.objects.get_or_create(
                    code=code,
                    defaults={
                        "english_name": row.get("english_name", "").strip() or None,
                        "description": row.get("description", "").strip() or None,
                        "tamil_name": row.get("tamil_name", "").strip() or None,
                        "romanized_name": row.get("romanized_name", "").strip() or None,
                        "reference": row.get("reference", "").strip() or None,
                    },
                )

                if created:
                    created_count += 1
                else:
                    # Update existing record
                    updated_fields = []

                    new_english_name = row.get("english_name", "").strip() or None
                    if new_english_name and new_english_name != siddha_obj.english_name:
                        siddha_obj.english_name = new_english_name
                        updated_fields.append("english_name")

                    new_description = row.get("description", "").strip() or None
                    if new_description and new_description != siddha_obj.description:
                        siddha_obj.description = new_description
                        updated_fields.append("description")

                    new_tamil_name = row.get("tamil_name", "").strip() or None
                    if new_tamil_name and new_tamil_name != siddha_obj.tamil_name:
                        siddha_obj.tamil_name = new_tamil_name
                        updated_fields.append("tamil_name")

                    new_romanized_name = row.get("romanized_name", "").strip() or None
                    if (
                        new_romanized_name
                        and new_romanized_name != siddha_obj.romanized_name
                    ):
                        siddha_obj.romanized_name = new_romanized_name
                        updated_fields.append("romanized_name")

                    new_reference = row.get("reference", "").strip() or None
                    if new_reference and new_reference != siddha_obj.reference:
                        siddha_obj.reference = new_reference
                        updated_fields.append("reference")

                    if updated_fields:
                        siddha_obj.save()
                        updated_count += 1

            except Exception as e:
                errors.append(
                    f'Row {row_num}: Failed to process code "{code}" - {str(e)}'
                )
                skipped_count += 1
                continue

    # Prepare response
    result = {
        "created": created_count,
        "updated": updated_count,
        "skipped": skipped_count,
        "total_processed": total_processed,
        "errors": errors,
        "summary": f"Processed {total_processed} rows: {created_count} created, {updated_count} updated, {skipped_count} skipped",
    }

    # Determine response status
    if errors and (created_count == 0 and updated_count == 0):
        status_code = status.HTTP_400_BAD_REQUEST
    elif errors:
        status_code = status.HTTP_207_MULTI_STATUS  # Partial success
    else:
        status_code = status.HTTP_200_OK

    return Response(result, status=status_code)


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
    summary="Unani Autocomplete",
    description="Fast autocomplete for Unani terms - returns only English name titles for optimal performance",
    parameters=[
        OpenApiParameter(
            name="q",
            type=OpenApiTypes.STR,
            location=OpenApiParameter.QUERY,
            description="Search term for autocomplete (searches across English, Arabic, and romanized names)",
            required=True,
            examples=[
                OpenApiExample(
                    "Partial English", value="fev", description="Matches 'fever'"
                ),
                OpenApiExample(
                    "Arabic term", value="حمى", description="Searches Arabic names"
                ),
                OpenApiExample(
                    "Romanized",
                    value="hummA",
                    description="Searches romanized Arabic terms",
                ),
                OpenApiExample(
                    "Code search", value="U001", description="Searches by Unani codes"
                ),
            ],
        ),
        OpenApiParameter(
            name="limit",
            type=OpenApiTypes.INT,
            location=OpenApiParameter.QUERY,
            description="Maximum results to return (default: 10, max: 20)",
            required=False,
        ),
    ],
    responses={
        200: OpenApiResponse(
            description="List of matching English name titles only",
            response={
                "type": "array",
                "items": {"type": "string"},
                "example": ["fever", "fever with headache", "febrile condition"],
            },
        )
    },
    tags=["Unani"],
    operation_id="unani_autocomplete",
)
@api_view(["GET"])
def unani_autocomplete(request):
    search_term = request.query_params.get("q", "").strip()
    limit = min(int(request.query_params.get("limit", 10)), 20)

    if not search_term or len(search_term) < 1:
        return Response([])

    # Use the same pattern as your working Ayurveda fuzzy search
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

    # Exact matches for better autocomplete experience
    exact_qs = Unani.objects.filter(
        Q(code__icontains=search_term)
        | Q(english_name__icontains=search_term)
        | Q(arabic_name__icontains=search_term)
        | Q(romanized_name__icontains=search_term)
    )

    # Combine queries and apply weighted scoring
    queryset = (
        (fuzzy_qs | exact_qs)
        .distinct()
        .annotate(
            weighted_score=(
                TrigramSimilarity("english_name", search_term)
                * 2.5  # Prioritize English
                + TrigramSimilarity("code", search_term)
                * 1.0  # Standard weight for codes
                + TrigramSimilarity("arabic_name", search_term)
                * 0.8  # Arabic language support
                + TrigramSimilarity("romanized_name", search_term)
                * 0.8  # Romanized Arabic
            )
        )
        .order_by("-weighted_score", "english_name")
    )

    # Extract only english_name titles
    titles = list(queryset.values_list("english_name", flat=True)[:limit])
    return Response(titles)


@extend_schema(
    summary="Upload Unani CSV",
    description="Upload CSV file to populate Unani terms database. Updates existing records by code or creates new ones.",
    request={
        "multipart/form-data": {
            "type": "object",
            "properties": {
                "file": {
                    "type": "string",
                    "format": "binary",
                    "description": "CSV file with Unani terms (columns: code, english_name, description, arabic_name, romanized_name, reference)",
                },
                "update_search_vector": {
                    "type": "boolean",
                    "description": "Whether to update search vectors after import",
                    "default": True,
                },
            },
            "required": ["file"],
        }
    },
    responses={
        200: OpenApiResponse(
            description="CSV processed successfully",
            response={
                "type": "object",
                "properties": {
                    "created": {
                        "type": "integer",
                        "description": "Number of new records created",
                    },
                    "updated": {
                        "type": "integer",
                        "description": "Number of existing records updated",
                    },
                    "skipped": {
                        "type": "integer",
                        "description": "Number of rows skipped due to errors",
                    },
                    "total_processed": {
                        "type": "integer",
                        "description": "Total rows processed",
                    },
                    "errors": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of error messages for failed rows",
                    },
                    "summary": {"type": "string", "description": "Processing summary"},
                },
            },
        ),
        400: OpenApiResponse(
            description="Bad request - invalid file or format",
            examples=[
                OpenApiExample(
                    "No file provided", value={"error": "CSV file not provided"}
                ),
                OpenApiExample(
                    "Invalid CSV format",
                    value={"error": "Invalid CSV format: missing required headers"},
                ),
            ],
        ),
        413: OpenApiResponse(description="File too large"),
    },
    tags=["Unani"],
    operation_id="unani_csv_upload",
)
@api_view(["POST"])
@parser_classes([MultiPartParser, FormParser])
def unani_csv_upload(request):
    """
    Upload CSV file to populate Unani terms database.

    CSV Format:
    - Required columns: code
    - Optional columns: english_name, description, arabic_name, romanized_name, reference
    - First row should contain headers
    - Encoding: UTF-8

    Processing Logic:
    - If code exists: Update the existing record
    - If code doesn't exist: Create new record
    - Ensures no duplicate codes in database
    """

    # Validate file upload
    file = request.FILES.get("file")
    if not file:
        return Response(
            {"error": "CSV file not provided"}, status=status.HTTP_400_BAD_REQUEST
        )

    # Check file size (limit to 10MB)
    if file.size > 10 * 1024 * 1024:
        return Response(
            {"error": "File size too large. Maximum 10MB allowed"},
            status=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
        )

    # Check file extension
    if not file.name.endswith(".csv"):
        return Response(
            {"error": "Invalid file format. Only CSV files allowed"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    update_search_vector = (
        request.data.get("update_search_vector", "true").lower() == "true"
    )

    try:
        # Read CSV file
        csv_file = TextIOWrapper(file.file, encoding="utf-8")
        reader = csv.DictReader(csv_file)

        # Validate required headers
        required_headers = ["code"]

        if not all(header in reader.fieldnames for header in required_headers):
            return Response(
                {"error": f"Missing required headers: {required_headers}"},
                status=status.HTTP_400_BAD_REQUEST,
            )

    except UnicodeDecodeError:
        return Response(
            {"error": "Invalid file encoding. Please use UTF-8 encoding"},
            status=status.HTTP_400_BAD_REQUEST,
        )
    except Exception as e:
        return Response(
            {"error": f"Failed to read CSV file: {str(e)}"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    # Process CSV data
    created_count = 0
    updated_count = 0
    skipped_count = 0
    total_processed = 0
    errors = []

    with transaction.atomic():
        for row_num, row in enumerate(
            reader, start=2
        ):  # Start at 2 because of header row
            total_processed += 1

            # Validate required fields
            code = row.get("code", "").strip()
            if not code:
                errors.append(f"Row {row_num}: Missing or empty code field")
                skipped_count += 1
                continue

            # Validate code format (basic validation)
            if len(code) > 50:
                errors.append(
                    f'Row {row_num}: Code "{code}" exceeds maximum length of 50 characters'
                )
                skipped_count += 1
                continue

            try:
                # Check if record exists
                unani_obj, created = Unani.objects.get_or_create(
                    code=code,
                    defaults={
                        "english_name": row.get("english_name", "").strip() or None,
                        "description": row.get("description", "").strip() or None,
                        "arabic_name": row.get("arabic_name", "").strip() or None,
                        "romanized_name": row.get("romanized_name", "").strip() or None,
                        "reference": row.get("reference", "").strip() or None,
                    },
                )

                if created:
                    created_count += 1
                else:
                    # Update existing record
                    updated_fields = []

                    new_english_name = row.get("english_name", "").strip() or None
                    if new_english_name and new_english_name != unani_obj.english_name:
                        unani_obj.english_name = new_english_name
                        updated_fields.append("english_name")

                    new_description = row.get("description", "").strip() or None
                    if new_description and new_description != unani_obj.description:
                        unani_obj.description = new_description
                        updated_fields.append("description")

                    new_arabic_name = row.get("arabic_name", "").strip() or None
                    if new_arabic_name and new_arabic_name != unani_obj.arabic_name:
                        unani_obj.arabic_name = new_arabic_name
                        updated_fields.append("arabic_name")

                    new_romanized_name = row.get("romanized_name", "").strip() or None
                    if (
                        new_romanized_name
                        and new_romanized_name != unani_obj.romanized_name
                    ):
                        unani_obj.romanized_name = new_romanized_name
                        updated_fields.append("romanized_name")

                    new_reference = row.get("reference", "").strip() or None
                    if new_reference and new_reference != unani_obj.reference:
                        unani_obj.reference = new_reference
                        updated_fields.append("reference")

                    if updated_fields:
                        unani_obj.save()
                        updated_count += 1

            except Exception as e:
                errors.append(
                    f'Row {row_num}: Failed to process code "{code}" - {str(e)}'
                )
                skipped_count += 1
                continue

    # Prepare response
    result = {
        "created": created_count,
        "updated": updated_count,
        "skipped": skipped_count,
        "total_processed": total_processed,
        "errors": errors,
        "summary": f"Processed {total_processed} rows: {created_count} created, {updated_count} updated, {skipped_count} skipped",
    }

    # Determine response status
    if errors and (created_count == 0 and updated_count == 0):
        status_code = status.HTTP_400_BAD_REQUEST
    elif errors:
        status_code = status.HTTP_207_MULTI_STATUS  # Partial success
    else:
        status_code = status.HTTP_200_OK

    return Response(result, status=status_code)


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


@extend_schema(
    summary="ICD-11 Autocomplete",
    description="Fast autocomplete for ICD-11 terms - returns only matching titles for optimal performance",
    parameters=[
        OpenApiParameter(
            name="q",
            type=OpenApiTypes.STR,
            location=OpenApiParameter.QUERY,
            description="Search term for autocomplete (searches across title, code, and definition)",
            required=True,
            examples=[
                OpenApiExample(
                    "Partial term", value="diab", description="Matches 'diabetes'"
                ),
                OpenApiExample(
                    "Code search", value="E10", description="Matches ICD codes"
                ),
                OpenApiExample(
                    "Medical term",
                    value="blood",
                    description="Matches blood-related conditions",
                ),
            ],
        ),
        OpenApiParameter(
            name="limit",
            type=OpenApiTypes.INT,
            location=OpenApiParameter.QUERY,
            description="Maximum results to return (default: 10, max: 20)",
            required=False,
        ),
    ],
    responses={
        200: OpenApiResponse(
            description="List of matching title names only",
            response={
                "type": "array",
                "items": {"type": "string"},
                "example": [
                    "diabetes mellitus",
                    "diabetes type 1",
                    "diabetic nephropathy",
                ],
            },
        )
    },
    tags=["ICD-11"],
    operation_id="icd11_autocomplete",
)
@api_view(["GET"])
def icd11_autocomplete(request):
    search_term = request.query_params.get("q", "").strip()
    limit = min(int(request.query_params.get("limit", 10)), 20)

    if not search_term or len(search_term) < 1:
        return Response([])

    # Use the same pattern as your working Ayurveda fuzzy search
    fuzzy_qs = ICD11Term.objects.annotate(
        similarity_code=TrigramSimilarity("code", search_term),
        similarity_title=TrigramSimilarity("title", search_term),
        similarity_definition=TrigramSimilarity("definition", search_term),
    ).filter(
        Q(similarity_code__gt=0.1)
        | Q(similarity_title__gt=0.1)
        | Q(similarity_definition__gt=0.1)
    )

    # Exact matches for better autocomplete experience
    exact_qs = ICD11Term.objects.filter(
        Q(code__icontains=search_term)
        | Q(title__icontains=search_term)
        | Q(definition__icontains=search_term)
    )

    # Combine queries and apply weighted scoring
    queryset = (
        (fuzzy_qs | exact_qs)
        .distinct()
        .annotate(
            weighted_score=(
                TrigramSimilarity("title", search_term) * 2.5  # Prioritize title
                + TrigramSimilarity("code", search_term)
                * 1.0  # Standard weight for codes
                + TrigramSimilarity("definition", search_term)
                * 0.8  # Definition support
            )
        )
        .order_by("-weighted_score", "title")
    )

    # Extract only title
    titles = list(queryset.values_list("title", flat=True)[:limit])
    return Response(titles)


@extend_schema(
    summary="Combined ICD-11 and NAMASTE Concept Search",
    description="""
    Search for ICD-11 terms and retrieve their related NAMASTE concepts in one API call.
    
    **Features:**
    - Advanced fuzzy search across ICD-11 terms using PostgreSQL trigrams
    - Full-text search with ranking when search_vector is available
    - Returns ICD-11: id, code, title, definition
    - Returns related NAMASTE concepts: id, code, english_name, local_name
    - Returns null for NAMASTE systems with no mappings
    - Confidence scores and mapping metadata
    - Pagination support with configurable page size
    - Multiple search strategies (fuzzy, full-text, exact)
    """,
    parameters=[
        OpenApiParameter(
            name="q",
            type=OpenApiTypes.STR,
            location=OpenApiParameter.QUERY,
            description="Search query term",
            required=True,
            examples=[
                OpenApiExample("Medical condition", value="diabetes"),
                OpenApiExample("ICD code", value="E11"),
                OpenApiExample("Complex term", value="chronic kidney disease"),
            ],
        ),
        OpenApiParameter(
            name="fuzzy",
            type=OpenApiTypes.BOOL,
            location=OpenApiParameter.QUERY,
            description="Use fuzzy search with trigram similarity",
            required=False,
            examples=[
                OpenApiExample("Enable fuzzy search", value=True),
                OpenApiExample("Disable fuzzy search", value=False),
            ],
        ),
        OpenApiParameter(
            name="use_fts",
            type=OpenApiTypes.BOOL,
            location=OpenApiParameter.QUERY,
            description="Use full-text search with search_vector",
            required=False,
            examples=[
                OpenApiExample("Enable FTS", value=True),
                OpenApiExample("Disable FTS", value=False),
            ],
        ),
        OpenApiParameter(
            name="threshold",
            type=OpenApiTypes.FLOAT,
            location=OpenApiParameter.QUERY,
            description="Similarity threshold for fuzzy search (0.0-1.0)",
            required=False,
            examples=[
                OpenApiExample("Low threshold", value=0.2),
                OpenApiExample("Medium threshold", value=0.4),
                OpenApiExample("High threshold", value=0.6),
            ],
        ),
        OpenApiParameter(
            name="page",
            type=OpenApiTypes.INT,
            location=OpenApiParameter.QUERY,
            description="Page number for pagination",
            required=False,
            examples=[
                OpenApiExample("First page", value=1),
                OpenApiExample("Second page", value=2),
            ],
        ),
        OpenApiParameter(
            name="page_size",
            type=OpenApiTypes.INT,
            location=OpenApiParameter.QUERY,
            description="Number of results per page (max 50)",
            required=False,
            examples=[
                OpenApiExample("Small page", value=10),
                OpenApiExample("Default page", value=20),
                OpenApiExample("Large page", value=50),
            ],
        ),
    ],
    responses={
        200: OpenApiResponse(
            response=CombinedSearchResponseSerializer,
            description="Successful search with ICD-11 results and related NAMASTE concepts",
            examples=[
                OpenApiExample(
                    "Successful search response",
                    value={
                        "results": [
                            {
                                "id": 123,
                                "code": "E11",
                                "title": "Type 2 diabetes mellitus",
                                "definition": "A metabolic disorder characterized by high blood sugar",
                                "related_ayurveda": {
                                    "id": 456,
                                    "code": "AY-MADHUMEHA-001",
                                    "english_name": "Madhumeha",
                                    "local_name": "मधुमेह",
                                },
                                "related_siddha": None,
                                "related_unani": {
                                    "id": 789,
                                    "code": "UN-ZIABITUS-001",
                                    "english_name": "Ziabitus Sukari",
                                    "local_name": "ذیابیطس سکری",
                                },
                                "mapping_info": {
                                    "id": 101,
                                    "confidence_score": 0.92,
                                    "icd_similarity": 0.88,
                                    "source_system": "ayurveda",
                                },
                                "search_score": 8.5,
                            }
                        ],
                        "pagination": {
                            "page": 1,
                            "page_size": 20,
                            "total_pages": 5,
                            "total_count": 95,
                            "has_next": True,
                            "has_previous": False,
                        },
                        "search_metadata": {
                            "query": "diabetes",
                            "search_strategy": "fuzzy",
                            "total_icd_matches": 95,
                            "matches_with_namaste": 23,
                            "executed_at": "2025-09-26T22:17:00Z",
                        },
                    },
                )
            ],
        ),
        400: OpenApiResponse(
            response=ErrorResponseSerializer,
            description="Bad request - missing or invalid parameters",
        ),
        500: OpenApiResponse(
            response=ErrorResponseSerializer, description="Internal server error"
        ),
    },
    tags=["Combined Search"],
    operation_id="combinedICD11NAMASTESearch",
)
@api_view(["GET"])
@renderer_classes([JSONRenderer])
@cache_page(60 * 5)  # 5-minute cache
def combined_icd11_namaste_search(request):
    """
    Combined search API that searches ICD-11 terms and returns related NAMASTE concepts
    URL: /terminologies/search/combined/
    """

    try:
        # Get and validate search parameters
        search_term = request.query_params.get("q", "").strip()
        if not search_term:
            return Response(
                {
                    "error": "Missing search query",
                    "message": "Parameter 'q' is required",
                    "code": "MISSING_QUERY",
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Search configuration
        use_fuzzy = request.query_params.get("fuzzy", "").lower() in ["true", "1"]
        use_fts = request.query_params.get("use_fts", "").lower() in ["true", "1"]
        similarity_threshold = float(request.query_params.get("threshold", "0.2"))

        # Pagination
        page = int(request.query_params.get("page", 1))
        page_size = min(int(request.query_params.get("page_size", 20)), 50)

        # Perform ICD-11 search based on strategy
        icd_queryset, search_strategy = perform_icd11_search(
            search_term, use_fuzzy, use_fts, similarity_threshold
        )

        # Apply pagination to ICD-11 results
        paginator = Paginator(icd_queryset, page_size)
        try:
            page_obj = paginator.get_page(page)
        except Exception as e:
            logger.warning(f"Pagination error: {str(e)}")
            page_obj = paginator.get_page(1)

        # Get ICD-11 results for current page
        icd_results = list(page_obj.object_list)

        # Build combined results with NAMASTE mappings
        combined_results = []
        namaste_match_count = 0

        for icd_term in icd_results:
            # Get the best mapping for this ICD-11 term
            mapping = get_best_namaste_mapping(icd_term)

            # Build result object
            result = {
                "id": icd_term.id,
                "code": icd_term.code,
                "title": icd_term.title,
                "definition": icd_term.definition,
                "related_ayurveda": None,
                "related_siddha": None,
                "related_unani": None,
                "mapping_info": None,
                "search_score": getattr(icd_term, "search_score", 0.0),
            }

            # Add NAMASTE mappings if they exist
            if mapping:
                namaste_match_count += 1
                result["mapping_info"] = {
                    "id": mapping.id,
                    "confidence_score": mapping.confidence_score,
                    "icd_similarity": mapping.icd_similarity,
                    "source_system": mapping.source_system,
                }

                # Add related concepts based on source system
                if (
                    mapping.source_system == "ayurveda"
                    and mapping.primary_ayurveda_term
                ):
                    result["related_ayurveda"] = format_namaste_concept(
                        mapping.primary_ayurveda_term, "ayurveda"
                    )
                elif mapping.source_system == "siddha" and mapping.primary_siddha_term:
                    result["related_siddha"] = format_namaste_concept(
                        mapping.primary_siddha_term, "siddha"
                    )
                elif mapping.source_system == "unani" and mapping.primary_unani_term:
                    result["related_unani"] = format_namaste_concept(
                        mapping.primary_unani_term, "unani"
                    )

                # Add cross-system mappings if they exist
                if mapping.cross_ayurveda_term:
                    result["related_ayurveda"] = format_namaste_concept(
                        mapping.cross_ayurveda_term, "ayurveda"
                    )
                if mapping.cross_siddha_term:
                    result["related_siddha"] = format_namaste_concept(
                        mapping.cross_siddha_term, "siddha"
                    )
                if mapping.cross_unani_term:
                    result["related_unani"] = format_namaste_concept(
                        mapping.cross_unani_term, "unani"
                    )

            combined_results.append(result)

        # Build response
        response_data = {
            "results": combined_results,
            "pagination": {
                "page": page_obj.number,
                "page_size": page_size,
                "total_pages": paginator.num_pages,
                "total_count": paginator.count,
                "has_next": page_obj.has_next(),
                "has_previous": page_obj.has_previous(),
            },
            "search_metadata": {
                "query": search_term,
                "search_strategy": search_strategy,
                "total_icd_matches": paginator.count,
                "matches_with_namaste": namaste_match_count,
                "executed_at": timezone.now().isoformat(),
                "similarity_threshold": similarity_threshold if use_fuzzy else None,
            },
        }

        return Response(response_data, status=status.HTTP_200_OK)

    except ValueError as e:
        return Response(
            {
                "error": "Invalid parameter",
                "message": str(e),
                "code": "INVALID_PARAMETER",
            },
            status=status.HTTP_400_BAD_REQUEST,
        )
    except Exception as e:
        logger.error(f"Error in combined search: {str(e)}", exc_info=True)
        return Response(
            {
                "error": "Internal server error",
                "message": "Failed to perform combined search",
                "code": "INTERNAL_ERROR",
                "debug_info": str(e) if settings.DEBUG else None,
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def perform_icd11_search(search_term, use_fuzzy, use_fts, similarity_threshold):
    """
    Perform ICD-11 search based on selected strategy
    Returns tuple of (queryset, strategy_name)
    """

    if use_fts and hasattr(ICD11Term, "search_vector"):
        # Full-text search strategy
        search_query = SearchQuery(search_term)
        queryset = (
            ICD11Term.objects.filter(search_vector=search_query)
            .annotate(
                search_score=SearchRank(F("search_vector"), search_query),
                rank=SearchRank(F("search_vector"), search_query),
            )
            .order_by("-rank", "code")
        )
        return queryset, "full_text_search"

    elif use_fuzzy:
        # Fuzzy search strategy with trigram similarity
        queryset = (
            ICD11Term.objects.annotate(
                code_sim=TrigramSimilarity("code", search_term),
                title_sim=TrigramSimilarity("title", search_term),
                definition_sim=TrigramSimilarity("definition", search_term),
                # JSON field similarity
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
            )
            .filter(
                Q(code_sim__gte=similarity_threshold)
                | Q(title_sim__gte=similarity_threshold)
                | Q(definition_sim__gte=similarity_threshold)
                | Q(index_terms_sim__gte=similarity_threshold)
            )
            .annotate(
                search_score=(
                    F("code_sim")
                    + F("title_sim")
                    + F("definition_sim")
                    + F("index_terms_sim")
                )
            )
            .distinct()
            .order_by("-search_score", "code")
        )
        return queryset, "fuzzy_search"

    else:
        # Traditional icontains search
        basic_filter = (
            Q(code__icontains=search_term)
            | Q(title__icontains=search_term)
            | Q(definition__icontains=search_term)
        )

        json_filter = (
            Q(index_terms__icontains=search_term)
            | Q(inclusions__icontains=search_term)
            | Q(exclusions__icontains=search_term)
        )

        queryset = (
            ICD11Term.objects.filter(basic_filter | json_filter)
            .distinct()
            .annotate(
                search_score=Case(
                    When(title__iexact=search_term, then=Value(10.0)),
                    When(code__iexact=search_term, then=Value(9.0)),
                    When(title__istartswith=search_term, then=Value(8.0)),
                    When(code__istartswith=search_term, then=Value(7.0)),
                    When(title__icontains=search_term, then=Value(6.0)),
                    default=Value(1.0),
                    output_field=FloatField(),
                )
            )
            .order_by("-search_score", "code")
        )
        return queryset, "exact_search"


def get_best_namaste_mapping(icd_term):
    """
    Get the best NAMASTE mapping for an ICD-11 term
    Returns the mapping with highest confidence score
    """
    try:
        return (
            TermMapping.objects.filter(icd_term=icd_term)
            .select_related(
                "primary_ayurveda_term",
                "primary_siddha_term",
                "primary_unani_term",
                "cross_ayurveda_term",
                "cross_siddha_term",
                "cross_unani_term",
            )
            .order_by("-confidence_score", "-icd_similarity")
            .first()
        )
    except Exception as e:
        logger.warning(f"Error getting NAMASTE mapping: {str(e)}")
        return None


def format_namaste_concept(concept, system_type):
    """
    Format a NAMASTE concept for API response
    Returns dict with id, code, english_name, local_name
    """
    if not concept:
        return None

    try:
        local_name = None
        if system_type == "ayurveda":
            local_name = getattr(concept, "hindi_name", None)
        elif system_type == "siddha":
            local_name = getattr(concept, "tamil_name", None)
        elif system_type == "unani":
            local_name = getattr(concept, "arabic_name", None)

        return {
            "id": concept.id,
            "code": concept.code,
            "english_name": concept.english_name,
            "local_name": local_name,
        }
    except Exception as e:
        logger.warning(f"Error formatting {system_type} concept: {str(e)}")
        return None


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
