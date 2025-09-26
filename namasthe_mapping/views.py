"""
views.py - Optimized Function-Based Views for NAMASTE Concept Detail APIs
Three separate views for Ayurveda, Siddha, and Unani systems with ICD-11 mappings

Optimized for Django + PostgreSQL + TinyBioBERT + ONNX integration
Includes Swagger/OpenAPI documentation with drf-spectacular
"""

import logging

from django.conf import settings
from django.contrib.contenttypes.models import ContentType
from django.core.exceptions import ValidationError
from django.db.models import Avg, Count, Max, Min, Prefetch, Q
from django.http import Http404, JsonResponse
from django.utils import timezone
from django.views.decorators.cache import cache_page
from django.views.decorators.http import require_http_methods
from drf_spectacular.openapi import AutoSchema
from drf_spectacular.types import OpenApiTypes
from drf_spectacular.utils import (
    OpenApiExample,
    OpenApiParameter,
    OpenApiResponse,
    extend_schema,
)
from rest_framework import status

# DRF and Swagger imports
from rest_framework.decorators import api_view, renderer_classes
from rest_framework.renderers import JSONRenderer
from rest_framework.response import Response

# Local imports
from terminologies.models import Ayurvedha, Siddha, Unani

from .models import ConceptMapping, MappingAudit, TerminologyMapping
from .serializers import (
    AyurvedaConceptSerializer,
    ConceptDetailResponseSerializer,
    ConceptMappingDetailSerializer,
    ErrorResponseSerializer,
    ICD11SearchResponseSerializer,
    ManualMappingCreateSerializer,
    MappingCreateResponseSerializer,
    MappingUpdateResponseSerializer,
    MappingUpdateSerializer,
    SiddhaConceptSerializer,
    UnaniConceptSerializer,
)

logger = logging.getLogger(__name__)


def get_optimized_mappings_query(content_type, concept_id, filters=None):
    """Build optimized ConceptMapping query with all necessary joins"""
    query = (
        ConceptMapping.objects.filter(
            source_content_type=content_type, source_object_id=concept_id
        )
        .select_related("mapping", "target_concept", "source_content_type")
        .prefetch_related(
            Prefetch(
                "audit_entries",
                queryset=MappingAudit.objects.select_related().order_by("-timestamp")[
                    :3
                ],
                to_attr="recent_audit_entries",
            )
        )
    )

    # Apply filters
    if filters:
        if filters.get("min_confidence"):
            try:
                min_conf = float(filters["min_confidence"])
                if 0.0 <= min_conf <= 1.0:
                    query = query.filter(confidence_score__gte=min_conf)
            except (ValueError, TypeError):
                pass

        if filters.get("validated_only"):
            query = query.filter(is_validated=True)

        if filters.get("high_confidence_only"):
            query = query.filter(is_high_confidence=True)

    return query.order_by("-confidence_score", "-similarity_score", "-created_at")


def format_statistics_response(mapping_stats):
    """Format mapping statistics for consistent response"""
    total = mapping_stats["total_mappings"]

    return {
        "total_mappings": total,
        "validated_mappings": mapping_stats["validated_mappings"],
        "high_confidence_mappings": mapping_stats["high_confidence_mappings"],
        "needs_review": mapping_stats.get("needs_review_count", 0),
        "has_issues": mapping_stats.get("has_issues_count", 0),
        "rates": {
            "validation_rate": round(
                (mapping_stats["validated_mappings"] / total * 100)
                if total > 0
                else 0.0,
                1,
            ),
            "high_confidence_rate": round(
                (mapping_stats["high_confidence_mappings"] / total * 100)
                if total > 0
                else 0.0,
                1,
            ),
        },
        "quality_metrics": {
            "average_confidence": round(mapping_stats.get("avg_confidence") or 0.0, 3),
            "average_similarity": round(mapping_stats.get("avg_similarity") or 0.0, 3),
            "confidence_range": {
                "max": round(mapping_stats.get("max_confidence") or 0.0, 3),
                "min": round(mapping_stats.get("min_confidence") or 0.0, 3),
            },
        },
    }


def build_pagination_info(page, page_size, total_count):
    """Build pagination information"""
    total_pages = (total_count + page_size - 1) // page_size if total_count > 0 else 0

    return {
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
        "total_count": total_count,
        "has_next": page < total_pages,
        "has_previous": page > 1,
        "next_page": page + 1 if page < total_pages else None,
        "previous_page": page - 1 if page > 1 else None,
    }


# =============================================================================
# AYURVEDA CONCEPT DETAIL VIEW WITH SWAGGER DOCUMENTATION
# =============================================================================


@extend_schema(
    summary="Get Ayurveda Concept Details",
    description="""
    Retrieve detailed information for an Ayurveda concept with all mapped ICD-11 terms.
    
    **Features:**
    - Complete Ayurveda concept details (Hindi name, diacritical name, etc.)
    - All mapped ICD-11 terms with TinyBioBERT confidence scores
    - Comprehensive mapping statistics and quality metrics
    - Advanced filtering by confidence score and validation status
    - Pagination for large mapping sets
    - Optional 768-dimensional TinyBioBERT embeddings for ONNX processing
    """,
    parameters=[
        OpenApiParameter(
            name="concept_id",
            type=OpenApiTypes.INT,
            location=OpenApiParameter.PATH,
            description="Primary key ID of the Ayurveda concept",
            required=True,
            examples=[
                OpenApiExample("Small ID", value=123),
                OpenApiExample("Large ID", value=456789),
                OpenApiExample("Single digit", value=7),
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
                OpenApiExample("Later page", value=10),
            ],
        ),
        OpenApiParameter(
            name="min_confidence",
            type=OpenApiTypes.FLOAT,
            location=OpenApiParameter.QUERY,
            description="Filter mappings by minimum TinyBioBERT confidence score (0.0-1.0)",
            required=False,
            examples=[
                OpenApiExample("Low confidence", value=0.5),
                OpenApiExample("Medium confidence", value=0.75),
                OpenApiExample("High confidence", value=0.85),
                OpenApiExample("Very high confidence", value=0.95),
            ],
        ),
        OpenApiParameter(
            name="validated_only",
            type=OpenApiTypes.BOOL,
            location=OpenApiParameter.QUERY,
            description="Show only expert-validated mappings",
            required=False,
            examples=[
                OpenApiExample("Show all mappings", value=False),
                OpenApiExample("Only validated mappings", value=True),
            ],
        ),
        OpenApiParameter(
            name="include_embeddings",
            type=OpenApiTypes.BOOL,
            location=OpenApiParameter.QUERY,
            description="Include 768-dimensional TinyBioBERT embeddings in response",
            required=False,
            examples=[
                OpenApiExample("Without embeddings", value=False),
                OpenApiExample("With TinyBioBERT embeddings", value=True),
            ],
        ),
    ],
    responses={
        200: OpenApiResponse(
            response=ConceptDetailResponseSerializer,
            description="Successful retrieval of Ayurveda concept with all mappings and statistics",
            examples=[
                OpenApiExample(
                    "Complete Ayurveda concept response",
                    value={
                        "concept": {
                            "id": 123,
                            "code": "AY-FEVER-001",
                            "english_name": "Fever",
                            "hindi_name": "ज्वर",
                            "diacritical_name": "Jwara",
                        },
                        "mapping_statistics": {
                            "total_mappings": 25,
                            "validated_mappings": 20,
                            "high_confidence_mappings": 15,
                        },
                    },
                )
            ],
        ),
        400: OpenApiResponse(
            response=ErrorResponseSerializer,
            description="Bad request - invalid concept ID or parameters",
            examples=[
                OpenApiExample(
                    "Invalid concept ID",
                    value={
                        "error": "Invalid concept ID",
                        "message": "Concept ID must be a valid integer",
                        "code": "INVALID_CONCEPT_ID",
                    },
                )
            ],
        ),
        404: OpenApiResponse(
            response=ErrorResponseSerializer, description="Concept not found"
        ),
        500: OpenApiResponse(
            response=ErrorResponseSerializer, description="Internal server error"
        ),
    },
    tags=["NAMASTE Concepts"],
    operation_id="getAyurvedaConceptDetail",
)
@api_view(["GET"])
@renderer_classes([JSONRenderer])
@cache_page(60 * 10)  # 10-minute cache
def get_ayurveda_concept_detail(request, concept_id):
    """
    Get detailed information for an Ayurveda concept with all ICD-11 mappings
    URL: /namasthe_mapping/ayurveda/{concept_id}/detail/
    """

    try:
        # Validate concept_id
        try:
            concept_id = int(concept_id)
        except (ValueError, TypeError):
            return Response(
                {
                    "error": "Invalid concept ID",
                    "message": "Concept ID must be a valid integer",
                    "code": "INVALID_CONCEPT_ID",
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Get Ayurveda concept
        try:
            ayurveda_concept = Ayurvedha.objects.get(pk=concept_id)
        except Ayurvedha.DoesNotExist:
            return Response(
                {
                    "error": "Concept not found",
                    "message": f"Ayurveda concept with ID {concept_id} not found",
                    "code": "CONCEPT_NOT_FOUND",
                },
                status=status.HTTP_404_NOT_FOUND,
            )

        # Parse and validate query parameters
        page = max(1, int(request.GET.get("page", 1)))
        page_size = min(int(request.GET.get("page_size", 20)), 100)
        include_embeddings = request.GET.get("include_embeddings", "").lower() == "true"

        filters = {
            "min_confidence": request.GET.get("min_confidence"),
            "validated_only": request.GET.get("validated_only", "").lower() == "true",
            "high_confidence_only": request.GET.get("high_confidence_only", "").lower()
            == "true",
        }

        # Get ContentType and build optimized query
        content_type = ContentType.objects.get_for_model(Ayurvedha)
        mappings_query = get_optimized_mappings_query(content_type, concept_id, filters)

        # Get comprehensive statistics with single aggregate query
        mapping_stats = mappings_query.aggregate(
            total_mappings=Count("id"),
            validated_mappings=Count("id", filter=Q(is_validated=True)),
            high_confidence_mappings=Count("id", filter=Q(is_high_confidence=True)),
            needs_review_count=Count("id", filter=Q(needs_review=True)),
            has_issues_count=Count("id", filter=Q(has_issues=True)),
            avg_confidence=Avg("confidence_score"),
            avg_similarity=Avg("similarity_score"),
            max_confidence=Max("confidence_score"),
            min_confidence=Min("confidence_score"),
        )

        # Implement pagination
        total_count = mapping_stats["total_mappings"]
        offset = (page - 1) * page_size
        mappings_page = mappings_query[offset : offset + page_size]

        # Serialize concept and mappings
        concept_serializer = AyurvedaConceptSerializer(ayurveda_concept)
        mappings_serializer = ConceptMappingDetailSerializer(
            mappings_page,
            many=True,
            context={"request": request, "include_embeddings": include_embeddings},
        )

        # Build comprehensive response
        response_data = {
            "concept": concept_serializer.data,
            "mapping_statistics": format_statistics_response(mapping_stats),
            "mappings": {
                "data": mappings_serializer.data,
                "pagination": build_pagination_info(page, page_size, total_count),
            },
            "filters_applied": {
                "min_confidence": filters["min_confidence"],
                "validated_only": filters["validated_only"],
                "high_confidence_only": filters["high_confidence_only"],
                "include_embeddings": include_embeddings,
            },
            "meta": {
                "system": "ayurveda",
                "concept_id": concept_id,
                "retrieved_at": timezone.now().isoformat(),
                "cache_duration": 600,
                "api_version": "1.0.0",
            },
        }

        return Response(response_data, status=status.HTTP_200_OK)

    except Exception as e:
        logger.error(f"Error in Ayurveda concept detail view: {str(e)}", exc_info=True)
        return Response(
            {
                "error": "Internal server error",
                "message": "Failed to retrieve Ayurveda concept details",
                "code": "INTERNAL_ERROR",
                "debug_info": str(e) if settings.DEBUG else None,
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


# =============================================================================
# SIDDHA CONCEPT DETAIL VIEW WITH SWAGGER DOCUMENTATION
# =============================================================================


@extend_schema(
    summary="Get Siddha Concept Details",
    description="""
    Retrieve detailed information for a Siddha concept with all mapped ICD-11 terms.
    
    **Features:**
    - Complete Siddha concept details (Tamil name, romanized name, references)
    - All mapped ICD-11 terms with TinyBioBERT confidence scores
    - Comprehensive mapping statistics and quality metrics
    - Advanced filtering by confidence score and validation status
    - Pagination for large mapping sets
    - Optional 768-dimensional TinyBioBERT embeddings for ONNX processing
    """,
    parameters=[
        OpenApiParameter(
            name="concept_id",
            type=OpenApiTypes.INT,
            location=OpenApiParameter.PATH,
            description="Primary key ID of the Siddha concept",
            required=True,
            examples=[
                OpenApiExample("Small ID", value=123),
                OpenApiExample("Large ID", value=456789),
                OpenApiExample("Single digit", value=7),
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
                OpenApiExample("Later page", value=10),
            ],
        ),
        OpenApiParameter(
            name="min_confidence",
            type=OpenApiTypes.FLOAT,
            location=OpenApiParameter.QUERY,
            description="Filter mappings by minimum TinyBioBERT confidence score (0.0-1.0)",
            required=False,
            examples=[
                OpenApiExample("Low confidence", value=0.5),
                OpenApiExample("Medium confidence", value=0.75),
                OpenApiExample("High confidence", value=0.85),
                OpenApiExample("Very high confidence", value=0.95),
            ],
        ),
        OpenApiParameter(
            name="validated_only",
            type=OpenApiTypes.BOOL,
            location=OpenApiParameter.QUERY,
            description="Show only expert-validated mappings",
            required=False,
            examples=[
                OpenApiExample("Show all mappings", value=False),
                OpenApiExample("Only validated mappings", value=True),
            ],
        ),
        OpenApiParameter(
            name="include_embeddings",
            type=OpenApiTypes.BOOL,
            location=OpenApiParameter.QUERY,
            description="Include 768-dimensional TinyBioBERT embeddings in response",
            required=False,
            examples=[
                OpenApiExample("Without embeddings", value=False),
                OpenApiExample("With TinyBioBERT embeddings", value=True),
            ],
        ),
    ],
    responses={
        200: OpenApiResponse(
            response=ConceptDetailResponseSerializer,
            description="Successful retrieval of Siddha concept with all mappings and statistics",
            examples=[
                OpenApiExample(
                    "Complete Siddha concept response",
                    value={
                        "concept": {
                            "id": 123,
                            "code": "SI-FEVER-001",
                            "english_name": "Fever",
                            "tamil_name": "காய்ச்சல்",
                            "romanized_name": "Kaychaal",
                            "reference": "Agasthiyar Gunavagadam",
                        },
                        "mapping_statistics": {
                            "total_mappings": 25,
                            "validated_mappings": 20,
                            "high_confidence_mappings": 15,
                        },
                    },
                )
            ],
        ),
        400: OpenApiResponse(
            response=ErrorResponseSerializer,
            description="Bad request - invalid concept ID or parameters",
            examples=[
                OpenApiExample(
                    "Invalid concept ID",
                    value={
                        "error": "Invalid concept ID",
                        "message": "Concept ID must be a valid integer",
                        "code": "INVALID_CONCEPT_ID",
                    },
                )
            ],
        ),
        404: OpenApiResponse(
            response=ErrorResponseSerializer, description="Concept not found"
        ),
        500: OpenApiResponse(
            response=ErrorResponseSerializer, description="Internal server error"
        ),
    },
    tags=["NAMASTE Concepts"],
    operation_id="getSiddhaConceptDetail",
)
@api_view(["GET"])
@renderer_classes([JSONRenderer])
@cache_page(60 * 10)  # 10-minute cache
def get_siddha_concept_detail(request, concept_id):
    """
    Get detailed information for a Siddha concept with all ICD-11 mappings
    URL: /namasthe_mapping/siddha/{concept_id}/detail/
    """

    try:
        # Validate concept_id
        try:
            concept_id = int(concept_id)
        except (ValueError, TypeError):
            return Response(
                {
                    "error": "Invalid concept ID",
                    "message": "Concept ID must be a valid integer",
                    "code": "INVALID_CONCEPT_ID",
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Get Siddha concept
        try:
            siddha_concept = Siddha.objects.get(pk=concept_id)
        except Siddha.DoesNotExist:
            return Response(
                {
                    "error": "Concept not found",
                    "message": f"Siddha concept with ID {concept_id} not found",
                    "code": "CONCEPT_NOT_FOUND",
                },
                status=status.HTTP_404_NOT_FOUND,
            )

        # Parse and validate query parameters
        page = max(1, int(request.GET.get("page", 1)))
        page_size = min(int(request.GET.get("page_size", 20)), 100)
        include_embeddings = request.GET.get("include_embeddings", "").lower() == "true"

        filters = {
            "min_confidence": request.GET.get("min_confidence"),
            "validated_only": request.GET.get("validated_only", "").lower() == "true",
            "high_confidence_only": request.GET.get("high_confidence_only", "").lower()
            == "true",
        }

        # Get ContentType and build optimized query
        content_type = ContentType.objects.get_for_model(Siddha)
        mappings_query = get_optimized_mappings_query(content_type, concept_id, filters)

        # Get comprehensive statistics with single aggregate query
        mapping_stats = mappings_query.aggregate(
            total_mappings=Count("id"),
            validated_mappings=Count("id", filter=Q(is_validated=True)),
            high_confidence_mappings=Count("id", filter=Q(is_high_confidence=True)),
            needs_review_count=Count("id", filter=Q(needs_review=True)),
            has_issues_count=Count("id", filter=Q(has_issues=True)),
            avg_confidence=Avg("confidence_score"),
            avg_similarity=Avg("similarity_score"),
            max_confidence=Max("confidence_score"),
            min_confidence=Min("confidence_score"),
        )

        # Implement pagination
        total_count = mapping_stats["total_mappings"]
        offset = (page - 1) * page_size
        mappings_page = mappings_query[offset : offset + page_size]

        # Serialize concept and mappings
        concept_serializer = SiddhaConceptSerializer(siddha_concept)
        mappings_serializer = ConceptMappingDetailSerializer(
            mappings_page,
            many=True,
            context={"request": request, "include_embeddings": include_embeddings},
        )

        # Build comprehensive response
        response_data = {
            "concept": concept_serializer.data,
            "mapping_statistics": format_statistics_response(mapping_stats),
            "mappings": {
                "data": mappings_serializer.data,
                "pagination": build_pagination_info(page, page_size, total_count),
            },
            "filters_applied": {
                "min_confidence": filters["min_confidence"],
                "validated_only": filters["validated_only"],
                "high_confidence_only": filters["high_confidence_only"],
                "include_embeddings": include_embeddings,
            },
            "meta": {
                "system": "siddha",
                "concept_id": concept_id,
                "retrieved_at": timezone.now().isoformat(),
                "cache_duration": 600,
                "api_version": "1.0.0",
            },
        }

        return Response(response_data, status=status.HTTP_200_OK)

    except Exception as e:
        logger.error(f"Error in Siddha concept detail view: {str(e)}", exc_info=True)
        return Response(
            {
                "error": "Internal server error",
                "message": "Failed to retrieve Siddha concept details",
                "code": "INTERNAL_ERROR",
                "debug_info": str(e) if settings.DEBUG else None,
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@extend_schema(
    summary="Create Manual NAMASTE to ICD-11 Mapping",
    description="""
    **Manual Mapping Creation API for Physicians**
    
    Allows physicians to manually create mappings between NAMASTE concepts 
    and ICD-11 codes, bypassing AI-generated mappings for expert clinical validation.
    
    ## Key Features
    - Map 1 NAMASTE concept to 1-3 ICD-11 codes
    - Set physician confidence scores and relationship types  
    - Add detailed clinical notes and reasoning
    - Complete audit trail for medical liability
    - Integration with existing mapping infrastructure
    
    ## Request Example
    ```
    {
        "source_system": "ayurveda",
        "source_concept_id": 155,
        "physician_name": "Dr. Rajesh Kumar",
        "physician_id": "AYUSH12345",
        "institution": "AIIMS Delhi",
        "reviewed_ai_suggestions": true,
        "icd11_mappings": [
            {
                "icd11_id": 1234,
                "code": "M54.3",
                "title": "Sciatica",
                "confidence_score": 0.95,
                "relationship": "equivalent-to",
                "notes": "Perfect clinical match - Gridhrasi is exactly sciatica"
            }
        ]
    }
    ```
    """,
    request=ManualMappingCreateSerializer,
    responses={
        201: OpenApiResponse(
            response=MappingCreateResponseSerializer,
            description="Mappings created successfully",
        ),
        400: OpenApiResponse(
            response=ErrorResponseSerializer, description="Invalid request data"
        ),
        409: OpenApiResponse(
            response=ErrorResponseSerializer, description="Mapping conflict"
        ),
        500: OpenApiResponse(
            response=ErrorResponseSerializer, description="Internal server error"
        ),
    },
    tags=["Manual Mapping API"],
    operation_id="createManualMapping",
)
@api_view(["POST"])
@renderer_classes([JSONRenderer])
def create_manual_mapping(request):
    """Create manual NAMASTE to ICD-11 mapping"""

    try:
        # Validate input data
        serializer = ManualMappingCreateSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(
                {
                    "success": False,
                    "message": "Validation failed",
                    "errors": serializer.errors,
                    "error_code": "VALIDATION_ERROR",
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        validated_data = serializer.validated_data

        # Get source concept
        source_system = validated_data["source_system"]
        source_concept_id = validated_data["source_concept_id"]

        model_map = {"ayurveda": Ayurvedha, "siddha": Siddha, "unani": Unani}
        source_model = model_map[source_system]
        source_content_type = ContentType.objects.get_for_model(source_model)

        try:
            source_concept = source_model.objects.get(pk=source_concept_id)
        except source_model.DoesNotExist:
            return Response(
                {
                    "success": False,
                    "message": f"{source_system.title()} concept with ID {source_concept_id} not found",
                    "error_code": "CONCEPT_NOT_FOUND",
                },
                status=status.HTTP_404_NOT_FOUND,
            )

        # Check for existing manual mappings
        existing_manual = ConceptMapping.objects.filter(
            source_content_type=source_content_type,
            source_object_id=source_concept_id,
            mapping_method__icontains="manual",
        ).exists()

        if existing_manual and not request.GET.get("force", False):
            return Response(
                {
                    "success": False,
                    "message": f"{source_system.title()} concept {source_concept_id} already has manual mappings. Use update endpoint or add ?force=true to override.",
                    "error_code": "MAPPING_EXISTS",
                },
                status=status.HTTP_409_CONFLICT,
            )

        created_mappings = []
        created_mapping_details = []
        warnings = []

        # Create mappings in transaction
        with transactio.atomic():
            # Get or create terminology mapping
            terminology_mapping, created = TerminologyMapping.objects.get_or_create(
                name=f"Manual {source_system.title()} to ICD-11",
                defaults={
                    "description": f"Manual physician mapping from {source_system.title()} to ICD-11",
                    "source_system": f"NAMASTE-{source_system.title()}",
                    "target_system": "ICD-11",
                    "biobert_model": "manual_physician_review",
                    "confidence_threshold": 0.0,
                    "similarity_threshold": 0.0,
                    "max_mappings": 5,
                    "is_active": True,
                },
            )

            # Create each mapping
            for idx, icd_mapping in enumerate(validated_data["icd11_mappings"]):
                try:
                    concept_mapping = ConceptMapping.objects.create(
                        mapping=terminology_mapping,
                        source_content_type=source_content_type,
                        source_object_id=source_concept_id,
                        target_concept=None,  # TODO: Link to ICD-11 model
                        relationship=icd_mapping["relationship"],
                        confidence_score=icd_mapping["confidence_score"],
                        similarity_score=icd_mapping["confidence_score"],
                        mapping_method=validated_data["mapping_method"],
                        is_validated=True,
                        is_high_confidence=icd_mapping["confidence_score"] >= 0.8,
                        needs_review=False,
                        has_issues=False,
                        validated_by=validated_data["physician_name"],
                        validated_at=timezone.now(),
                        validation_notes=icd_mapping.get("notes", ""),
                    )

                    # Create audit trail
                    MappingAudit.objects.create(
                        concept_mapping=concept_mapping,
                        action="created",
                        user_name=validated_data["physician_name"],
                        user_id=validated_data["physician_id"],
                        reason="Manual mapping creation by physician",
                        field_changes={
                            "source_system": source_system,
                            "source_concept_id": source_concept_id,
                            "icd11_code": icd_mapping["code"],
                            "icd11_title": icd_mapping["title"],
                            "confidence_score": icd_mapping["confidence_score"],
                            "relationship": icd_mapping["relationship"],
                            "physician_notes": icd_mapping.get("notes", ""),
                            "institution": validated_data.get("institution", ""),
                        },
                        ip_address=request.META.get("REMOTE_ADDR", ""),
                        timestamp=timezone.now(),
                    )

                    created_mappings.append(concept_mapping.id)
                    created_mapping_details.append(
                        {
                            "mapping_id": concept_mapping.id,
                            "icd11_code": icd_mapping["code"],
                            "relationship": icd_mapping["relationship"],
                            "confidence_score": icd_mapping["confidence_score"],
                        }
                    )

                except Exception as e:
                    logger.error(f"Error creating mapping {idx + 1}: {str(e)}")
                    warnings.append(
                        f"Warning creating mapping for {icd_mapping['code']}: {str(e)}"
                    )

        # Prepare response
        concept_info = {
            "system": source_system,
            "id": source_concept_id,
            "code": getattr(source_concept, "code", ""),
            "name": getattr(source_concept, "english_name", ""),
        }

        if source_system == "ayurveda":
            concept_info.update(
                {
                    "hindi_name": getattr(source_concept, "hindi_name", ""),
                    "diacritical_name": getattr(source_concept, "diacritical_name", ""),
                }
            )

        return Response(
            {
                "success": True,
                "message": f"Successfully created {len(created_mappings)} mappings for {source_system} concept {source_concept_id}",
                "mapping_ids": created_mappings,
                "total_created": len(created_mappings),
                "concept_info": concept_info,
                "created_mappings": created_mapping_details,
                "warnings": warnings,
            },
            status=status.HTTP_201_CREATED,
        )

    except Exception as e:
        logger.error(f"Error creating manual mapping: {str(e)}", exc_info=True)
        return Response(
            {
                "success": False,
                "message": "Failed to create manual mapping",
                "error": str(e) if settings.DEBUG else "Internal server error",
                "error_code": "INTERNAL_ERROR",
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@extend_schema(
    summary="Update Existing NAMASTE ↔ ICD-11 Mapping",
    description="""
    **Update Existing Mapping API for Physicians**
    
    Allows physicians to modify existing mappings, whether AI-generated or manually created.
    Essential for correcting mappings and adding expert validation.
    """,
    request=MappingUpdateSerializer,
    responses={
        200: OpenApiResponse(
            response=MappingUpdateResponseSerializer,
            description="Mapping updated successfully",
        ),
        400: OpenApiResponse(
            response=ErrorResponseSerializer, description="Invalid request data"
        ),
        404: OpenApiResponse(
            response=ErrorResponseSerializer, description="Mapping not found"
        ),
        500: OpenApiResponse(
            response=ErrorResponseSerializer, description="Internal server error"
        ),
    },
    tags=["Manual Mapping API"],
    operation_id="updateMapping",
)
@api_view(["PUT"])
@renderer_classes([JSONRenderer])
def update_mapping(request):
    """Update existing NAMASTE to ICD-11 mapping"""

    try:
        # Validate input
        serializer = MappingUpdateSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(
                {
                    "success": False,
                    "message": "Validation failed",
                    "errors": serializer.errors,
                    "error_code": "VALIDATION_ERROR",
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        validated_data = serializer.validated_data

        # Get mapping
        try:
            mapping = ConceptMapping.objects.get(id=validated_data["mapping_id"])
        except ConceptMapping.DoesNotExist:
            return Response(
                {
                    "success": False,
                    "message": f"Mapping with ID {validated_data['mapping_id']} not found",
                    "error_code": "MAPPING_NOT_FOUND",
                },
                status=status.HTTP_404_NOT_FOUND,
            )

        field_changes = {}
        previous_values = {}

        # Update in transaction
        with transaction.atomic():
            # Update fields
            if "confidence_score" in validated_data:
                old_confidence = mapping.confidence_score
                new_confidence = validated_data["confidence_score"]
                mapping.confidence_score = new_confidence
                mapping.similarity_score = new_confidence
                mapping.is_high_confidence = new_confidence >= 0.8
                field_changes["confidence_score"] = {
                    "old": old_confidence,
                    "new": new_confidence,
                }
                previous_values["confidence_score"] = old_confidence

            if "relationship" in validated_data:
                old_relationship = mapping.relationship
                new_relationship = validated_data["relationship"]
                mapping.relationship = new_relationship
                field_changes["relationship"] = {
                    "old": old_relationship,
                    "new": new_relationship,
                }
                previous_values["relationship"] = old_relationship

            if "is_validated" in validated_data:
                old_validated = mapping.is_validated
                new_validated = validated_data["is_validated"]
                mapping.is_validated = new_validated
                if new_validated:
                    mapping.validated_by = validated_data["physician_name"]
                    mapping.validated_at = timezone.now()
                field_changes["is_validated"] = {
                    "old": old_validated,
                    "new": new_validated,
                }
                previous_values["is_validated"] = old_validated

            if "needs_review" in validated_data:
                old_needs_review = mapping.needs_review
                new_needs_review = validated_data["needs_review"]
                mapping.needs_review = new_needs_review
                field_changes["needs_review"] = {
                    "old": old_needs_review,
                    "new": new_needs_review,
                }
                previous_values["needs_review"] = old_needs_review

            if "has_issues" in validated_data:
                old_has_issues = mapping.has_issues
                new_has_issues = validated_data["has_issues"]
                mapping.has_issues = new_has_issues
                field_changes["has_issues"] = {
                    "old": old_has_issues,
                    "new": new_has_issues,
                }
                previous_values["has_issues"] = old_has_issues

            if "notes" in validated_data:
                old_notes = mapping.validation_notes or ""
                new_notes = validated_data["notes"]
                mapping.validation_notes = new_notes
                field_changes["validation_notes"] = {"old": old_notes, "new": new_notes}
                previous_values["validation_notes"] = old_notes

            mapping.updated_at = timezone.now()
            mapping.save()

            # Create audit trail
            MappingAudit.objects.create(
                concept_mapping=mapping,
                action="updated",
                user_name=validated_data["physician_name"],
                user_id=validated_data["physician_id"],
                reason=validated_data["update_reason"],
                field_changes=field_changes,
                ip_address=request.META.get("REMOTE_ADDR", ""),
                timestamp=timezone.now(),
            )

        return Response(
            {
                "success": True,
                "message": f"Successfully updated mapping {mapping.id}",
                "mapping_id": mapping.id,
                "changes_made": list(field_changes.keys()),
                "previous_values": previous_values,
            },
            status=status.HTTP_200_OK,
        )

    except Exception as e:
        logger.error(f"Error updating mapping: {str(e)}", exc_info=True)
        return Response(
            {
                "success": False,
                "message": "Failed to update mapping",
                "error": str(e) if settings.DEBUG else "Internal server error",
                "error_code": "INTERNAL_ERROR",
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@extend_schema(
    summary="Search ICD-11 Codes for Mapping",
    description="""**ICD-11 Code Search API for Physicians**""",
    parameters=[
        OpenApiParameter(
            name="q",
            type=OpenApiTypes.STR,
            location=OpenApiParameter.QUERY,
            description="Search query",
            required=True,
        ),
        OpenApiParameter(
            name="limit",
            type=OpenApiTypes.INT,
            location=OpenApiParameter.QUERY,
            description="Max results (default: 20)",
        ),
    ],
    responses={
        200: OpenApiResponse(
            response=ICD11SearchResponseSerializer, description="Search results"
        )
    },
    tags=["Manual Mapping API"],
    operation_id="searchICD11",
)
@api_view(["GET"])
@renderer_classes([JSONRenderer])
def search_icd11_codes(request):
    """Search ICD-11 codes for manual mapping assistance"""

    try:
        query = request.GET.get("q", "").strip()
        if not query:
            return Response(
                {
                    "success": False,
                    "message": "Search query 'q' parameter is required",
                    "error_code": "MISSING_QUERY",
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        if len(query) < 2:
            return Response(
                {
                    "success": False,
                    "message": "Search query must be at least 2 characters long",
                    "error_code": "QUERY_TOO_SHORT",
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        limit = min(int(request.GET.get("limit", 20)), 100)

        # TODO: Replace with actual ICD-11 database search
        example_results = [
            {
                "id": 1234,
                "code": "M54.3",
                "title": "Sciatica",
                "definition": "Pain in the distribution of the sciatic nerve",
                "chapter": "13 - Diseases of the musculoskeletal system",
                "category": "Dorsopathies",
                "synonyms": ["sciatic neuralgia", "sciatic nerve pain"],
            },
            {
                "id": 1235,
                "code": "M54.1",
                "title": "Radiculopathy",
                "definition": "Disease of nerve roots",
                "chapter": "13 - Diseases of the musculoskeletal system",
                "category": "Dorsopathies",
                "synonyms": ["nerve root disorder", "radicular pain"],
            },
        ]

        return Response(
            {
                "query": query,
                "total_results": len(example_results),
                "results": example_results[:limit],
                "suggestions": [
                    f"{query} pain",
                    f"{query} disorder",
                    f"chronic {query}",
                ],
            },
            status=status.HTTP_200_OK,
        )

    except Exception as e:
        logger.error(f"Error searching ICD-11: {str(e)}", exc_info=True)
        return Response(
            {
                "success": False,
                "message": "ICD-11 search failed",
                "error": str(e) if settings.DEBUG else "Internal server error",
                "error_code": "SEARCH_ERROR",
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
