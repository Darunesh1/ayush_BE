"""
serializers.py - Complete Django REST Framework Serializers with Safe Object Access
NAMASTE Concept Detail Views for Ayurveda, Siddha, and Unani with ICD-11 Mappings

Optimized for TinyBioBERT + ONNX + PostgreSQL + Django setup
Includes all serializers for Swagger documentation
Fixed: 'list' object has no attribute 'pk' errors with safe object access
"""

import logging

from django.contrib.contenttypes.models import ContentType
from django.db.models import Avg, Count, Q
from rest_framework import serializers

from terminologies.models import Ayurvedha, ICD11Term, Siddha, TermMapping, Unani

# Import your models - adjust paths according to your project structure
from .models import ConceptMapping, MappingAudit, TerminologyMapping

logger = logging.getLogger(__name__)


class BaseNAMASTEConceptSerializer(serializers.ModelSerializer):
    """
    Base serializer for common NAMASTE concept fields
    Contains optimized field selection for all systems with TinyBioBERT metadata
    """

    # TinyBioBERT embedding metadata
    has_embedding = serializers.SerializerMethodField()
    embedding_info = serializers.SerializerMethodField()

    class Meta:
        abstract = True
        fields = [
            "id",
            "code",
            "english_name",
            "description",
            "has_embedding",
            "embedding_info",
        ]

    def get_has_embedding(self, obj):
        """Check if concept has valid TinyBioBERT embedding (768 dimensions)"""
        try:
            embedding = getattr(obj, "embedding", None)
            return bool(
                embedding and isinstance(embedding, list) and len(embedding) == 768
            )
        except Exception as e:
            logger.error(f"Error checking embedding: {str(e)}")
            return False

    def get_embedding_info(self, obj):
        """Get embedding metadata without the actual 768-dimensional vector"""
        try:
            embedding = getattr(obj, "embedding", None)
            if not embedding:
                return None

            return {
                "model_version": getattr(obj, "embedding_model_version", None)
                or "unknown",
                "updated_at": getattr(obj, "embedding_updated_at", None),
                "dimensions": len(embedding) if isinstance(embedding, list) else 0,
                "has_valid_embedding": len(embedding) == 768
                if isinstance(embedding, list)
                else False,
            }
        except Exception as e:
            logger.error(f"Error getting embedding info: {str(e)}")
            return None


class AyurvedaConceptSerializer(BaseNAMASTEConceptSerializer):
    """Optimized serializer for Ayurveda concepts with hindi and diacritical names"""

    class Meta:
        model = Ayurvedha
        fields = BaseNAMASTEConceptSerializer.Meta.fields + [
            "hindi_name",
            "diacritical_name",
        ]


class SiddhaConceptSerializer(BaseNAMASTEConceptSerializer):
    """Optimized serializer for Siddha concepts with tamil names and references"""

    class Meta:
        model = Siddha
        fields = BaseNAMASTEConceptSerializer.Meta.fields + [
            "tamil_name",
            "romanized_name",
            "reference",
        ]


class UnaniConceptSerializer(BaseNAMASTEConceptSerializer):
    """Optimized serializer for Unani concepts with arabic names and references"""

    class Meta:
        model = Unani
        fields = BaseNAMASTEConceptSerializer.Meta.fields + [
            "arabic_name",
            "romanized_name",
            "reference",
        ]


class ConceptMappingDetailSerializer(serializers.ModelSerializer):
    """
    Comprehensive serializer for ConceptMapping with ICD-11 details
    Includes TinyBioBERT confidence scores, validation status, and audit trails
    Conditionally includes 768-dimensional embeddings for ONNX processing

    FIXED: All methods now use safe object access to prevent 'list' object errors
    """

    # Computed fields from model methods
    source_code = serializers.SerializerMethodField()
    source_display = serializers.SerializerMethodField()
    target_code = serializers.SerializerMethodField()
    target_display = serializers.SerializerMethodField()
    source_system_display = serializers.SerializerMethodField()

    # Quality metrics from model properties
    confidence_category = serializers.SerializerMethodField()
    quality_score = serializers.SerializerMethodField()

    # Enhanced ICD-11 target details
    icd11_details = serializers.SerializerMethodField()

    # Validation information
    validation_info = serializers.SerializerMethodField()

    # TinyBioBERT embeddings (conditional)
    source_embedding = serializers.SerializerMethodField()
    target_embedding = serializers.SerializerMethodField()

    # Audit trail (recent entries only)
    recent_audits = serializers.SerializerMethodField()

    class Meta:
        model = ConceptMapping
        fields = [
            "id",
            "source_code",
            "source_display",
            "source_system_display",
            "target_code",
            "target_display",
            "icd11_details",
            "relationship",
            "similarity_score",
            "confidence_score",
            "confidence_category",
            "quality_score",
            "mapping_method",
            "source_embedding",
            "target_embedding",
            "is_validated",
            "is_high_confidence",
            "needs_review",
            "has_issues",
            "validation_info",
            "recent_audits",
            "created_at",
            "updated_at",
        ]

    def get_source_code(self, obj):
        """Get source concept code with safe object access"""
        try:
            if hasattr(obj, "get_source_code"):
                return obj.get_source_code()

            # Fallback implementation with safe access
            if getattr(obj, "source_content_type", None) and getattr(
                obj, "source_object_id", None
            ):
                try:
                    source_obj = obj.source_content_type.get_object_for_this_type(
                        pk=obj.source_object_id
                    )
                    return getattr(source_obj, "code", "")
                except Exception as e:
                    logger.warning(f"Error getting source object: {str(e)}")
                    return ""
            return ""
        except Exception as e:
            logger.error(f"Error in get_source_code: {str(e)}")
            return ""

    def get_source_display(self, obj):
        """Get source concept display name with safe object access"""
        try:
            if hasattr(obj, "get_source_display"):
                return obj.get_source_display()

            # Fallback implementation with safe access
            if getattr(obj, "source_content_type", None) and getattr(
                obj, "source_object_id", None
            ):
                try:
                    source_obj = obj.source_content_type.get_object_for_this_type(
                        pk=obj.source_object_id
                    )
                    return getattr(source_obj, "english_name", "")
                except Exception as e:
                    logger.warning(f"Error getting source object: {str(e)}")
                    return ""
            return ""
        except Exception as e:
            logger.error(f"Error in get_source_display: {str(e)}")
            return ""

    def get_target_code(self, obj):
        """Get ICD-11 target code with safe object access"""
        try:
            if hasattr(obj, "get_target_code"):
                return obj.get_target_code()

            target_concept = getattr(obj, "target_concept", None)
            if target_concept:
                # Handle case where target_concept might be a list
                if isinstance(target_concept, list):
                    target_concept = target_concept[0] if target_concept else None

                if target_concept:
                    return getattr(target_concept, "code", "")
            return ""
        except Exception as e:
            logger.error(f"Error in get_target_code: {str(e)}")
            return ""

    def get_target_display(self, obj):
        """Get ICD-11 target display name with safe object access"""
        try:
            if hasattr(obj, "get_target_display"):
                return obj.get_target_display()

            target_concept = getattr(obj, "target_concept", None)
            if target_concept:
                # Handle case where target_concept might be a list
                if isinstance(target_concept, list):
                    target_concept = target_concept[0] if target_concept else None

                if target_concept:
                    return getattr(target_concept, "title", "")
            return ""
        except Exception as e:
            logger.error(f"Error in get_target_display: {str(e)}")
            return ""

    def get_source_system_display(self, obj):
        """Get source system display name with safe object access"""
        try:
            if hasattr(obj, "get_source_system_display"):
                return obj.get_source_system_display()

            source_content_type = getattr(obj, "source_content_type", None)
            if source_content_type:
                model_name = getattr(source_content_type, "model", "").lower()
                if model_name == "ayurvedha":
                    return "Ayurveda"
                elif model_name == "siddha":
                    return "Siddha"
                elif model_name == "unani":
                    return "Unani"
            return ""
        except Exception as e:
            logger.error(f"Error in get_source_system_display: {str(e)}")
            return ""

    def get_confidence_category(self, obj):
        """Get confidence category based on TinyBioBERT score with safe access"""
        try:
            if hasattr(obj, "confidence_category"):
                return obj.confidence_category

            confidence_score = getattr(obj, "confidence_score", 0.0) or 0.0
            if confidence_score >= 0.95:
                return "very_high"
            elif confidence_score >= 0.85:
                return "high"
            elif confidence_score >= 0.75:
                return "medium"
            else:
                return "low"
        except Exception as e:
            logger.error(f"Error in get_confidence_category: {str(e)}")
            return "low"

    def get_quality_score(self, obj):
        """Get overall quality score with safe access"""
        try:
            if hasattr(obj, "quality_score"):
                return obj.quality_score

            confidence = getattr(obj, "confidence_score", None) or 0.0
            similarity = getattr(obj, "similarity_score", None) or 0.0
            return round((confidence + similarity) / 2, 3)
        except Exception as e:
            logger.error(f"Error in get_quality_score: {str(e)}")
            return 0.0

    def get_icd11_details(self, obj):
        """Get comprehensive ICD-11 target details with safe object access"""
        try:
            target_concept = getattr(obj, "target_concept", None)
            if not target_concept:
                return None

            # Handle case where target_concept might be a list
            if isinstance(target_concept, list):
                logger.warning(f"target_concept is a list: {target_concept}")
                if target_concept:
                    target_concept = target_concept[0]
                else:
                    return None

            # Safe access to target object attributes
            if not hasattr(target_concept, "pk"):
                logger.warning(
                    f"target_concept has no pk attribute: {type(target_concept)}"
                )
                return None

            return {
                "id": getattr(target_concept, "pk", None),
                "code": getattr(target_concept, "code", ""),
                "title": getattr(target_concept, "title", ""),
                "definition": getattr(target_concept, "definition", ""),
                "chapter": getattr(target_concept, "chapter", ""),
                "block_id": getattr(target_concept, "block_id", ""),
                "category": getattr(target_concept, "category", ""),
                "hierarchy_level": getattr(target_concept, "hierarchy_level", 0),
                "is_leaf": getattr(target_concept, "is_leaf", True),
                "synonyms": getattr(target_concept, "synonyms", []),
                "url": f"/icd11/{getattr(target_concept, 'code', '')}/"
                if getattr(target_concept, "code", "")
                else None,
                "who_foundation_url": getattr(
                    target_concept, "who_foundation_url", None
                ),
            }
        except Exception as e:
            logger.error(f"Error in get_icd11_details: {str(e)}", exc_info=True)
            return None

    def get_validation_info(self, obj):
        """Get expert validation details with safe access"""
        try:
            if not getattr(obj, "is_validated", False):
                return None

            return {
                "validated_by": getattr(obj, "validated_by", None),
                "validated_at": getattr(obj, "validated_at", None),
                "validation_notes": getattr(obj, "validation_notes", ""),
                "validation_method": "expert_review",
                "is_reliable": getattr(obj, "is_validated", False)
                and not getattr(obj, "has_issues", False),
            }
        except Exception as e:
            logger.error(f"Error in get_validation_info: {str(e)}")
            return None

    def get_source_embedding(self, obj):
        """Conditionally include TinyBioBERT source embedding (768-dimensional) with safe access"""
        try:
            context = self.context or {}
            if context.get("include_embeddings", False):
                embedding = getattr(obj, "source_embedding", None)
                # Ensure embedding is a list and has correct dimensions
                if isinstance(embedding, list) and len(embedding) == 768:
                    return embedding
            return None
        except Exception as e:
            logger.error(f"Error in get_source_embedding: {str(e)}")
            return None

    def get_target_embedding(self, obj):
        """Conditionally include TinyBioBERT target embedding (768-dimensional) with safe access"""
        try:
            context = self.context or {}
            if context.get("include_embeddings", False):
                embedding = getattr(obj, "target_embedding", None)
                # Ensure embedding is a list and has correct dimensions
                if isinstance(embedding, list) and len(embedding) == 768:
                    return embedding
            return None
        except Exception as e:
            logger.error(f"Error in get_target_embedding: {str(e)}")
            return None

    def get_recent_audits(self, obj):
        """Get recent audit entries with safe access"""
        try:
            # Check if audit_entries exists and is accessible
            if hasattr(obj, "recent_audit_entries"):
                audits = []
                audit_entries = getattr(obj, "recent_audit_entries", [])

                # Handle case where audit_entries might be a list or queryset
                if hasattr(audit_entries, "__iter__"):
                    for audit in audit_entries:
                        if hasattr(audit, "pk"):  # Ensure it's a proper model instance
                            audits.append(
                                {
                                    "id": str(getattr(audit, "id", "")),
                                    "action": getattr(audit, "action", ""),
                                    "timestamp": getattr(audit, "timestamp", None),
                                    "user_name": getattr(audit, "user_name", ""),
                                    "reason": getattr(audit, "reason", ""),
                                    "field_changes": getattr(
                                        audit, "field_changes", {}
                                    ),
                                }
                            )
                return audits
            return []
        except Exception as e:
            logger.error(f"Error in get_recent_audits: {str(e)}")
            return []


# =============================================================================
# ADDITIONAL SERIALIZERS FOR SWAGGER DOCUMENTATION AND RESPONSES
# =============================================================================


class MappingStatisticsSerializer(serializers.Serializer):
    """Serializer for mapping statistics and quality metrics"""

    total_mappings = serializers.IntegerField(help_text="Total number of mappings")
    validated_mappings = serializers.IntegerField(
        help_text="Number of expert-validated mappings"
    )
    high_confidence_mappings = serializers.IntegerField(
        help_text="Number of high-confidence mappings (>= 0.85)"
    )
    needs_review = serializers.IntegerField(
        help_text="Number of mappings that need review"
    )
    has_issues = serializers.IntegerField(
        help_text="Number of mappings with reported issues"
    )
    rates = serializers.DictField(
        help_text="Validation and quality rates as percentages"
    )
    quality_metrics = serializers.DictField(
        help_text="Average confidence and similarity scores from TinyBioBERT"
    )


class PaginationInfoSerializer(serializers.Serializer):
    """Serializer for pagination metadata"""

    page = serializers.IntegerField(help_text="Current page number")
    page_size = serializers.IntegerField(help_text="Number of items per page")
    total_pages = serializers.IntegerField(help_text="Total number of pages")
    total_count = serializers.IntegerField(help_text="Total number of items")
    has_next = serializers.BooleanField(help_text="Whether there is a next page")
    has_previous = serializers.BooleanField(
        help_text="Whether there is a previous page"
    )
    next_page = serializers.IntegerField(
        allow_null=True, help_text="Next page number (null if no next page)"
    )
    previous_page = serializers.IntegerField(
        allow_null=True, help_text="Previous page number (null if no previous page)"
    )


class MappingsResponseSerializer(serializers.Serializer):
    """Serializer for mappings section of the response"""

    data = ConceptMappingDetailSerializer(
        many=True, help_text="Array of concept mappings with ICD-11 details"
    )
    pagination = PaginationInfoSerializer(help_text="Pagination metadata")


class FiltersAppliedSerializer(serializers.Serializer):
    """Serializer for applied filters"""

    min_confidence = serializers.CharField(
        allow_null=True, help_text="Minimum confidence filter applied"
    )
    validated_only = serializers.BooleanField(
        help_text="Whether only validated mappings are shown"
    )
    high_confidence_only = serializers.BooleanField(
        help_text="Whether only high-confidence mappings are shown"
    )
    include_embeddings = serializers.BooleanField(
        help_text="Whether TinyBioBERT embeddings are included"
    )


class MetaInfoSerializer(serializers.Serializer):
    """Serializer for metadata information"""

    system = serializers.CharField(
        help_text="NAMASTE system type (ayurveda, siddha, unani)"
    )
    concept_id = serializers.IntegerField(help_text="ID of the concept")
    retrieved_at = serializers.DateTimeField(
        help_text="Timestamp when data was retrieved"
    )
    cache_duration = serializers.IntegerField(help_text="Cache duration in seconds")
    api_version = serializers.CharField(help_text="API version")


class ErrorResponseSerializer(serializers.Serializer):
    """Serializer for error responses in Swagger documentation"""

    error = serializers.CharField(help_text="Error type")
    message = serializers.CharField(help_text="Human-readable error message")
    code = serializers.CharField(
        help_text="Machine-readable error code", required=False
    )
    debug_info = serializers.CharField(
        help_text="Debug information (only in DEBUG mode)", required=False
    )


class ConceptDetailResponseSerializer(serializers.Serializer):
    """Complete response serializer for concept detail endpoints"""

    concept = serializers.DictField(
        help_text="Complete concept information with system-specific fields"
    )
    mapping_statistics = MappingStatisticsSerializer(
        help_text="Comprehensive mapping statistics and quality metrics"
    )
    mappings = MappingsResponseSerializer(
        help_text="Paginated mappings with ICD-11 details"
    )
    filters_applied = FiltersAppliedSerializer(
        help_text="Filters that were applied to the request"
    )
    meta = MetaInfoSerializer(help_text="Metadata about the request and response")


# =============================================================================
# LIGHTWEIGHT SERIALIZERS FOR SUMMARY ENDPOINTS
# =============================================================================


class ConceptSummarySerializer(serializers.Serializer):
    """Lightweight serializer for concept summaries used in search results"""

    id = serializers.IntegerField()
    code = serializers.CharField()
    english_name = serializers.CharField()
    system = serializers.CharField()
    hindi_name = serializers.CharField(required=False)  # Ayurveda
    diacritical_name = serializers.CharField(required=False)  # Ayurveda
    tamil_name = serializers.CharField(required=False)  # Siddha
    arabic_name = serializers.CharField(required=False)  # Unani
    romanized_name = serializers.CharField(required=False)  # Siddha, Unani


class MappingSummarySerializer(serializers.Serializer):
    """Lightweight mapping summary for quick previews"""

    total_mappings = serializers.IntegerField()
    validated_mappings = serializers.IntegerField()
    high_confidence_mappings = serializers.IntegerField()


class ConceptSummaryResponseSerializer(serializers.Serializer):
    """Response serializer for concept summary endpoints"""

    concept = ConceptSummarySerializer()
    mapping_summary = MappingSummarySerializer()


# =============================================================================
# SPECIALIZED SERIALIZERS FOR DIFFERENT ENDPOINTS
# =============================================================================


class AyurvedaDetailResponseSerializer(ConceptDetailResponseSerializer):
    """Specialized response serializer for Ayurveda concept details"""

    concept = AyurvedaConceptSerializer(
        help_text="Complete Ayurveda concept with Hindi and diacritical names"
    )


class SiddhaDetailResponseSerializer(ConceptDetailResponseSerializer):
    """Specialized response serializer for Siddha concept details"""

    concept = SiddhaConceptSerializer(
        help_text="Complete Siddha concept with Tamil names and references"
    )


class UnaniDetailResponseSerializer(ConceptDetailResponseSerializer):
    """Specialized response serializer for Unani concept details"""

    concept = UnaniConceptSerializer(
        help_text="Complete Unani concept with Arabic names and references"
    )


# =============================================================================
# VALIDATION SERIALIZERS
# =============================================================================


class ConceptIDValidator(serializers.Serializer):
    """Validator for concept ID parameters"""

    concept_id = serializers.IntegerField(
        min_value=1, help_text="Valid concept ID (positive integer)"
    )


class QueryParametersSerializer(serializers.Serializer):
    """Serializer for validating query parameters"""

    page = serializers.IntegerField(
        min_value=1, default=1, help_text="Page number (minimum 1)"
    )
    page_size = serializers.IntegerField(
        min_value=1, max_value=100, default=20, help_text="Page size (1-100)"
    )
    min_confidence = serializers.FloatField(
        min_value=0.0,
        max_value=1.0,
        required=False,
        help_text="Minimum confidence (0.0-1.0)",
    )
    validated_only = serializers.BooleanField(
        default=False, help_text="Show only validated mappings"
    )
    high_confidence_only = serializers.BooleanField(
        default=False, help_text="Show only high-confidence mappings"
    )
    include_embeddings = serializers.BooleanField(
        default=False, help_text="Include TinyBioBERT embeddings"
    )


# =============================================================================
# AUDIT AND HISTORY SERIALIZERS
# =============================================================================


class MappingAuditSerializer(serializers.ModelSerializer):
    """Serializer for mapping audit entries"""

    class Meta:
        model = MappingAudit
        fields = [
            "id",
            "action",
            "timestamp",
            "user_name",
            "reason",
            "field_changes",
            "ip_address",
        ]
        read_only_fields = fields


class TerminologyMappingSerializer(serializers.ModelSerializer):
    """Serializer for terminology mapping entries"""

    class Meta:
        model = TerminologyMapping
        fields = ["id", "created_at", "updated_at", "is_active"]
        read_only_fields = fields


class ICD11ConceptInputSerializer(serializers.Serializer):
    """Serializer for ICD-11 concept selection in manual mapping"""

    icd11_id = serializers.IntegerField(
        help_text="ICD-11 concept primary key from your ICD-11 database"
    )
    code = serializers.CharField(
        max_length=20, help_text="ICD-11 code (e.g., M54.3, 8B26.5)"
    )
    title = serializers.CharField(max_length=500, help_text="ICD-11 title/description")
    confidence_score = serializers.FloatField(
        min_value=0.0,
        max_value=1.0,
        help_text="Physician's confidence in this mapping (0.0-1.0)",
    )
    relationship = serializers.ChoiceField(
        choices=[
            ("equivalent-to", "Equivalent to"),
            ("broader-than", "Broader than"),
            ("narrower-than", "Narrower than"),
            ("related-to", "Related to"),
            ("not-mappable", "Not mappable"),
        ],
        help_text="Type of relationship between NAMASTE and ICD-11 concepts",
    )
    notes = serializers.CharField(
        max_length=1000,
        required=False,
        allow_blank=True,
        help_text="Physician's clinical notes about this specific mapping",
    )

    def validate_code(self, value):
        """Validate ICD-11 code format"""
        if not value or len(value.strip()) == 0:
            raise serializers.ValidationError("ICD-11 code cannot be empty")
        return value.strip().upper()


class ManualMappingCreateSerializer(serializers.Serializer):
    """Serializer for creating manual NAMASTE to ICD-11 mappings"""

    # Source NAMASTE concept
    source_system = serializers.ChoiceField(
        choices=[("ayurveda", "Ayurveda"), ("siddha", "Siddha"), ("unani", "Unani")],
        help_text="NAMASTE system type",
    )
    source_concept_id = serializers.IntegerField(
        min_value=1, help_text="NAMASTE concept ID (must exist in database)"
    )

    # ICD-11 mappings (1-3 options)
    icd11_mappings = ICD11ConceptInputSerializer(
        many=True,
        min_length=1,
        max_length=3,
        help_text="1-3 ICD-11 mapping options with physician assessment",
    )

    # Physician information (required for audit trail)
    physician_name = serializers.CharField(
        max_length=200, help_text="Full name of the physician creating the mapping"
    )
    physician_id = serializers.CharField(
        max_length=100, help_text="Physician ID, license number, or institutional ID"
    )
    institution = serializers.CharField(
        max_length=300,
        required=False,
        allow_blank=True,
        help_text="Institution, hospital, or clinic name",
    )

    # Mapping process metadata
    mapping_method = serializers.CharField(
        default="manual_physician_review",
        max_length=100,
        help_text="Method used for mapping (default: manual_physician_review)",
    )
    reviewed_ai_suggestions = serializers.BooleanField(
        default=False,
        help_text="Whether physician reviewed existing AI suggestions before mapping",
    )
    general_notes = serializers.CharField(
        max_length=2000,
        required=False,
        allow_blank=True,
        help_text="General notes about the mapping process or concept",
    )

    def validate(self, data):
        """Validate the complete mapping data"""
        # Validate source concept exists
        source_system = data["source_system"]
        source_concept_id = data["source_concept_id"]

        model_map = {"ayurveda": Ayurvedha, "siddha": Siddha, "unani": Unani}

        model_class = model_map.get(source_system)
        if not model_class.objects.filter(pk=source_concept_id).exists():
            raise serializers.ValidationError(
                f"{source_system.title()} concept with ID {source_concept_id} does not exist"
            )

        # Validate physician information
        if not data.get("physician_name", "").strip():
            raise serializers.ValidationError("Physician name is required")

        if not data.get("physician_id", "").strip():
            raise serializers.ValidationError("Physician ID is required")

        # Validate ICD-11 mappings
        icd_codes = []
        for mapping in data["icd11_mappings"]:
            code = mapping["code"]
            if code in icd_codes:
                raise serializers.ValidationError(
                    f"Duplicate ICD-11 code: {code}. Each code can only be mapped once."
                )
            icd_codes.append(code)

            # Validate confidence score makes sense for relationship
            confidence = mapping["confidence_score"]
            relationship = mapping["relationship"]

            if relationship == "not-mappable" and confidence > 0.5:
                raise serializers.ValidationError(
                    f"Confidence score too high ({confidence}) for 'not-mappable' relationship"
                )

        return data


class MappingUpdateSerializer(serializers.Serializer):
    """Serializer for updating existing mappings"""

    mapping_id = serializers.UUIDField(help_text="UUID of the mapping to update")

    # Updated mapping details (all optional)
    confidence_score = serializers.FloatField(
        min_value=0.0,
        max_value=1.0,
        required=False,
        help_text="Updated physician confidence score",
    )
    relationship = serializers.ChoiceField(
        choices=[
            ("equivalent-to", "Equivalent to"),
            ("broader-than", "Broader than"),
            ("narrower-than", "Narrower than"),
            ("related-to", "Related to"),
            ("not-mappable", "Not mappable"),
        ],
        required=False,
        help_text="Updated relationship type",
    )
    is_validated = serializers.BooleanField(
        required=False,
        help_text="Mark as validated by physician (True) or remove validation (False)",
    )
    needs_review = serializers.BooleanField(
        required=False, help_text="Mark as needing additional review"
    )
    has_issues = serializers.BooleanField(
        required=False, help_text="Mark as having clinical or technical issues"
    )

    # Physician information (required for audit trail)
    physician_name = serializers.CharField(
        max_length=200, help_text="Name of physician making the update"
    )
    physician_id = serializers.CharField(
        max_length=100, help_text="Physician ID or license number"
    )

    # Update reason (required for audit trail)
    update_reason = serializers.CharField(
        max_length=1000,
        help_text="Clinical reason for the update (required for audit trail)",
    )
    notes = serializers.CharField(
        max_length=2000,
        required=False,
        allow_blank=True,
        help_text="Additional clinical notes about the update",
    )

    def validate(self, data):
        """Validate the update data"""
        # Ensure at least one field is being updated
        update_fields = [
            "confidence_score",
            "relationship",
            "is_validated",
            "needs_review",
            "has_issues",
            "notes",
        ]

        if not any(field in data for field in update_fields):
            raise serializers.ValidationError(
                "At least one field must be provided for update"
            )

        # Validate required audit fields
        if not data.get("update_reason", "").strip():
            raise serializers.ValidationError(
                "Update reason is required for audit trail"
            )

        if not data.get("physician_name", "").strip():
            raise serializers.ValidationError("Physician name is required")

        if not data.get("physician_id", "").strip():
            raise serializers.ValidationError("Physician ID is required")

        return data


# =============================================================================
# RESPONSE SERIALIZERS
# =============================================================================


class CreatedMappingSerializer(serializers.Serializer):
    """Serializer for individual created mapping info"""

    mapping_id = serializers.UUIDField(help_text="UUID of created mapping")
    icd11_code = serializers.CharField(help_text="ICD-11 code that was mapped")
    relationship = serializers.CharField(help_text="Relationship type assigned")
    confidence_score = serializers.FloatField(help_text="Physician confidence score")


class ConceptInfoSerializer(serializers.Serializer):
    """Serializer for source concept information"""

    system = serializers.CharField(help_text="NAMASTE system (ayurveda/siddha/unani)")
    id = serializers.IntegerField(help_text="Concept ID")
    code = serializers.CharField(help_text="NAMASTE concept code")
    name = serializers.CharField(help_text="English name of concept")
    hindi_name = serializers.CharField(
        required=False, help_text="Hindi/Tamil/Arabic name"
    )
    diacritical_name = serializers.CharField(
        required=False, help_text="Diacritical name"
    )


class MappingCreateResponseSerializer(serializers.Serializer):
    """Response serializer for manual mapping creation"""

    success = serializers.BooleanField(help_text="Whether operation was successful")
    message = serializers.CharField(help_text="Human-readable response message")
    mapping_ids = serializers.ListField(
        child=serializers.UUIDField(), help_text="List of created mapping UUIDs"
    )
    total_created = serializers.IntegerField(help_text="Number of mappings created")
    concept_info = ConceptInfoSerializer(
        help_text="Information about the mapped concept"
    )
    created_mappings = CreatedMappingSerializer(
        many=True, help_text="Details of each created mapping"
    )
    warnings = serializers.ListField(
        child=serializers.CharField(),
        required=False,
        help_text="Any warnings during processing",
    )


class MappingUpdateResponseSerializer(serializers.Serializer):
    """Response serializer for mapping updates"""

    success = serializers.BooleanField(help_text="Whether update was successful")
    message = serializers.CharField(help_text="Response message")
    mapping_id = serializers.UUIDField(help_text="UUID of updated mapping")
    changes_made = serializers.ListField(
        child=serializers.CharField(), help_text="List of fields that were changed"
    )
    previous_values = serializers.DictField(
        required=False, help_text="Previous values of changed fields"
    )


class ICD11SearchResultSerializer(serializers.Serializer):
    """Serializer for ICD-11 search results"""

    id = serializers.IntegerField(help_text="ICD-11 concept ID")
    code = serializers.CharField(help_text="ICD-11 code")
    title = serializers.CharField(help_text="ICD-11 title")
    definition = serializers.CharField(required=False, help_text="ICD-11 definition")
    chapter = serializers.CharField(required=False, help_text="ICD-11 chapter")
    category = serializers.CharField(required=False, help_text="ICD-11 category")
    synonyms = serializers.ListField(
        child=serializers.CharField(), required=False, help_text="Alternative terms"
    )


class ICD11SearchResponseSerializer(serializers.Serializer):
    """Response serializer for ICD-11 search"""

    query = serializers.CharField(help_text="Search query used")
    total_results = serializers.IntegerField(help_text="Total number of results")
    results = ICD11SearchResultSerializer(many=True, help_text="Search results")
    suggestions = serializers.ListField(
        child=serializers.CharField(), required=False, help_text="Search suggestions"
    )


class ICD11ConceptDetailSerializer(serializers.ModelSerializer):
    """
    Comprehensive serializer for ICD-11 concepts with all related NAMASTE mappings
    CORRECTED to use actual model names from your models.py
    """

    # Related NAMASTE concepts
    related_ayurveda = serializers.SerializerMethodField()
    related_siddha = serializers.SerializerMethodField()
    related_unani = serializers.SerializerMethodField()

    # Mapping statistics
    namaste_mapping_summary = serializers.SerializerMethodField()

    class Meta:
        model = ICD11Term  # CORRECTED: Using actual model name
        fields = [
            "id",
            "code",
            "title",
            "definition",
            "long_definition",
            "foundation_uri",
            "browser_url",
            "class_kind",
            "index_terms",
            "parent",
            "inclusions",
            "exclusions",
            "related_ayurveda",
            "related_siddha",
            "related_unani",
            "namaste_mapping_summary",
        ]

    def get_related_ayurveda(self, obj):
        """Get all related Ayurveda concepts with mapping details"""
        try:
            # Using TermMapping model with correct field names
            mappings = TermMapping.objects.filter(
                icd_term=obj, source_system="ayurveda"
            ).select_related("primary_ayurveda_term")

            if not mappings.exists():
                return None

            related_concepts = []
            for mapping in mappings:
                try:
                    ayurveda_term = mapping.primary_ayurveda_term
                    if ayurveda_term:
                        related_concepts.append(
                            {
                                "concept": AyurvedaConceptSerializer(
                                    ayurveda_term
                                ).data,
                                "mapping": {
                                    "id": mapping.id,
                                    "confidence_score": mapping.confidence_score,
                                    "icd_similarity": mapping.icd_similarity,
                                    "source_system": mapping.source_system,
                                    "created_at": mapping.created_at,
                                },
                            }
                        )
                except Exception as e:
                    logger.warning(f"Error getting Ayurveda concept: {str(e)}")
                    continue

            return related_concepts if related_concepts else None

        except Exception as e:
            logger.error(f"Error in get_related_ayurveda: {str(e)}")
            return None

    def get_related_siddha(self, obj):
        """Get all related Siddha concepts with mapping details"""
        try:
            mappings = TermMapping.objects.filter(
                icd_term=obj, source_system="siddha"
            ).select_related("primary_siddha_term")

            if not mappings.exists():
                return None

            related_concepts = []
            for mapping in mappings:
                try:
                    siddha_term = mapping.primary_siddha_term
                    if siddha_term:
                        related_concepts.append(
                            {
                                "concept": SiddhaConceptSerializer(siddha_term).data,
                                "mapping": {
                                    "id": mapping.id,
                                    "confidence_score": mapping.confidence_score,
                                    "icd_similarity": mapping.icd_similarity,
                                    "source_system": mapping.source_system,
                                    "created_at": mapping.created_at,
                                },
                            }
                        )
                except Exception as e:
                    logger.warning(f"Error getting Siddha concept: {str(e)}")
                    continue

            return related_concepts if related_concepts else None

        except Exception as e:
            logger.error(f"Error in get_related_siddha: {str(e)}")
            return None

    def get_related_unani(self, obj):
        """Get all related Unani concepts with mapping details"""
        try:
            mappings = TermMapping.objects.filter(
                icd_term=obj, source_system="unani"
            ).select_related("primary_unani_term")

            if not mappings.exists():
                return None

            related_concepts = []
            for mapping in mappings:
                try:
                    unani_term = mapping.primary_unani_term
                    if unani_term:
                        related_concepts.append(
                            {
                                "concept": UnaniConceptSerializer(unani_term).data,
                                "mapping": {
                                    "id": mapping.id,
                                    "confidence_score": mapping.confidence_score,
                                    "icd_similarity": mapping.icd_similarity,
                                    "source_system": mapping.source_system,
                                    "created_at": mapping.created_at,
                                },
                            }
                        )
                except Exception as e:
                    logger.warning(f"Error getting Unani concept: {str(e)}")
                    continue

            return related_concepts if related_concepts else None

        except Exception as e:
            logger.error(f"Error in get_related_unani: {str(e)}")
            return None

    def get_namaste_mapping_summary(self, obj):
        """Get summary statistics of all NAMASTE mappings"""
        try:
            # Using TermMapping model instead of ConceptMapping
            all_mappings = TermMapping.objects.filter(icd_term=obj)

            total_mappings = all_mappings.count()
            if total_mappings == 0:
                return None

            # Count by system using source_system field
            ayurveda_count = all_mappings.filter(source_system="ayurveda").count()
            siddha_count = all_mappings.filter(source_system="siddha").count()
            unani_count = all_mappings.filter(source_system="unani").count()

            # Quality statistics - using confidence_score field
            stats = all_mappings.aggregate(
                high_confidence_count=Count("id", filter=Q(confidence_score__gte=0.85)),
                avg_confidence=Avg("confidence_score"),
                avg_icd_similarity=Avg("icd_similarity"),
            )

            return {
                "total_mappings": total_mappings,
                "by_system": {
                    "ayurveda": ayurveda_count,
                    "siddha": siddha_count,
                    "unani": unani_count,
                },
                "quality_metrics": {
                    "high_confidence_count": stats["high_confidence_count"],
                    "avg_confidence": round(stats["avg_confidence"] or 0.0, 3),
                    "avg_icd_similarity": round(stats["avg_icd_similarity"] or 0.0, 3),
                    "high_confidence_rate": round(
                        (stats["high_confidence_count"] / total_mappings * 100)
                        if total_mappings > 0
                        else 0.0,
                        1,
                    ),
                },
            }

        except Exception as e:
            logger.error(f"Error in get_namaste_mapping_summary: {str(e)}")
            return None


class MetaInfoSerializer(serializers.Serializer):
    """Serializer for metadata information"""

    system = serializers.CharField(help_text="System type (icd11)")
    concept_id = serializers.IntegerField(help_text="ID of the ICD-11 concept")
    retrieved_at = serializers.DateTimeField(
        help_text="Timestamp when data was retrieved"
    )
    cache_duration = serializers.IntegerField(help_text="Cache duration in seconds")
    api_version = serializers.CharField(help_text="API version")


class ICD11DetailResponseSerializer(serializers.Serializer):
    """Response serializer for ICD-11 concept detail endpoints"""

    concept = ICD11ConceptDetailSerializer(
        help_text="Complete ICD-11 concept with all related NAMASTE mappings"
    )
    meta = MetaInfoSerializer(help_text="Metadata about the request and response")
