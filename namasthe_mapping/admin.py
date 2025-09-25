from django.contrib import admin
from .models import TerminologyMapping, ConceptMapping, MappingAudit


class ConceptMappingInline(admin.TabularInline):
    """Inline to show ConceptMappings inside TerminologyMapping"""
    model = ConceptMapping
    extra = 0
    fields = (
        "get_source_display",
        "get_target_display",
        "relationship",
        "confidence_score",
        "similarity_score",
        "is_validated",
        "is_high_confidence",
    )
    readonly_fields = (
        "get_source_display",
        "get_target_display",
        "confidence_score",
        "similarity_score",
        "is_validated",
        "is_high_confidence",
    )
    show_change_link = True


@admin.register(TerminologyMapping)
class TerminologyMappingAdmin(admin.ModelAdmin):
    list_display = (
        "name",
        "source_system",
        "target_system",
        "status",
        "is_active",
        "total_mappings",
        "validated_mappings",
        "high_confidence_mappings",
        "average_confidence",
        "average_similarity",
        "last_mapping_run",
        "created_at",
    )
    list_filter = ("status", "is_active", "source_system", "target_system", "created_at")
    search_fields = ("name", "source_system", "target_system", "description")
    ordering = ("-created_at",)
    inlines = [ConceptMappingInline]
    readonly_fields = (
        "total_mappings",
        "validated_mappings",
        "high_confidence_mappings",
        "average_confidence",
        "average_similarity",
        "last_mapping_run",
        "created_at",
        "updated_at",
    )


@admin.register(ConceptMapping)
class ConceptMappingAdmin(admin.ModelAdmin):
    list_display = (
        "get_source_display",
        "get_target_display",
        "relationship",
        "similarity_score",
        "confidence_score",
        "confidence_category",
        "is_validated",
        "needs_review",
        "is_high_confidence",
        "mapping_method",
        "created_at",
    )
    list_filter = (
        "relationship",
        "is_validated",
        "needs_review",
        "is_high_confidence",
        "mapping_method",
        "created_at",
    )
    search_fields = (
        "validation_notes",
        "validated_by",
        "target_concept__title",
        "target_concept__code",
    )
    ordering = ("-confidence_score", "-similarity_score")
    readonly_fields = (
        "created_at",
        "updated_at",
        "validated_at",
        "confidence_category",
        "quality_score",
    )


@admin.register(MappingAudit)
class MappingAuditAdmin(admin.ModelAdmin):
    list_display = (
        "concept_mapping",
        "action",
        "user_name",
        "ip_address",
        "timestamp",
    )
    list_filter = ("action", "user_name", "timestamp")
    search_fields = ("concept_mapping__id", "user_name", "reason")
    ordering = ("-timestamp",)
    readonly_fields = (
        "concept_mapping",
        "action",
        "field_changes",
        "reason",
        "user_name",
        "ip_address",
        "user_agent",
        "timestamp",
        "system_version",
        "biobert_model_version",
    )
