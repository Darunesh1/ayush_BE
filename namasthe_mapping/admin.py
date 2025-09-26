"""
FIXED Django Admin for NAMASTE-ICD Terminology Mappings
All format_html SafeString errors resolved
"""

import json
from typing import Any, Dict, List, Optional, Union

from django.contrib import admin, messages
from django.contrib.admin import SimpleListFilter
from django.contrib.contenttypes.models import ContentType
from django.db import models, transaction
from django.db.models import Avg, Case, Count, FloatField, Max, Min, Q, Value, When
from django.forms import ModelForm, ValidationError
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import render
from django.urls import path, reverse
from django.utils import timezone
from django.utils.html import format_html
from django.utils.safestring import mark_safe

from .models import ConceptMapping, MappingAudit, TerminologyMapping

# =============================================================================
# CUSTOM FILTERS FOR PERFORMANCE
# =============================================================================


class ConfidenceRangeFilter(SimpleListFilter):
    """Optimized confidence score filter with predefined ranges"""

    title = "Confidence Score"
    parameter_name = "confidence_range"

    def lookups(self, request, model_admin):
        return [
            ("very_high", "Very High (≥ 0.95)"),
            ("high", "High (0.85-0.94)"),
            ("medium", "Medium (0.75-0.84)"),
            ("low", "Low (0.65-0.74)"),
            ("very_low", "Very Low (< 0.65)"),
        ]

    def queryset(self, request, queryset):
        value = self.value()
        if value == "very_high":
            return queryset.filter(confidence_score__gte=0.95)
        elif value == "high":
            return queryset.filter(
                confidence_score__gte=0.85, confidence_score__lt=0.95
            )
        elif value == "medium":
            return queryset.filter(
                confidence_score__gte=0.75, confidence_score__lt=0.85
            )
        elif value == "low":
            return queryset.filter(
                confidence_score__gte=0.65, confidence_score__lt=0.75
            )
        elif value == "very_low":
            return queryset.filter(confidence_score__lt=0.65)
        return queryset


class ValidationStatusFilter(SimpleListFilter):
    """Filter for validation status with counts"""

    title = "Validation Status"
    parameter_name = "validation_status"

    def lookups(self, request, model_admin):
        return [
            ("validated", "✅ Validated"),
            ("needs_review", "⚠️ Needs Review"),
            ("has_issues", "❌ Has Issues"),
            ("high_conf_unvalidated", "🔵 High Confidence (Unvalidated)"),
        ]

    def queryset(self, request, queryset):
        value = self.value()
        if value == "validated":
            return queryset.filter(is_validated=True)
        elif value == "needs_review":
            return queryset.filter(needs_review=True)
        elif value == "has_issues":
            return queryset.filter(has_issues=True)
        elif value == "high_conf_unvalidated":
            return queryset.filter(is_high_confidence=True, is_validated=False)
        return queryset


class SourceSystemFilter(SimpleListFilter):
    """Filter by NAMASTE source system"""

    title = "Source System"
    parameter_name = "source_system"

    def lookups(self, request, model_admin):
        try:
            from terminologies.models import Ayurvedha, Siddha, Unani

            return [
                ("ayurvedha", "🕉️ Ayurveda"),
                ("siddha", "🏺 Siddha"),
                ("unani", "☪️ Unani"),
            ]
        except ImportError:
            return []

    def queryset(self, request, queryset):
        value = self.value()
        if value:
            try:
                from terminologies.models import Ayurvedha, Siddha, Unani

                model_map = {
                    "ayurvedha": Ayurvedha,
                    "siddha": Siddha,
                    "unani": Unani,
                }
                if value in model_map:
                    content_type = ContentType.objects.get_for_model(model_map[value])
                    return queryset.filter(source_content_type=content_type)
            except ImportError:
                pass
        return queryset


# =============================================================================
# CUSTOM FORMS FOR VALIDATION
# =============================================================================


class ConceptMappingForm(ModelForm):
    """Custom form with validation for concept mappings"""

    class Meta:
        model = ConceptMapping
        fields = "__all__"

    def clean(self):
        cleaned_data = super().clean()

        # Validate confidence and similarity scores
        confidence = cleaned_data.get("confidence_score")
        similarity = cleaned_data.get("similarity_score")

        if confidence is not None and not (0.0 <= confidence <= 1.0):
            raise ValidationError("Confidence score must be between 0.0 and 1.0")

        if similarity is not None and not (0.0 <= similarity <= 1.0):
            raise ValidationError("Similarity score must be between 0.0 and 1.0")

        return cleaned_data


# =============================================================================
# OPTIMIZED ADMIN CLASSES - ALL FORMAT_HTML ISSUES FIXED
# =============================================================================


@admin.register(TerminologyMapping)
class TerminologyMappingAdmin(admin.ModelAdmin):
    """Highly optimized admin for TerminologyMapping with fixed formatting"""

    # List view optimization
    list_display = [
        "name_with_icon",
        "source_target_display",
        "status_badge",
        "mapping_statistics",
        "performance_metrics",
        "last_run_display",
        "quick_actions",
    ]
    list_display_links = ["name_with_icon"]
    list_filter = [
        "status",
        "is_active",
        "source_system",
        "target_system",
        "created_at",
    ]
    list_per_page = 25
    list_max_show_all = 100

    # Search optimization
    search_fields = [
        "name",
        "description",
        "source_system",
        "target_system",
    ]

    # Form optimization
    fields = [
        ("name", "status"),
        "description",
        ("source_system", "target_system"),
        ("biobert_model", "similarity_threshold", "confidence_boost"),
        ("is_active", "created_by"),
        ("total_mappings", "validated_mappings", "high_confidence_mappings"),
        ("average_confidence", "average_similarity"),
        "last_mapping_run",
    ]
    readonly_fields = [
        "total_mappings",
        "validated_mappings",
        "high_confidence_mappings",
        "average_confidence",
        "average_similarity",
        "last_mapping_run",
        "created_at",
        "updated_at",
    ]

    # Performance optimization
    def get_queryset(self, request):
        """Optimize queryset with annotations for computed fields"""
        return (
            super()
            .get_queryset(request)
            .annotate(
                concept_count=Count("concept_mappings"),
                validated_count=Count(
                    "concept_mappings", filter=Q(concept_mappings__is_validated=True)
                ),
                high_conf_count=Count(
                    "concept_mappings",
                    filter=Q(concept_mappings__is_high_confidence=True),
                ),
                avg_confidence=Avg("concept_mappings__confidence_score"),
                avg_similarity=Avg("concept_mappings__similarity_score"),
            )
        )

    # FIXED: All display methods with proper format_html usage
    def name_with_icon(self, obj):
        """Display name with status icon"""
        icons = {
            "draft": "📝",
            "active": "✅",
            "review": "⚠️",
            "archived": "📦",
        }
        icon = icons.get(obj.status, "❓")
        return format_html("<span>{}</span> <strong>{}</strong>", icon, obj.name)

    name_with_icon.short_description = "Mapping Configuration"
    name_with_icon.admin_order_field = "name"

    def source_target_display(self, obj):
        """Display source → target with formatting"""
        return format_html(
            "<code>{}</code> → <code>{}</code>", obj.source_system, obj.target_system
        )

    source_target_display.short_description = "System Mapping"

    def status_badge(self, obj):
        """Display status as colored badge"""
        colors = {
            "draft": "#6c757d",
            "active": "#28a745",
            "review": "#ffc107",
            "archived": "#6c757d",
        }
        color = colors.get(obj.status, "#6c757d")
        return format_html(
            '<span style="background-color: {}; color: white; padding: 2px 8px; border-radius: 3px; font-size: 11px;">{}</span>',
            color,
            obj.get_status_display(),
        )

    status_badge.short_description = "Status"
    status_badge.admin_order_field = "status"

    def mapping_statistics(self, obj):
        """Display mapping counts with progress indicators"""
        total = getattr(obj, "concept_count", obj.total_mappings) or 0
        validated = getattr(obj, "validated_count", obj.validated_mappings) or 0
        high_conf = getattr(obj, "high_conf_count", obj.high_confidence_mappings) or 0

        if total == 0:
            return format_html("<em>No mappings</em>")

        validation_rate = (validated / total) * 100 if total > 0 else 0
        high_conf_rate = (high_conf / total) * 100 if total > 0 else 0

        # FIXED: Pre-format decimal values
        val_rate_str = "{:.1f}".format(validation_rate)
        conf_rate_str = "{:.1f}".format(high_conf_rate)

        return format_html(
            '<div style="font-size: 11px;">'
            "📊 Total: <strong>{}</strong><br>"
            "✅ Validated: <strong>{}</strong> ({}%)<br>"
            "🔥 High Conf: <strong>{}</strong> ({}%)"
            "</div>",
            total,
            validated,
            val_rate_str,
            high_conf,
            conf_rate_str,
        )

    mapping_statistics.short_description = "Statistics"

    def performance_metrics(self, obj):
        """Display performance metrics"""
        avg_conf = getattr(obj, "avg_confidence", obj.average_confidence) or 0
        avg_sim = getattr(obj, "avg_similarity", obj.average_similarity) or 0

        # FIXED: Pre-format decimal values
        conf_str = "{:.3f}".format(avg_conf)
        sim_str = "{:.3f}".format(avg_sim)
        quality_str = "{:.3f}".format((avg_conf + avg_sim) / 2)

        return format_html(
            '<div style="font-size: 11px;">'
            "🎯 Avg Confidence: <strong>{}</strong><br>"
            "📏 Avg Similarity: <strong>{}</strong><br>"
            "⚡ Quality: <strong>{}</strong>"
            "</div>",
            conf_str,
            sim_str,
            quality_str,
        )

    performance_metrics.short_description = "Performance"

    def last_run_display(self, obj):
        """Display last mapping run with relative time"""
        if not obj.last_mapping_run:
            return format_html('<em style="color: #6c757d;">Never run</em>')

        return format_html(
            '<span title="{}">{}</span>',
            obj.last_mapping_run.strftime("%Y-%m-%d %H:%M:%S"),
            obj.last_mapping_run.strftime("%m/%d %H:%M"),
        )

    last_run_display.short_description = "Last Run"
    last_run_display.admin_order_field = "last_mapping_run"

    def quick_actions(self, obj):
        """Display quick action buttons"""
        if obj.pk:
            changelist_url = reverse("admin:namasthe_mapping_conceptmapping_changelist")
            change_url = reverse(
                "admin:namasthe_mapping_terminologymapping_change", args=[obj.pk]
            )
            return format_html(
                '<div style="white-space: nowrap;">'
                '<a href="{}?mapping__id__exact={}" style="background: #007cba; color: white; padding: 2px 6px; border-radius: 3px; text-decoration: none; font-size: 10px;">📊 Stats</a> '
                '<a href="{}" style="background: #28a745; color: white; padding: 2px 6px; border-radius: 3px; text-decoration: none; font-size: 10px;">🔄 Refresh</a>'
                "</div>",
                changelist_url,
                obj.pk,
                change_url,
            )
        return "-"

    quick_actions.short_description = "Actions"

    # Actions
    actions = ["refresh_statistics", "mark_active", "mark_for_review"]

    @admin.action(description="🔄 Refresh mapping statistics")
    def refresh_statistics(self, request, queryset):
        """Bulk refresh statistics for selected mappings"""
        count = 0
        for mapping in queryset:
            mapping.update_statistics()
            count += 1
        self.message_user(
            request,
            "Successfully refreshed statistics for {} mapping configuration(s).".format(
                count
            ),
            messages.SUCCESS,
        )

    @admin.action(description="✅ Mark as active")
    def mark_active(self, request, queryset):
        """Mark selected mappings as active"""
        count = queryset.update(status="active", is_active=True)
        self.message_user(
            request,
            "Successfully marked {} mapping(s) as active.".format(count),
            messages.SUCCESS,
        )

    @admin.action(description="⚠️ Mark for review")
    def mark_for_review(self, request, queryset):
        """Mark selected mappings for review"""
        count = queryset.update(status="review")
        self.message_user(
            request,
            "Successfully marked {} mapping(s) for review.".format(count),
            messages.WARNING,
        )


@admin.register(ConceptMapping)
class ConceptMappingAdmin(admin.ModelAdmin):
    """Ultra-optimized admin for ConceptMapping with ALL format_html issues fixed"""

    form = ConceptMappingForm

    # Optimized list view
    list_display = [
        "mapping_summary",
        "source_target_info",
        "confidence_similarity_display",  # This was causing the error
        "validation_status_display",
        "relationship_badge",
        "method_display",
    ]
    list_display_links = ["mapping_summary"]
    list_filter = [
        ConfidenceRangeFilter,
        ValidationStatusFilter,
        SourceSystemFilter,
        "relationship",
        "mapping_method",
        "is_validated",
        "needs_review",
        "has_issues",
        "mapping",
    ]
    list_per_page = 50
    list_max_show_all = 200

    # Advanced search
    search_fields = [
        "target_concept__code",
        "target_concept__title",
        "mapping__name",
        "validation_notes",
        "validated_by",
    ]

    # Form organization
    fieldsets = [
        (
            "Mapping Configuration",
            {
                "fields": [
                    "mapping",
                    ("source_content_type", "source_object_id"),
                    "target_concept",
                    "relationship",
                ]
            },
        ),
        (
            "AI Scores & Methods",
            {
                "fields": [
                    ("similarity_score", "confidence_score"),
                    ("is_high_confidence", "mapping_method"),
                ],
            },
        ),
        (
            "Validation & Review",
            {
                "fields": [
                    ("is_validated", "needs_review", "has_issues"),
                    ("validated_by", "validated_at"),
                    "validation_notes",
                ]
            },
        ),
        (
            "System Information",
            {
                "fields": [
                    ("created_at", "updated_at"),
                ],
                "classes": ["collapse"],
            },
        ),
    ]
    readonly_fields = [
        "is_high_confidence",
        "created_at",
        "updated_at",
    ]

    # Performance optimization
    def get_queryset(self, request):
        """Heavily optimized queryset with all necessary joins"""
        return (
            super()
            .get_queryset(request)
            .select_related("mapping", "source_content_type", "target_concept")
            .annotate(
                target_code=models.F("target_concept__code"),
                target_title=models.F("target_concept__title"),
                mapping_name=models.F("mapping__name"),
            )
        )

    # FIXED: All display methods with proper format_html usage
    def mapping_summary(self, obj):
        """Compact mapping summary with visual indicators"""
        confidence_color = self._get_confidence_color(obj.confidence_score)
        mapping_name = getattr(
            obj, "mapping_name", obj.mapping.name if obj.mapping else "Unknown"
        )

        return format_html(
            '<div style="line-height: 1.2;">'
            '<strong style="color: {};">ID: {}</strong><br>'
            '<small style="color: #666;">{}</small>'
            "</div>",
            confidence_color,
            str(obj.id)[:8],
            mapping_name,
        )

    mapping_summary.short_description = "Mapping"
    mapping_summary.admin_order_field = "mapping"

    def source_target_info(self, obj):
        """Display source and target concept information efficiently"""
        # Get source info
        source_display = self._get_source_display(obj)

        # Get target info (pre-fetched)
        target_code = getattr(
            obj, "target_code", obj.target_concept.code if obj.target_concept else "N/A"
        )
        target_title = getattr(
            obj,
            "target_title",
            obj.target_concept.title if obj.target_concept else "Unknown",
        )

        # Truncate title if too long
        display_title = (
            target_title[:50] + "..."
            if len(str(target_title)) > 50
            else str(target_title)
        )

        return format_html(
            '<div style="font-size: 11px; line-height: 1.3;">'
            "<div><strong>From:</strong> {}</div>"
            "<div><strong>To:</strong> <code>{}</code></div>"
            '<div style="color: #666; max-width: 200px; overflow: hidden; text-overflow: ellipsis;">{}</div>'
            "</div>",
            source_display,
            target_code,
            display_title,
        )

    source_target_info.short_description = "Source → Target"

    def confidence_similarity_display(self, obj):
        """FIXED: Display confidence and similarity with visual indicators"""
        conf_color = self._get_confidence_color(obj.confidence_score)
        sim_color = self._get_confidence_color(obj.similarity_score)
        quality = (obj.confidence_score + obj.similarity_score) / 2

        # FIXED: Pre-format the decimal values before passing to format_html
        conf_formatted = "{:.3f}".format(obj.confidence_score)
        sim_formatted = "{:.3f}".format(obj.similarity_score)
        quality_formatted = "{:.3f}".format(quality)

        return format_html(
            '<div style="font-size: 11px;">'
            '<div>🎯 <span style="color: {}; font-weight: bold;">{}</span></div>'
            '<div>📏 <span style="color: {}; font-weight: bold;">{}</span></div>'
            "<div>💎 <strong>{}</strong></div>"
            "</div>",
            conf_color,
            conf_formatted,
            sim_color,
            sim_formatted,
            quality_formatted,
        )

    confidence_similarity_display.short_description = "Scores"
    confidence_similarity_display.admin_order_field = "confidence_score"

    def validation_status_display(self, obj):
        """Compact validation status with visual indicators"""
        status_parts = []

        if obj.is_validated:
            status_parts.append('<span style="color: #28a745;">✅ Validated</span>')
            if obj.validated_by:
                validated_by_short = str(obj.validated_by)[:10]
                status_parts.append("<small>by {}</small>".format(validated_by_short))
        elif obj.needs_review:
            status_parts.append('<span style="color: #ffc107;">⚠️ Review</span>')
        elif obj.has_issues:
            status_parts.append('<span style="color: #dc3545;">❌ Issues</span>')
        else:
            status_parts.append('<span style="color: #6c757d;">➖ Pending</span>')

        return format_html(
            '<div style="font-size: 11px;">{}</div>', "<br>".join(status_parts)
        )

    validation_status_display.short_description = "Status"
    validation_status_display.admin_order_field = "is_validated"

    def relationship_badge(self, obj):
        """Display relationship as compact badge"""
        colors = {
            "equivalent": "#28a745",
            "related-to": "#17a2b8",
            "source-is-narrower-than-target": "#ffc107",
            "source-is-broader-than-target": "#fd7e14",
            "not-related-to": "#dc3545",
        }
        color = colors.get(obj.relationship, "#6c757d")
        short_names = {
            "equivalent": "EQV",
            "related-to": "REL",
            "source-is-narrower-than-target": "NAR",
            "source-is-broader-than-target": "BRD",
            "not-related-to": "NOT",
        }
        short_name = short_names.get(obj.relationship, obj.relationship[:3].upper())

        return format_html(
            '<span style="background: {}; color: white; padding: 1px 4px; border-radius: 2px; font-size: 9px; font-weight: bold;" title="{}">{}</span>',
            color,
            obj.get_relationship_display(),
            short_name,
        )

    relationship_badge.short_description = "Rel"
    relationship_badge.admin_order_field = "relationship"

    def method_display(self, obj):
        """Display mapping method with icon"""
        icons = {
            "onnx_biobert": "🤖",
            "manual": "👤",
            "hybrid": "🔗",
            "imported": "📥",
        }
        icon = icons.get(obj.mapping_method, "❓")
        return format_html(
            '<span title="{}">{}</span>', obj.get_mapping_method_display(), icon
        )

    method_display.short_description = "Method"
    method_display.admin_order_field = "mapping_method"

    # Helper methods
    def _get_confidence_color(self, score):
        """Get color based on confidence score"""
        if score >= 0.95:
            return "#28a745"  # Green
        elif score >= 0.85:
            return "#28a745"  # Green
        elif score >= 0.75:
            return "#ffc107"  # Yellow
        elif score >= 0.65:
            return "#fd7e14"  # Orange
        else:
            return "#dc3545"  # Red

    def _get_source_display(self, obj):
        """Efficiently get source display name"""
        if not obj.source_concept:
            return "Unknown Source"

        # Get source system type
        model_name = obj.source_content_type.model
        system_icons = {
            "ayurvedha": "🕉️",
            "siddha": "🏺",
            "unani": "☪️",
        }
        icon = system_icons.get(model_name, "❓")

        # Get source name efficiently
        source_name = getattr(
            obj.source_concept, "english_name", str(obj.source_concept)
        )
        source_code = getattr(obj.source_concept, "code", "")

        if source_code:
            display = "{} {}".format(icon, source_code[:10])
        else:
            display = "{} {}".format(icon, source_name[:15])

        return display

    # Bulk actions
    actions = [
        "mark_as_validated",
        "flag_for_review",
        "update_confidence_flags",
    ]

    @admin.action(description="✅ Mark as validated")
    def mark_as_validated(self, request, queryset):
        """Bulk validate selected mappings"""
        count = 0
        with transaction.atomic():
            for mapping in queryset:
                mapping.mark_as_validated(request.user.username, "Bulk validation")
                count += 1
        self.message_user(
            request,
            "Successfully validated {} mapping(s).".format(count),
            messages.SUCCESS,
        )

    @admin.action(description="⚠️ Flag for review")
    def flag_for_review(self, request, queryset):
        """Bulk flag mappings for review"""
        count = 0
        with transaction.atomic():
            for mapping in queryset:
                mapping.flag_for_review("Bulk flagged for review")
                count += 1
        self.message_user(
            request,
            "Successfully flagged {} mapping(s) for review.".format(count),
            messages.WARNING,
        )

    @admin.action(description="🔄 Update confidence flags")
    def update_confidence_flags(self, request, queryset):
        """Update is_high_confidence flags based on current scores"""
        high_conf_updated = queryset.filter(confidence_score__gte=0.9).update(
            is_high_confidence=True
        )
        low_conf_updated = queryset.filter(confidence_score__lt=0.9).update(
            is_high_confidence=False
        )
        total_updated = high_conf_updated + low_conf_updated
        self.message_user(
            request,
            "Updated confidence flags for {} mapping(s).".format(total_updated),
            messages.INFO,
        )


@admin.register(MappingAudit)
class MappingAuditAdmin(admin.ModelAdmin):
    """Read-only optimized admin for audit trails with fixed formatting"""

    # Read-only configuration
    list_display = [
        "audit_summary",
        "mapping_info",
        "action_badge",
        "user_info",
        "timestamp_display",
        "change_summary",
    ]
    list_display_links = None
    list_filter = [
        "action",
        "timestamp",
        "user_name",
        "concept_mapping__mapping",
    ]
    list_per_page = 100

    # Search configuration
    search_fields = [
        "concept_mapping__id",
        "user_name",
        "reason",
        "ip_address",
    ]

    # Read-only fields
    readonly_fields = [
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
    ]

    # No add/delete permissions
    def has_add_permission(self, request):
        return False

    def has_delete_permission(self, request, obj=None):
        return False

    def has_change_permission(self, request, obj=None):
        return request.user.is_superuser

    # Performance optimization
    def get_queryset(self, request):
        return (
            super()
            .get_queryset(request)
            .select_related(
                "concept_mapping",
                "concept_mapping__mapping",
                "concept_mapping__target_concept",
            )
        )

    # FIXED: Display methods with proper format_html usage
    def audit_summary(self, obj):
        """Compact audit entry summary"""
        mapping_id = (
            str(obj.concept_mapping.id)[:8] if obj.concept_mapping else "Unknown"
        )
        return format_html(
            "<strong>{}</strong><br><small>{}</small>", str(obj.id)[:8], mapping_id
        )

    audit_summary.short_description = "Audit Entry"

    def mapping_info(self, obj):
        """Display related mapping information"""
        if obj.concept_mapping:
            mapping_name = (
                obj.concept_mapping.mapping.name
                if obj.concept_mapping.mapping
                else "Unknown"
            )
            target_info = (
                str(obj.concept_mapping.target_concept)[:50]
                if obj.concept_mapping.target_concept
                else "No target"
            )
            return format_html(
                '<div style="font-size: 11px;">{}<br><em>{}</em></div>',
                mapping_name,
                target_info,
            )
        return "No mapping"

    mapping_info.short_description = "Mapping"

    def action_badge(self, obj):
        """Display action as colored badge"""
        colors = {
            "create": "#28a745",
            "update": "#17a2b8",
            "validate": "#ffc107",
            "flag": "#fd7e14",
            "delete": "#dc3545",
        }
        color = colors.get(obj.action, "#6c757d")
        return format_html(
            '<span style="background: {}; color: white; padding: 2px 6px; border-radius: 3px; font-size: 10px;">{}</span>',
            color,
            obj.get_action_display(),
        )

    action_badge.short_description = "Action"
    action_badge.admin_order_field = "action"

    def user_info(self, obj):
        """Display user and IP information"""
        user_name = obj.user_name or "System"
        ip_address = obj.ip_address or "Unknown IP"
        return format_html(
            '<div style="font-size: 11px;">{}<br><small>{}</small></div>',
            user_name,
            ip_address,
        )

    user_info.short_description = "User"

    def timestamp_display(self, obj):
        """Display timestamp in local format"""
        return obj.timestamp.strftime("%m/%d %H:%M:%S")

    timestamp_display.short_description = "When"
    timestamp_display.admin_order_field = "timestamp"

    def change_summary(self, obj):
        """Display change summary from JSON field"""
        if obj.field_changes:
            changes = obj.field_changes
            if isinstance(changes, dict) and changes:
                summary = []
                for field, change in list(changes.items())[:3]:  # Show max 3 changes
                    change_str = (
                        str(change)[:20] + "..."
                        if len(str(change)) > 20
                        else str(change)
                    )
                    summary.append("{}: {}".format(field, change_str))
                return format_html(
                    '<div style="font-size: 10px;">{}</div>', "<br>".join(summary)
                )
        return format_html('<em style="color: #666;">No field changes</em>')

    change_summary.short_description = "Changes"


# Custom admin site configuration
admin.site.site_header = "NAMASTE-ICD Terminology Mapping Administration"
admin.site.site_title = "NAMASTE Admin"
admin.site.index_title = "Terminology Mapping Management"
