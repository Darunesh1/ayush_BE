import json

from django.contrib import admin, messages
from django.utils import timezone
from django.utils.html import format_html

from .models import (
    Ayurvedha,
    ICD11Term,
    Siddha,
    Unani,
)


@admin.register(Ayurvedha)
class AyurvedhaAdmin(admin.ModelAdmin):
    list_display = ("code", "english_name", "hindi_name", "diacritical_name")
    search_fields = ("code", "english_name", "hindi_name", "diacritical_name")


@admin.register(Siddha)
class SiddhaAdmin(admin.ModelAdmin):
    list_display = ("code", "english_name", "tamil_name", "romanized_name")
    search_fields = ("code", "english_name", "tamil_name", "romanized_name")


@admin.register(Unani)
class UnaniAdmin(admin.ModelAdmin):
    list_display = ("code", "english_name", "arabic_name", "romanized_name")
    search_fields = ("code", "english_name", "arabic_name", "romanized_name")


@admin.register(ICD11Term)
class ICDTermAdmin(admin.ModelAdmin):
    # Limit queries and pagination
    list_per_page = 50
    list_max_show_all = 200

    # Optimized display fields
    list_display = (
        "code",
        "title_truncated",
        "class_kind",
        "has_definition",
        "synonym_count",
        "created_at",
    )

    # Efficient filtering
    list_filter = (
        "class_kind",
        ("created_at", admin.DateFieldListFilter),
        "updated_at",
    )

    # Optimized search - avoid heavy text fields
    search_fields = (
        "code__iexact",  # Exact match first (fastest)
        "code__istartswith",  # Then starts with
        "title__icontains",  # Title search only
        "foundation_uri__icontains",  # URI search
    )

    # Optimize ordering
    ordering = ("code", "title")

    # Enhanced fieldsets with pretty JSON displays
    fieldsets = (
        (
            "Core Information",
            {
                "fields": (
                    ("code", "class_kind"),
                    "title",
                    "foundation_uri",
                )
            },
        ),
        (
            "Clinical Definitions",
            {
                "fields": ("definition", "long_definition"),
                "classes": ("collapse",),
            },
        ),
        (
            "Reference Data",
            {
                "fields": ("browser_url", "source"),
                "classes": ("collapse",),
            },
        ),
        (
            "JSON Data - Summary",
            {
                "fields": (
                    "index_terms_summary",
                    "inclusions_summary",
                    "exclusions_summary",
                    "parent_summary",
                ),
            },
        ),
        (
            "JSON Data - Full Display",
            {
                "fields": (
                    "index_terms_pretty",
                    "inclusions_pretty",
                    "exclusions_pretty",
                    "parent_pretty",
                    "postcoordination_scales_pretty",
                ),
                "classes": ("collapse",),
                "description": "Click to expand for full formatted JSON data",
            },
        ),
        (
            "System Fields",
            {
                "fields": ("search_vector", "created_at", "updated_at"),
                "classes": ("collapse",),
            },
        ),
    )

    # Read-only fields (including search_vector and pretty displays)
    readonly_fields = (
        "search_vector",
        "foundation_uri",
        "created_at",
        "updated_at",
        # Add pretty JSON fields to readonly
        "index_terms_summary",
        "index_terms_pretty",
        "inclusions_summary",
        "inclusions_pretty",
        "exclusions_summary",
        "exclusions_pretty",
        "parent_summary",
        "parent_pretty",
        "postcoordination_scales_pretty",
    )

    # Custom actions for bulk operations
    actions = ["update_search_vectors"]

    # Custom methods for optimized list display
    @admin.display(description="Title")
    def title_truncated(self, obj):
        return obj.title[:50] + "..." if len(obj.title) > 50 else obj.title

    @admin.display(description="Has Definition", boolean=True)
    def has_definition(self, obj):
        return bool(obj.definition or obj.long_definition)

    @admin.display(description="Synonyms")
    def synonym_count(self, obj):
        return len(obj.index_terms) if obj.index_terms else 0

    # Summary displays (for quick overview)
    @admin.display(description="Index Terms Summary")
    def index_terms_summary(self, obj):
        if not obj.index_terms:
            return format_html('<span style="color: #999;">None</span>')
        count = len(obj.index_terms)
        first_few = obj.index_terms[:3]
        display = ", ".join(first_few)
        if count > 3:
            display += f"... <strong>({count} total)</strong>"
        return format_html(
            '<div style="max-width: 500px; font-size: 13px;">{}</div>', display
        )

    @admin.display(description="Inclusions Summary")
    def inclusions_summary(self, obj):
        if not obj.inclusions:
            return format_html('<span style="color: #999;">None</span>')
        count = len(obj.inclusions)
        labels = [inc.get("label", "Unknown") for inc in obj.inclusions[:2]]
        display = ", ".join(labels)
        if count > 2:
            display += f"... <strong>({count} total)</strong>"
        return format_html(
            '<div style="max-width: 400px; font-size: 13px;">{}</div>', display
        )

    @admin.display(description="Exclusions Summary")
    def exclusions_summary(self, obj):
        if not obj.exclusions:
            return format_html('<span style="color: #999;">None</span>')
        count = len(obj.exclusions)
        labels = [exc.get("label", "Unknown") for exc in obj.exclusions[:2]]
        display = ", ".join(labels)
        if count > 2:
            display += f"... <strong>({count} total)</strong>"
        return format_html(
            '<div style="max-width: 400px; font-size: 13px;">{}</div>', display
        )

    @admin.display(description="Parent Categories Summary")
    def parent_summary(self, obj):
        if not obj.parent:
            return format_html('<span style="color: #999;">None</span>')
        return format_html("<strong>{}</strong> parent categories", len(obj.parent))

    # Pretty JSON displays (full formatted data)
    @admin.display(description="Index Terms (Full)")
    def index_terms_pretty(self, obj):
        if not obj.index_terms:
            return format_html('<span style="color: #999;">No index terms</span>')
        return format_html(
            """<div style="background: #f8f9fa; padding: 10px; border-radius: 4px; border-left: 3px solid #007cba;">
                <strong>Index Terms ({} total):</strong>
                <pre style="margin-top: 8px; font-size: 12px; max-height: 300px; overflow-y: auto;">{}</pre>
            </div>""",
            len(obj.index_terms),
            json.dumps(obj.index_terms, indent=2, ensure_ascii=False),
        )

    @admin.display(description="Inclusions (Full)")
    def inclusions_pretty(self, obj):
        if not obj.inclusions:
            return format_html('<span style="color: #999;">No inclusions</span>')
        return format_html(
            """<div style="background: #e8f5e8; padding: 10px; border-radius: 4px; border-left: 3px solid #28a745;">
                <strong>Inclusions ({} total):</strong>
                <pre style="margin-top: 8px; font-size: 12px; max-height: 300px; overflow-y: auto;">{}</pre>
            </div>""",
            len(obj.inclusions),
            json.dumps(obj.inclusions, indent=2, ensure_ascii=False),
        )

    @admin.display(description="Exclusions (Full)")
    def exclusions_pretty(self, obj):
        if not obj.exclusions:
            return format_html('<span style="color: #999;">No exclusions</span>')
        return format_html(
            """<div style="background: #fdeaea; padding: 10px; border-radius: 4px; border-left: 3px solid #dc3545;">
                <strong>Exclusions ({} total):</strong>
                <pre style="margin-top: 8px; font-size: 12px; max-height: 300px; overflow-y: auto;">{}</pre>
            </div>""",
            len(obj.exclusions),
            json.dumps(obj.exclusions, indent=2, ensure_ascii=False),
        )

    @admin.display(description="Parent Categories (Full)")
    def parent_pretty(self, obj):
        if not obj.parent:
            return format_html('<span style="color: #999;">No parent categories</span>')
        return format_html(
            """<div style="background: #fff3cd; padding: 10px; border-radius: 4px; border-left: 3px solid #ffc107;">
                <strong>Parent Categories ({} total):</strong>
                <pre style="margin-top: 8px; font-size: 12px; max-height: 200px; overflow-y: auto;">{}</pre>
            </div>""",
            len(obj.parent),
            json.dumps(obj.parent, indent=2, ensure_ascii=False),
        )

    @admin.display(description="Postcoordination Scales (Full)")
    def postcoordination_scales_pretty(self, obj):
        if not obj.postcoordination_scales:
            return format_html(
                '<span style="color: #999;">No postcoordination scales</span>'
            )
        return format_html(
            """<div style="background: #e7f3ff; padding: 10px; border-radius: 4px; border-left: 3px solid #0066cc;">
                <strong>Postcoordination Scales ({} total):</strong>
                <pre style="margin-top: 8px; font-size: 12px; max-height: 400px; overflow-y: auto;">{}</pre>
            </div>""",
            len(obj.postcoordination_scales),
            json.dumps(obj.postcoordination_scales, indent=2, ensure_ascii=False),
        )

    # Performance optimizations
    def get_queryset(self, request):
        """Optimize the base queryset"""
        qs = super().get_queryset(request)
        # Add any select_related/prefetch_related if you add foreign keys later
        return qs

    def has_add_permission(self, request):
        """Disable add - data comes from WHO API"""
        return False

    def has_delete_permission(self, request, obj=None):
        """Limit delete permissions"""
        return request.user.is_superuser

    # Custom action for bulk operations
    @admin.action(description="Update search vectors for selected terms")
    def update_search_vectors(self, request, queryset):
        """Trigger search vector update"""
        updated = queryset.update(updated_at=timezone.now())
        self.message_user(
            request,
            f"Updated search vectors for {updated} terms.",
            messages.SUCCESS,
        )

    # Optional: Add custom CSS for better JSON display
    class Media:
        css = {
            "all": (
                "admin/css/custom_json.css",
            )  # Create this file for additional styling
        }
