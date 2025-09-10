from django.contrib import admin

from .models import (
    AyurvedhaModel,
    ICD11Term,
    ICDClassKind,
    SiddhaModel,
    UnaniModel,
)


@admin.register(AyurvedhaModel)
class AyurvedhaModelAdmin(admin.ModelAdmin):
    list_display = ("code", "english_name", "hindi_name", "diacritical_name")
    search_fields = ("code", "english_name", "hindi_name", "diacritical_name")


@admin.register(SiddhaModel)
class SiddhaModelAdmin(admin.ModelAdmin):
    list_display = ("code", "english_name", "tamil_name", "romanized_name")
    search_fields = ("code", "english_name", "tamil_name", "romanized_name")


@admin.register(UnaniModel)
class UnaniModelAdmin(admin.ModelAdmin):
    list_display = ("code", "english_name", "arabic_name", "romanized_name")
    search_fields = ("code", "english_name", "arabic_name", "romanized_name")


@admin.register(ICDClassKind)
class ICDClassKindAdmin(admin.ModelAdmin):
    list_display = ("name",)
    search_fields = ("name",)


@admin.register(ICD11Term)
class ICD11TermAdmin(admin.ModelAdmin):
    list_display = ("code", "title", "chapter_no", "is_leaf", "is_residual")
    search_fields = ("code", "title", "chapter_no")
    list_filter = ("is_leaf", "is_residual", "chapter_no")
