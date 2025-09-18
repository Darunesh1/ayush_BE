from django.contrib import admin

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
class ICD11TermAdmin(admin.ModelAdmin):
    list_display = ("code", "title", "foundation_uri")
    search_fields = ("code", "title")
