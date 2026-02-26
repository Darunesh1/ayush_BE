from django.urls import path
from . import views

urlpatterns = [
    # ── Ayurveda ───────────────────────────────────
    path("ayurveda/", views.ayurveda_list, name="ayurveda-list"),
    path("ayurveda/search/", views.ayurveda_search, name="ayurveda-search"),
    path("ayurveda/autocomplete/", views.ayurveda_autocomplete, name="ayurveda-autocomplete"),
    path("ayurveda/<str:code>", views.ayurveda_detail, name="ayurveda-detail"),

    # ── Siddha ─────────────────────────────────────
    path("siddha/", views.siddha_list, name="siddha-list"),
    path("siddha/search/", views.siddha_search, name="siddha-search"),
    path("siddha/autocomplete/", views.siddha_autocomplete, name="siddha-autocomplete"),
    path("siddha/<str:code>", views.siddha_detail, name="siddha-detail"),

    # ── Unani ──────────────────────────────────────
    path("unani/", views.unani_list, name="unani-list"),
    path("unani/search/", views.unani_search, name="unani-search"),
    path("unani/autocomplete/", views.unani_autocomplete, name="unani-autocomplete"),
    path("unani/<str:code>", views.unani_detail, name="unani-detail"),

    # ── ICD-11 ─────────────────────────────────────
    path("icd11/", views.icd11_list, name="icd11-list"),
    path("icd11/search/", views.icd11_search, name="icd11-search"),
    path("icd11/autocomplete/", views.icd11_autocomplete, name="icd11-autocomplete"),
    path("icd11/<str:code>", views.icd11_detail, name="icd11-detail"),

    # ── Mappings ───────────────────────────────────
    path("mappings/", views.mappings_list, name="mappings-list"),
    path("mappings/stats/", views.mapping_stats, name="mapping-stats"),
    path("mappings/<str:code>", views.mapping_detail, name="mapping-detail"),

    # ── Combined search ────────────────────────────
    path("search/combined/", views.combined_search, name="combined-search"),

    # ── Autocomplete (all systems) ─────────────────
    path("autocomplete/", views.autocomplete, name="autocomplete"),
]