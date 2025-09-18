from django.urls import path

from terminologies import views

urlpatterns = [
    # ============================================================================
    # AYURVEDA TERMINOLOGY ENDPOINTS
    # ============================================================================
    path(
        "ayurveda/search/", views.ayurvedha_fuzzy_search, name="ayurvedha_fuzzy_search"
    ),
    path(
        "ayurveda/autocomplete/",
        views.ayurveda_autocomplete,
        name="ayurveda_autocomplete",
    ),
    path("ayurveda/csv/upload/", views.ayurveda_csv_upload, name="ayurveda_csv_upload"),
    # ============================================================================
    # SIDDHA TERMINOLOGY ENDPOINTS
    # ============================================================================
    path("siddha/search/", views.siddha_fuzzy_search, name="siddha_fuzzy_search"),
    path("siddha/autocomplete/", views.siddha_autocomplete, name="siddha_autocomplete"),
    path("siddha/csv/upload/", views.siddha_csv_upload, name="siddha_csv_upload"),
    # ============================================================================
    # UNANI TERMINOLOGY ENDPOINTS
    # ============================================================================
    path("unani/search/", views.unani_fuzzy_search, name="unani_fuzzy_search"),
    path("unani/autocomplete/", views.unani_autocomplete, name="unani_autocomplete"),
    path("unani/csv/upload/", views.unani_csv_upload, name="unani_csv_upload"),
    # ============================================================================
    # ICD-11 TERMINOLOGY ENDPOINTS
    # ============================================================================
    path("icd11/search/", views.icd11_advanced_search, name="icd11_advanced_search"),
    path("icd11/autocomplete/", views.icd11_autocomplete, name="icd11_autocomplete"),
    # ============================================================================
    # NAMASTE ↔ ICD-11 MAPPING ENDPOINTS
    # ============================================================================
    # Core mapping operations
    path("mappings/", views.search_namaste_mappings, name="search_namaste_mappings"),
    path("mappings/search/", views.search_mappings, name="search_mappings"),
    path("mappings/list/", views.list_mappings, name="list_mappings"),
    # Detailed views
    path(
        "mappings/details/<int:mapping_id>/",
        views.mapping_details,
        name="mapping_details",
    ),
    # Analytics and statistics
    path("mappings/stats/", views.mapping_statistics, name="mapping_statistics"),
]
