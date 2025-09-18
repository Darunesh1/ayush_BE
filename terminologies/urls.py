from django.urls import path

from terminologies import views

urlpatterns = [
    path(
        "ayurveda/search/", views.ayurvedha_fuzzy_search, name="ayurvedha-fuzzy-search"
    ),
    path(
        "ayurveda/autocomplete/",
        views.ayurveda_autocomplete,
        name="ayurvedha_autocomplete/",
    ),
    path("siddha/search/", views.siddha_fuzzy_search, name="siddha_fuzzy_search"),
    path("siddha/autocomplete/", views.siddha_autocomplete, name="siddha_autocomplete"),
    path("unani/search/", views.unani_fuzzy_search, name="unani_fuzzy_search"),
    path("unani/autocomplete/", views.unani_autocomplete, name="unani_autocomplete"),
    path("icd11/search/", views.icd11_advanced_search, name="icd11_advanced_search"),
    path("icd11/autocomplete/", views.icd11_autocomplete, name="icd11_autocomplete"),
    # Mapping
    # Get specific mapping
    path(
        "mappings/",
        views.search_namaste_mappings,
        name="get_namaste_mapping",
    ),
    # Search mappings
    path("mappings/search/", views.search_mappings, name="search_mappings"),
    # List all mappings with filters
    path("mappings/list", views.list_mappings, name="list_mappings"),
    # Detailed mapping view
    path(
        "mappings/details/<int:mapping_id>/",
        views.mapping_details,
        name="mapping_details",
    ),
    # Statistics
    path("mappings/stats/", views.mapping_statistics, name="mapping_statistics"),
]
