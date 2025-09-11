from django.urls import path

from terminologies import views

urlpatterns = [
    path(
        "ayurveda/search/", views.ayurvedha_fuzzy_search, name="ayurvedha-fuzzy-search"
    ),
    path(
        "ayurveda/autocomplete/",
        views.ayurvedha_autocomplete,
        name="ayurvedha_autocomplete",
    ),
    path("siddha/search/", views.siddha_fuzzy_search, name="siddha_fuzzy_search"),
    path("unani/search/", views.unani_fuzzy_search, name="unani_fuzzy_search"),
    path("icd11/search/", views.icd11_advanced_search, name="icd11_advanced_search"),
]
