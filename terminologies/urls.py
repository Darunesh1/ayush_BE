from django.urls import path

from .views import ayurvedha_fuzzy_search

urlpatterns = [
    path("api/ayurveda/search/", ayurvedha_fuzzy_search, name="ayurvedha-fuzzy-search"),
]

