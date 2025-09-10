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
]
