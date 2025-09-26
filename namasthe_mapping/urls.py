"""
urls.py - Complete URL Configuration for NAMASTE Concept Detail Views
"""

from django.urls import path

from . import views

app_name = "namasthe_mapping"

urlpatterns = [
    path(
        "ayurveda/<int:concept_id>/detail/",
        views.get_ayurveda_concept_detail,
        name="ayurveda-concept-detail",
    ),
    path(
        "siddha/<int:concept_id>/detail/",
        views.get_siddha_concept_detail,
        name="siddha-concept-detail",
    ),
    path(
        "unani/<int:concept_id>/detail/",
        views.get_unani_concept_detail,
        name="unani-concept-detail",
    ),
    # Manual mapping endpoints
    path(
        "manual/create/",
        views.create_manual_mapping,
        name="create_manual_mapping",
    ),
    path("manual/update/", views.update_mapping, name="update_mapping"),
    path(
        "manual/search_icd11/",
        views.search_icd11_codes,
        name="search_icd11",
    ),
]
