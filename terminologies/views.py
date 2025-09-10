from django.contrib.postgres.search import TrigramSimilarity
from django.db.models import Q
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.pagination import PageNumberPagination
from rest_framework.response import Response

from .models import AyurvedhaModel
from .serializers import AyurvedhaModelSerializer


@api_view(["GET"])
def ayurvedha_fuzzy_search(request):
    search_term = request.query_params.get("q", "").strip()
    if not search_term:
        # Return all records ordered by code if no search term
        queryset = AyurvedhaModel.objects.all().order_by("code")
    else:
        queryset = (
            AyurvedhaModel.objects.annotate(
                similarity_code=TrigramSimilarity("code", search_term),
                similarity_english=TrigramSimilarity("english_name", search_term),
                similarity_hindi=TrigramSimilarity("hindi_name", search_term),
                similarity_diacritical=TrigramSimilarity(
                    "diacritical_name", search_term
                ),
            )
            .filter(
                Q(similarity_code__gt=0.3)
                | Q(similarity_english__gt=0.3)
                | Q(similarity_hindi__gt=0.3)
                | Q(similarity_diacritical__gt=0.3)
            )
            .order_by(
                "-similarity_code",
                "-similarity_english",
                "-similarity_hindi",
                "-similarity_diacritical",
            )
        )

    # Paginate results
    paginator = PageNumberPagination()
    paginator.page_size = 20
    page = paginator.paginate_queryset(queryset, request)

    serializer = AyurvedhaModelSerializer(page, many=True)
    return paginator.get_paginated_response(serializer.data)
