from django.contrib.postgres.search import TrigramSimilarity
from django.db.models import Q
from rest_framework.decorators import api_view
from rest_framework.pagination import PageNumberPagination
from rest_framework.response import Response

from .models import AyurvedhaModel
from .serializers import AyurvedhaModelSerializer


@api_view(["GET"])
def ayurvedha_fuzzy_search(request):
    search_term = request.query_params.get("q", "").strip()

    if not search_term:
        queryset = AyurvedhaModel.objects.all().order_by("code")
    else:
        fuzzy_qs = AyurvedhaModel.objects.annotate(
            similarity_code=TrigramSimilarity("code", search_term),
            similarity_english=TrigramSimilarity("english_name", search_term),
            similarity_hindi=TrigramSimilarity("hindi_name", search_term),
            similarity_diacritical=TrigramSimilarity("diacritical_name", search_term),
        ).filter(
            Q(similarity_code__gt=0.1)
            | Q(similarity_english__gt=0.1)
            | Q(similarity_hindi__gt=0.1)
            | Q(similarity_diacritical__gt=0.1)
        )

        exact_qs = AyurvedhaModel.objects.filter(
            Q(code__iexact=search_term)
            | Q(english_name__iexact=search_term)
            | Q(hindi_name__iexact=search_term)
            | Q(diacritical_name__iexact=search_term)
        )

        queryset = (fuzzy_qs | exact_qs).distinct()

        # queryset = queryset.annotate(
        #     max_similarity=(
        #         TrigramSimilarity("code", search_term)
        #         + TrigramSimilarity("english_name", search_term)
        #         + TrigramSimilarity("hindi_name", search_term)
        #         + TrigramSimilarity("diacritical_name", search_term)
        #     )
        # ).order_by("-max_similarity", "code")
        queryset = queryset.annotate(
            weighted_score=(
                TrigramSimilarity("english_name", search_term) * 2.5
                + TrigramSimilarity("code", search_term) * 1.0
                + TrigramSimilarity("hindi_name", search_term) * 0.8
                + TrigramSimilarity("diacritical_name", search_term) * 0.8
            )
        ).order_by("-weighted_score", "code")

    paginator = PageNumberPagination()
    paginator.page_size = 20
    page = paginator.paginate_queryset(queryset, request)

    serializer = AyurvedhaModelSerializer(page, many=True)
    return paginator.get_paginated_response(serializer.data)
