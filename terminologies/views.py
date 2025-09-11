from django.contrib.postgres.search import TrigramSimilarity
from django.db.models import Case, FloatField, Q, Value, When
from rest_framework.decorators import api_view
from rest_framework.pagination import PageNumberPagination
from rest_framework.response import Response

from .models import Ayurvedha
from .serializers import AyurvedhaModelSerializer


@api_view(["GET"])
def ayurvedha_fuzzy_search(request):
    search_term = request.query_params.get("q", "").strip()

    if not search_term:
        queryset = Ayurvedha.objects.all().order_by("code")
    else:
        fuzzy_qs = Ayurvedha.objects.annotate(
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

        exact_qs = Ayurvedha.objects.filter(
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


@api_view(["GET"])
def ayurvedha_autocomplete(request):
    search_term = request.query_params.get("q", "").strip()
    limit = min(int(request.query_params.get("limit", 8)), 12)

    if not search_term or len(search_term) < 2:
        return Response({"suggestions": []})

    queryset = get_autocomplete_queryset(search_term, limit)

    suggestions = [
        {
            "id": item.id,
            "code": item.code,
            "title": item.english_name,
            "subtitle": item.hindi_name if item.hindi_name else None,
            "score": round(float(getattr(item, "autocomplete_score", 0)), 1),
        }
        for item in queryset
    ]

    return Response(
        {"suggestions": suggestions, "query": search_term, "count": len(suggestions)}
    )


def get_autocomplete_queryset(search_term, limit):
    # Exact matches first (fastest)
    exact_qs = Ayurvedha.objects.filter(
        Q(code__iexact=search_term) | Q(english_name__iexact=search_term)
    ).only("id", "code", "english_name", "hindi_name")

    # Prefix matches (fast with indexes)
    prefix_qs = Ayurvedha.objects.filter(
        Q(code__istartswith=search_term) | Q(english_name__istartswith=search_term)
    ).only("id", "code", "english_name", "hindi_name")

    # Fuzzy matches (only for 3+ characters)
    fuzzy_qs = Ayurvedha.objects.none()
    if len(search_term) >= 3:
        fuzzy_qs = (
            Ayurvedha.objects.annotate(
                eng_sim=TrigramSimilarity("english_name", search_term),
                code_sim=TrigramSimilarity("code", search_term),
            )
            .filter(Q(eng_sim__gt=0.3) | Q(code_sim__gt=0.4))
            .only("id", "code", "english_name", "hindi_name")
        )

    # Combine and score
    combined_qs = (exact_qs | prefix_qs | fuzzy_qs).distinct()

    return combined_qs.annotate(
        autocomplete_score=Case(
            When(
                Q(code__iexact=search_term) | Q(english_name__iexact=search_term),
                then=Value(100.0),
            ),
            When(Q(code__istartswith=search_term), then=Value(90.0)),
            When(Q(english_name__istartswith=search_term), then=Value(85.0)),
            default=(
                TrigramSimilarity("english_name", search_term) * 60.0
                + TrigramSimilarity("code", search_term) * 40.0
            ),
            output_field=FloatField(),
        )
    ).order_by("-autocomplete_score", "english_name")[:limit]
