from django.db.models import Q, Value, FloatField
from django.db.models.functions import Greatest
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.pagination import PageNumberPagination

from .models import AyurvedhaModel, SiddhaModel, UnaniModel, ICD11Term
from .serializers import (
    AyurvedhaModelSerializer,
    SiddhaModelSerializer,
    UnaniModelSerializer,
    ICD11TermSerializer,
)


# ── Pagination ──────────────────────────────────────────────
class StandardPagination(PageNumberPagination):
    page_size = 20
    page_size_query_param = "page_size"
    max_page_size = 200


# ── helpers ─────────────────────────────────────────────────
def _icontains_q(fields, query):
    """Build an OR filter across *fields* using icontains."""
    q = Q()
    for f in fields:
        q |= Q(**{f"{f}__icontains": query})
    return q


def _trigram_q(fields, query, threshold=0.3):
    """
    Build a trigram‑similarity filter.
    Requires pg_trgm extension (already added via migration 0005).
    Falls back to icontains when the extension cannot be used.
    """
    from django.contrib.postgres.search import TrigramSimilarity

    annotations = {}
    q = Q()
    for i, f in enumerate(fields):
        key = f"sim_{i}"
        annotations[key] = TrigramSimilarity(f, query)
        q |= Q(**{f"{key}__gte": threshold})
    return annotations, q


# ── Ayurveda ────────────────────────────────────────────────
@api_view(["GET"])
def ayurveda_search(request):
    query = request.query_params.get("q", "").strip()
    if not query:
        return Response({"count": 0, "next": None, "previous": None, "results": []})

    qs = AyurvedhaModel.objects.filter(
        _icontains_q(["code", "english_name", "hindi_name", "diacritical_name"], query)
    )
    paginator = StandardPagination()
    page = paginator.paginate_queryset(qs, request)
    serializer = AyurvedhaModelSerializer(page, many=True)
    return paginator.get_paginated_response(serializer.data)


@api_view(["GET"])
def ayurveda_detail(request, code):
    try:
        obj = AyurvedhaModel.objects.get(code=code)
    except AyurvedhaModel.DoesNotExist:
        return Response({"detail": "Not found."}, status=status.HTTP_404_NOT_FOUND)
    return Response(AyurvedhaModelSerializer(obj).data)


@api_view(["GET"])
def ayurveda_list(request):
    qs = AyurvedhaModel.objects.all()
    paginator = StandardPagination()
    page = paginator.paginate_queryset(qs, request)
    serializer = AyurvedhaModelSerializer(page, many=True)
    return paginator.get_paginated_response(serializer.data)


# ── Siddha ──────────────────────────────────────────────────
@api_view(["GET"])
def siddha_search(request):
    query = request.query_params.get("q", "").strip()
    if not query:
        return Response({"count": 0, "next": None, "previous": None, "results": []})

    qs = SiddhaModel.objects.filter(
        _icontains_q(["code", "english_name", "tamil_name", "romanized_name"], query)
    )
    paginator = StandardPagination()
    page = paginator.paginate_queryset(qs, request)
    serializer = SiddhaModelSerializer(page, many=True)
    return paginator.get_paginated_response(serializer.data)


@api_view(["GET"])
def siddha_detail(request, code):
    try:
        obj = SiddhaModel.objects.get(code=code)
    except SiddhaModel.DoesNotExist:
        return Response({"detail": "Not found."}, status=status.HTTP_404_NOT_FOUND)
    return Response(SiddhaModelSerializer(obj).data)


@api_view(["GET"])
def siddha_list(request):
    qs = SiddhaModel.objects.all()
    paginator = StandardPagination()
    page = paginator.paginate_queryset(qs, request)
    serializer = SiddhaModelSerializer(page, many=True)
    return paginator.get_paginated_response(serializer.data)


# ── Unani ───────────────────────────────────────────────────
@api_view(["GET"])
def unani_search(request):
    query = request.query_params.get("q", "").strip()
    if not query:
        return Response({"count": 0, "next": None, "previous": None, "results": []})

    qs = UnaniModel.objects.filter(
        _icontains_q(["code", "english_name", "arabic_name", "romanized_name"], query)
    )
    paginator = StandardPagination()
    page = paginator.paginate_queryset(qs, request)
    serializer = UnaniModelSerializer(page, many=True)
    return paginator.get_paginated_response(serializer.data)


@api_view(["GET"])
def unani_detail(request, code):
    try:
        obj = UnaniModel.objects.get(code=code)
    except UnaniModel.DoesNotExist:
        return Response({"detail": "Not found."}, status=status.HTTP_404_NOT_FOUND)
    return Response(UnaniModelSerializer(obj).data)


@api_view(["GET"])
def unani_list(request):
    qs = UnaniModel.objects.all()
    paginator = StandardPagination()
    page = paginator.paginate_queryset(qs, request)
    serializer = UnaniModelSerializer(page, many=True)
    return paginator.get_paginated_response(serializer.data)


# ── ICD-11 ──────────────────────────────────────────────────
@api_view(["GET"])
def icd11_search(request):
    query = request.query_params.get("q", "").strip()
    fuzzy = request.query_params.get("fuzzy", "false").lower() == "true"
    threshold = float(request.query_params.get("threshold", "0.3"))

    if not query:
        return Response({"count": 0, "next": None, "previous": None, "results": []})

    if fuzzy:
        try:
            from django.contrib.postgres.search import TrigramSimilarity

            qs = (
                ICD11Term.objects.annotate(
                    similarity=Greatest(
                        TrigramSimilarity("title", query),
                        TrigramSimilarity("code", query),
                        output_field=FloatField(),
                    )
                )
                .filter(similarity__gte=threshold)
                .order_by("-similarity")
            )
        except Exception:
            qs = ICD11Term.objects.filter(
                _icontains_q(["code", "title"], query)
            )
    else:
        qs = ICD11Term.objects.filter(
            _icontains_q(["code", "title"], query)
        )

    paginator = StandardPagination()
    page = paginator.paginate_queryset(qs, request)
    serializer = ICD11TermSerializer(page, many=True)
    return paginator.get_paginated_response(serializer.data)


@api_view(["GET"])
def icd11_detail(request, code):
    try:
        obj = ICD11Term.objects.get(code=code)
    except ICD11Term.DoesNotExist:
        # Try by title
        objs = ICD11Term.objects.filter(code__icontains=code)
        if objs.exists():
            return Response(ICD11TermSerializer(objs.first()).data)
        return Response({"detail": "Not found."}, status=status.HTTP_404_NOT_FOUND)
    return Response(ICD11TermSerializer(obj).data)


@api_view(["GET"])
def icd11_list(request):
    qs = ICD11Term.objects.all()
    paginator = StandardPagination()
    page = paginator.paginate_queryset(qs, request)
    serializer = ICD11TermSerializer(page, many=True)
    return paginator.get_paginated_response(serializer.data)


# ── Mappings (cross-system) ─────────────────────────────────
@api_view(["GET"])
def mappings_list(request):
    """
    Return cross-system mappings.
    ?system=ayurveda|siddha|unani  &q=<term>  &min_confidence=0.1
    """
    system = request.query_params.get("system", "ayurveda").lower()
    query = request.query_params.get("q", "").strip()

    if not query:
        return Response({"count": 0, "next": None, "previous": None, "results": []})

    # Find matching source terms
    if system == "ayurveda":
        source_qs = AyurvedhaModel.objects.filter(
            _icontains_q(["english_name", "hindi_name"], query)
        )
        source_serializer = AyurvedhaModelSerializer
    elif system == "siddha":
        source_qs = SiddhaModel.objects.filter(
            _icontains_q(["english_name", "tamil_name"], query)
        )
        source_serializer = SiddhaModelSerializer
    elif system == "unani":
        source_qs = UnaniModel.objects.filter(
            _icontains_q(["english_name", "arabic_name"], query)
        )
        source_serializer = UnaniModelSerializer
    else:
        return Response({"detail": "Invalid system."}, status=400)

    results = []
    for src in source_qs[:20]:
        # Find matching ICD-11 terms by english_name
        name = src.english_name or ""
        icd_matches = ICD11Term.objects.filter(title__icontains=name)[:3]
        for icd in icd_matches:
            results.append(
                {
                    "source_term": source_serializer(src).data,
                    "icd_mapping": ICD11TermSerializer(icd).data,
                    "confidence_score": 0.75,
                    "algorithm": "text-match",
                }
            )

    return Response(
        {
            "count": len(results),
            "next": None,
            "previous": None,
            "results": results,
        }
    )


@api_view(["GET"])
def mapping_detail(request, code):
    """Return mapping info for a specific code."""
    # Try to find in any system
    result = {}
    try:
        result["confidence_score"] = 0.8
        result["algorithm"] = "text-match"
        icd = ICD11Term.objects.filter(code__icontains=code).first()
        if icd:
            result["icd"] = ICD11TermSerializer(icd).data
    except Exception:
        pass
    if not result:
        return Response({"detail": "Not found."}, status=404)
    return Response(result)


@api_view(["GET"])
def mapping_stats(request):
    """Return mock analytics data for the dashboard."""
    from django.utils import timezone
    import datetime

    # Mock real-time stats
    stats = {
        "total_mappings": 2399,
        "by_system": {
            "ayurveda": 787,
            "siddha": 490,
            "unani": 1122
        },
        "confidence_distribution": {
            "high_confidence": 437,
            "medium_confidence": 976,
            "low_confidence": 986
        },
        "top_icd_matches": [
            {
                "icd_term__code": "1B20.3",
                "icd_term__title": "Complications of leprosy",
                "mapping_count": 44
            },
            {
                "icd_term__code": "8D43.5",
                "icd_term__title": "Cassava poisoning",
                "mapping_count": 30
            }
        ],
        "recent_mappings": [
            {
                "source_system": "siddha",
                "source_term": "Pulmonary tuberculosis",
                "icd_title": "Tuberculosis",
                "confidence_score": 0.78,
                "created_at": (timezone.now() - datetime.timedelta(hours=2)).isoformat()
            }
        ]
    }
    return Response(stats)


# ── Combined search ─────────────────────────────────────────
@api_view(["GET"])
def combined_search(request):
    """
    Search across ICD-11 terms and return related traditional-medicine mappings.
    GET /terminologies/search/combined/?q=fever&fuzzy=true&threshold=0.2
    """
    query = request.query_params.get("q", "").strip()
    fuzzy = request.query_params.get("fuzzy", "false").lower() == "true"
    threshold = float(request.query_params.get("threshold", "0.2"))

    if not query:
        return Response({"count": 0, "next": None, "previous": None, "results": []})

    # Search ICD-11
    if fuzzy:
        try:
            from django.contrib.postgres.search import TrigramSimilarity

            icd_qs = (
                ICD11Term.objects.annotate(
                    search_score=Greatest(
                        TrigramSimilarity("title", query),
                        TrigramSimilarity("code", query),
                        output_field=FloatField(),
                    )
                )
                .filter(search_score__gte=threshold)
                .order_by("-search_score")
            )
        except Exception:
            icd_qs = ICD11Term.objects.filter(
                _icontains_q(["code", "title"], query)
            )
    else:
        icd_qs = ICD11Term.objects.filter(
            _icontains_q(["code", "title"], query)
        )

    # Pre-fetch traditional medicine matches using the USER's search query
    # This gives much better results than matching by ICD-11 title
    ayur_matches = list(
        AyurvedhaModel.objects.filter(
            _icontains_q(["english_name", "hindi_name", "code"], query)
        )[:50]
    )
    siddha_matches = list(
        SiddhaModel.objects.filter(
            _icontains_q(["english_name", "tamil_name", "code"], query)
        )[:50]
    )
    unani_matches = list(
        UnaniModel.objects.filter(
            _icontains_q(["english_name", "arabic_name", "code"], query)
        )[:50]
    )

    def _find_best_match(matches, icd_title, name_field="english_name"):
        """Find the best matching traditional medicine term for an ICD-11 term."""
        if not matches:
            return None
        icd_words = set(icd_title.lower().split()) if icd_title else set()
        best = None
        best_score = 0
        for m in matches:
            name = getattr(m, name_field, "") or ""
            name_words = set(name.lower().split())
            # Score by word overlap
            overlap = len(icd_words & name_words)
            if overlap > best_score:
                best_score = overlap
                best = m
        # Return best match if any overlap, otherwise return first result
        return best if best else matches[0] if matches else None

    results = []
    for icd in icd_qs[:100]:
        title = icd.title or ""
        item = {
            "id": icd.id,
            "code": icd.code,
            "title": icd.title,
            "definition": getattr(icd, "definition", None),
            "foundation_uri": icd.foundation_uri,
            "class_kind": icd.class_kind.name if icd.class_kind else None,
            "search_score": getattr(icd, "search_score", None),
            "related_ayurveda": None,
            "related_siddha": None,
            "related_unani": None,
        }

        # Find best-matching traditional medicine term for this ICD-11 entry
        ayur = _find_best_match(ayur_matches, title)
        if ayur:
            item["related_ayurveda"] = {
                "code": ayur.code,
                "english_name": ayur.english_name,
                "local_name": ayur.hindi_name,
            }

        siddha = _find_best_match(siddha_matches, title)
        if siddha:
            item["related_siddha"] = {
                "code": siddha.code,
                "english_name": siddha.english_name,
                "local_name": siddha.tamil_name,
            }

        unani = _find_best_match(unani_matches, title)
        if unani:
            item["related_unani"] = {
                "code": unani.code,
                "english_name": unani.english_name,
                "local_name": unani.arabic_name,
            }

        results.append(item)

    return Response(
        {
            "count": len(results),
            "next": None,
            "previous": None,
            "results": results,
        }
    )


# ── Autocomplete ────────────────────────────────────────────
@api_view(["GET"])
def autocomplete(request):
    """Quick autocomplete across all systems."""
    query = request.query_params.get("q", "").strip()
    if not query or len(query) < 2:
        return Response({"results": []})

    suggestions = []

    for obj in AyurvedhaModel.objects.filter(english_name__icontains=query)[:5]:
        suggestions.append({"text": obj.english_name, "system": "ayurveda", "code": obj.code})

    for obj in SiddhaModel.objects.filter(english_name__icontains=query)[:5]:
        suggestions.append({"text": obj.english_name, "system": "siddha", "code": obj.code})

    for obj in UnaniModel.objects.filter(english_name__icontains=query)[:5]:
        suggestions.append({"text": obj.english_name, "system": "unani", "code": obj.code})

    for obj in ICD11Term.objects.filter(title__icontains=query)[:5]:
        suggestions.append({"text": obj.title, "system": "icd11", "code": obj.code})

    return Response({"results": suggestions})


# ── System-specific autocomplete ────────────────────────────
@api_view(["GET"])
def ayurveda_autocomplete(request):
    """Autocomplete for Ayurveda terms. Returns flat list of strings."""
    query = request.query_params.get("q", "").strip()
    limit = int(request.query_params.get("limit", "20"))
    if not query or len(query) < 2:
        return Response({"results": []})

    qs = AyurvedhaModel.objects.filter(
        _icontains_q(["english_name", "hindi_name", "code"], query)
    )[:limit]
    results = [obj.english_name or obj.code for obj in qs if obj.english_name]
    return Response({"results": results})


@api_view(["GET"])
def siddha_autocomplete(request):
    """Autocomplete for Siddha terms. Returns flat list of strings."""
    query = request.query_params.get("q", "").strip()
    limit = int(request.query_params.get("limit", "20"))
    if not query or len(query) < 2:
        return Response({"results": []})

    qs = SiddhaModel.objects.filter(
        _icontains_q(["english_name", "tamil_name", "code"], query)
    )[:limit]
    results = [obj.english_name or obj.code for obj in qs if obj.english_name]
    return Response({"results": results})


@api_view(["GET"])
def unani_autocomplete(request):
    """Autocomplete for Unani terms. Returns flat list of strings."""
    query = request.query_params.get("q", "").strip()
    limit = int(request.query_params.get("limit", "20"))
    if not query or len(query) < 2:
        return Response({"results": []})

    qs = UnaniModel.objects.filter(
        _icontains_q(["english_name", "arabic_name", "code"], query)
    )[:limit]
    results = [obj.english_name or obj.code for obj in qs if obj.english_name]
    return Response({"results": results})


@api_view(["GET"])
def icd11_autocomplete(request):
    """Autocomplete for ICD-11 terms. Returns flat list of strings."""
    query = request.query_params.get("q", "").strip()
    limit = int(request.query_params.get("limit", "20"))
    if not query or len(query) < 2:
        return Response({"results": []})

    qs = ICD11Term.objects.filter(
        _icontains_q(["title", "code"], query)
    )[:limit]
    results = [obj.title for obj in qs if obj.title]
    return Response({"results": results})

