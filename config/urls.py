from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.http import HttpResponse
from django.urls import include, path


def home(request):
    return HttpResponse("🚀 AyushSync Backend is running!")


urlpatterns = [
    path("", home, name="home"),  # 👈 Root URL
    path("admin/", admin.site.urls),
    path("fhir/", include("fhir.urls")),
    path("auth/", include("auth_abha.urls")),
    path("analytics/", include("analytics.urls")),
    path("terminologies/", include("terminologies.urls")),
]

if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
