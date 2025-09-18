from django.contrib.postgres.indexes import GinIndex
from django.contrib.postgres.search import SearchVectorField
from django.db import models


class BaseNamasteModel(models.Model):
    code = models.CharField(max_length=50, unique=True, db_index=True)
    description = models.TextField(null=True, blank=True)
    english_name = models.CharField(
        max_length=255, null=True, blank=True, db_index=True
    )

    # Optional: Pre-computed search vector for full-text search
    search_vector = SearchVectorField(null=True, blank=True)

    class Meta:
        abstract = True
        indexes = [
            # Composite index for common search patterns
            models.Index(fields=["code", "english_name"]),
        ]

    def __str__(self):
        return f"{self.code} - {self.english_name}"


class Ayurvedha(BaseNamasteModel):
    hindi_name = models.CharField(max_length=255, null=True, blank=True, db_index=True)
    diacritical_name = models.CharField(
        max_length=255, null=True, blank=True, db_index=True
    )

    class Meta:
        db_table = "ayurveda_terms"
        verbose_name = "Ayurveda Term"
        verbose_name_plural = "Ayurveda Terms"
        indexes = [
            # Standard B-tree indexes for exact lookups
            models.Index(fields=["hindi_name"]),
            models.Index(fields=["diacritical_name"]),
            models.Index(fields=["english_name", "hindi_name"]),
            # GIN index for search vector
            GinIndex(fields=["search_vector"], name="ayurveda_search_gin"),
        ]


class Siddha(BaseNamasteModel):
    tamil_name = models.CharField(max_length=255, null=True, blank=True, db_index=True)
    romanized_name = models.CharField(
        max_length=255, null=True, blank=True, db_index=True
    )
    reference = models.TextField(null=True, blank=True)

    class Meta:
        db_table = "siddha_terms"
        verbose_name = "Siddha Term"
        verbose_name_plural = "Siddha Terms"
        indexes = [
            models.Index(fields=["tamil_name"]),
            models.Index(fields=["romanized_name"]),
            models.Index(fields=["english_name", "tamil_name"]),
            GinIndex(fields=["search_vector"], name="siddha_search_gin"),
        ]


class Unani(BaseNamasteModel):
    arabic_name = models.CharField(max_length=255, null=True, blank=True, db_index=True)
    romanized_name = models.CharField(
        max_length=255, null=True, blank=True, db_index=True
    )
    reference = models.TextField(null=True, blank=True)

    class Meta:
        db_table = "unani_terms"
        verbose_name = "Unani Term"
        verbose_name_plural = "Unani Terms"
        indexes = [
            models.Index(fields=["arabic_name"]),
            models.Index(fields=["romanized_name"]),
            models.Index(fields=["english_name", "arabic_name"]),
            GinIndex(fields=["search_vector"], name="unani_search_gin"),
        ]


class ICD11Term(models.Model):
    foundation_uri = models.URLField(max_length=500, unique=True)
    code = models.CharField(max_length=50, null=True, blank=True, db_index=True)
    title = models.CharField(max_length=255, db_index=True)
    definition = models.TextField(null=True, blank=True)
    long_definition = models.TextField(null=True, blank=True)

    # Search vector - will be maintained by database trigger
    search_vector = SearchVectorField(null=True, blank=True)

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "icd_terms"
        verbose_name = "ICD Term"
        verbose_name_plural = "ICD Terms"
        indexes = [
            models.Index(fields=["code"]),
            models.Index(fields=["title"]),
            models.Index(fields=["created_at"]),
            GinIndex(fields=["search_vector"], name="icd_search_gin"),
        ]

    def __str__(self):
        return f"{self.code} - {self.title}" if self.code else self.title


class TermMapping(models.Model):
    """Store mappings starting from NAMASTE terms to ICD-11"""

    # Primary NAMASTE term (the one we're mapping from)
    primary_ayurveda_term = models.ForeignKey(
        Ayurvedha, on_delete=models.CASCADE, null=True, blank=True
    )
    primary_siddha_term = models.ForeignKey(
        Siddha, on_delete=models.CASCADE, null=True, blank=True
    )
    primary_unani_term = models.ForeignKey(
        Unani, on_delete=models.CASCADE, null=True, blank=True
    )

    # Matched ICD-11 term
    icd_term = models.ForeignKey(
        ICD11Term, on_delete=models.CASCADE, related_name="namaste_mappings"
    )
    icd_similarity = models.FloatField()

    # Cross-system matches (if the same concept exists in other NAMASTE systems)
    cross_ayurveda_term = models.ForeignKey(
        Ayurvedha,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="cross_mappings",
    )
    cross_siddha_term = models.ForeignKey(
        Siddha,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="cross_mappings",
    )
    cross_unani_term = models.ForeignKey(
        Unani,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="cross_mappings",
    )

    # Cross-system similarity scores
    cross_ayurveda_similarity = models.FloatField(null=True, blank=True)
    cross_siddha_similarity = models.FloatField(null=True, blank=True)
    cross_unani_similarity = models.FloatField(null=True, blank=True)

    # Source system identifier
    SOURCE_CHOICES = [
        ("ayurveda", "Ayurveda"),
        ("siddha", "Siddha"),
        ("unani", "Unani"),
    ]
    source_system = models.CharField(max_length=20, choices=SOURCE_CHOICES)

    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    confidence_score = models.FloatField(default=0.0)

    class Meta:
        db_table = "namaste_to_icd_mappings"
        indexes = [
            models.Index(fields=["source_system"]),
            models.Index(fields=["confidence_score"]),
            models.Index(fields=["primary_ayurveda_term"]),
            models.Index(fields=["primary_siddha_term"]),
            models.Index(fields=["primary_unani_term"]),
        ]
        # Ensure uniqueness per source term
        unique_together = [
            ("primary_ayurveda_term", "icd_term"),
            ("primary_siddha_term", "icd_term"),
            ("primary_unani_term", "icd_term"),
        ]
