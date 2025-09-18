from django.contrib.postgres.indexes import GinIndex
from django.contrib.postgres.search import SearchVectorField
from django.db import models


class BaseNamasteModel(models.Model):
    code = models.CharField(max_length=50, unique=True, db_index=True)
    description = models.TextField(null=True, blank=True)
    english_name = models.CharField(
        max_length=255, null=True, blank=True, db_index=True
    )
    # Pre-computed search vector for full-text search
    search_vector = SearchVectorField(null=True, blank=True)

    class Meta:
        abstract = True
        indexes = [
            # Composite index for common search patterns
            models.Index(fields=["code", "english_name"]),
            # Trigram indexes for fuzzy search on base fields
            GinIndex(
                fields=["code"], name="base_code_trgm", opclasses=["gin_trgm_ops"]
            ),
            GinIndex(
                fields=["english_name"],
                name="base_english_name_trgm",
                opclasses=["gin_trgm_ops"],
            ),
            # Full-text search vector index
            GinIndex(fields=["search_vector"], name="base_search_vector_gin"),
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
            # GIN index for full-text search vector
            GinIndex(fields=["search_vector"], name="ayurveda_search_gin"),
            # Trigram indexes for fuzzy search
            GinIndex(
                fields=["english_name"],
                name="ayurveda_english_name_trgm",
                opclasses=["gin_trgm_ops"],
            ),
            GinIndex(
                fields=["hindi_name"],
                name="ayurveda_hindi_name_trgm",
                opclasses=["gin_trgm_ops"],
            ),
            GinIndex(
                fields=["diacritical_name"],
                name="ayurveda_diacritical_name_trgm",
                opclasses=["gin_trgm_ops"],
            ),
            GinIndex(
                fields=["code"], name="ayurveda_code_trgm", opclasses=["gin_trgm_ops"]
            ),
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
            # Standard B-tree indexes
            models.Index(fields=["tamil_name"]),
            models.Index(fields=["romanized_name"]),
            models.Index(fields=["english_name", "tamil_name"]),
            # GIN index for full-text search vector
            GinIndex(fields=["search_vector"], name="siddha_search_gin"),
            # Trigram indexes for fuzzy search
            GinIndex(
                fields=["english_name"],
                name="siddha_english_name_trgm",
                opclasses=["gin_trgm_ops"],
            ),
            GinIndex(
                fields=["tamil_name"],
                name="siddha_tamil_name_trgm",
                opclasses=["gin_trgm_ops"],
            ),
            GinIndex(
                fields=["romanized_name"],
                name="siddha_romanized_name_trgm",
                opclasses=["gin_trgm_ops"],
            ),
            GinIndex(
                fields=["code"], name="siddha_code_trgm", opclasses=["gin_trgm_ops"]
            ),
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
            # Standard B-tree indexes
            models.Index(fields=["arabic_name"]),
            models.Index(fields=["romanized_name"]),
            models.Index(fields=["english_name", "arabic_name"]),
            # GIN index for full-text search vector
            GinIndex(fields=["search_vector"], name="unani_search_gin"),
            # Trigram indexes for fuzzy search
            GinIndex(
                fields=["english_name"],
                name="unani_english_name_trgm",
                opclasses=["gin_trgm_ops"],
            ),
            GinIndex(
                fields=["arabic_name"],
                name="unani_arabic_name_trgm",
                opclasses=["gin_trgm_ops"],
            ),
            GinIndex(
                fields=["romanized_name"],
                name="unani_romanized_name_trgm",
                opclasses=["gin_trgm_ops"],
            ),
            GinIndex(
                fields=["code"], name="unani_code_trgm", opclasses=["gin_trgm_ops"]
            ),
        ]


class ICD11Term(models.Model):
    # Core identifiers
    foundation_uri = models.URLField(max_length=500, unique=True, db_index=True)
    code = models.CharField(max_length=50, null=True, blank=True, db_index=True)
    title = models.CharField(max_length=255, db_index=True)

    # Clinical definitions
    definition = models.TextField(null=True, blank=True)
    long_definition = models.TextField(null=True, blank=True)

    # JSON fields for all list data
    index_terms = models.JSONField(default=list, blank=True)
    parent = models.JSONField(default=list, blank=True)
    inclusions = models.JSONField(default=list, blank=True)
    exclusions = models.JSONField(default=list, blank=True)
    postcoordination_scales = models.JSONField(default=list, blank=True)
    related_perinatal_entities = models.JSONField(default=list, blank=True)

    # Metadata
    browser_url = models.URLField(max_length=500, blank=True)
    source = models.URLField(max_length=500, blank=True)
    class_kind = models.CharField(max_length=50, blank=True)

    # Search vector
    search_vector = SearchVectorField(null=True, blank=True)

    # Audit fields
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "icd11_terms"
        verbose_name = "ICD-11 Term"
        verbose_name_plural = "ICD-11 Terms"
        indexes = [
            # Standard B-tree indexes for exact lookups
            models.Index(fields=["code"]),
            models.Index(fields=["title"]),
            models.Index(fields=["foundation_uri"]),
            models.Index(fields=["class_kind"]),
            models.Index(fields=["created_at"]),
            # Composite indexes for common search patterns
            models.Index(fields=["code", "title"]),
            models.Index(fields=["class_kind", "created_at"]),
            # GIN indexes for full-text search and JSON fields
            GinIndex(fields=["search_vector"], name="icd11_search_gin"),
            GinIndex(fields=["index_terms"], name="icd11_index_terms_gin"),
            GinIndex(fields=["inclusions"], name="icd11_inclusions_gin"),
            GinIndex(fields=["exclusions"], name="icd11_exclusions_gin"),
            # Trigram indexes for fuzzy search
            GinIndex(
                fields=["title"], name="icd11_title_trgm", opclasses=["gin_trgm_ops"]
            ),
            GinIndex(
                fields=["code"], name="icd11_code_trgm", opclasses=["gin_trgm_ops"]
            ),
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
