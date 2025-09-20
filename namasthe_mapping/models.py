import uuid

from django.contrib.contenttypes.fields import GenericForeignKey
from django.contrib.contenttypes.models import ContentType
from django.contrib.postgres.indexes import GinIndex
from django.contrib.postgres.search import SearchVectorField
from django.db import models
from django.utils import timezone


class TerminologyMapping(models.Model):
    """
    Core mapping configuration between NAMASTE systems and ICD-11
    Tracks which terminology systems are being mapped and their settings
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    name = models.CharField(max_length=255, unique=True)
    description = models.TextField(blank=True)

    # Source and target systems
    source_system = models.CharField(
        max_length=200,
        help_text="e.g., NAMASTE-Ayurveda, NAMASTE-Siddha, NAMASTE-Unani",
    )
    target_system = models.CharField(
        max_length=200, default="ICD-11", help_text="Target terminology system"
    )

    # AI configuration for ONNX BioBERT
    biobert_model = models.CharField(
        max_length=200,
        default="nlpie/tiny-biobert",
        help_text="TinyBioBERT model identifier for ONNX",
    )
    similarity_threshold = models.FloatField(
        default=0.75, help_text="Minimum cosine similarity for creating mappings"
    )
    confidence_boost = models.FloatField(
        default=0.05, help_text="Confidence boost for high-quality mappings"
    )

    # Status and workflow
    STATUS_CHOICES = [
        ("draft", "Draft"),
        ("active", "Active"),
        ("review", "Under Review"),
        ("archived", "Archived"),
    ]
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="draft")
    is_active = models.BooleanField(default=True)

    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    created_by = models.CharField(max_length=255, blank=True)
    last_mapping_run = models.DateTimeField(null=True, blank=True)

    # Auto-computed statistics
    total_mappings = models.PositiveIntegerField(default=0)
    validated_mappings = models.PositiveIntegerField(default=0)
    high_confidence_mappings = models.PositiveIntegerField(default=0)
    average_confidence = models.FloatField(default=0.0)
    average_similarity = models.FloatField(default=0.0)

    class Meta:
        db_table = "terminology_mappings"
        verbose_name = "Terminology Mapping"
        verbose_name_plural = "Terminology Mappings"
        indexes = [
            models.Index(fields=["source_system", "target_system"]),
            models.Index(fields=["status", "is_active"]),
            models.Index(fields=["created_at"]),
            models.Index(fields=["last_mapping_run"]),
        ]
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.name} ({self.source_system} → {self.target_system})"

    def update_statistics(self):
        """Update mapping statistics from related ConceptMappings"""
        mappings = self.concept_mappings.all()

        self.total_mappings = mappings.count()
        self.validated_mappings = mappings.filter(is_validated=True).count()
        self.high_confidence_mappings = mappings.filter(
            confidence_score__gte=0.9
        ).count()

        if self.total_mappings > 0:
            # Calculate averages
            stats = mappings.aggregate(
                avg_confidence=models.Avg("confidence_score"),
                avg_similarity=models.Avg("similarity_score"),
            )
            self.average_confidence = round(stats["avg_confidence"] or 0.0, 3)
            self.average_similarity = round(stats["avg_similarity"] or 0.0, 3)
        else:
            self.average_confidence = 0.0
            self.average_similarity = 0.0

        self.last_mapping_run = timezone.now()
        self.save(
            update_fields=[
                "total_mappings",
                "validated_mappings",
                "high_confidence_mappings",
                "average_confidence",
                "average_similarity",
                "last_mapping_run",
            ]
        )

    @property
    def validation_rate(self):
        """Percentage of mappings that are validated"""
        if self.total_mappings == 0:
            return 0.0
        return round((self.validated_mappings / self.total_mappings) * 100, 1)

    @property
    def high_confidence_rate(self):
        """Percentage of high-confidence mappings"""
        if self.total_mappings == 0:
            return 0.0
        return round((self.high_confidence_mappings / self.total_mappings) * 100, 1)


class ConceptMapping(models.Model):
    """
    Individual concept mapping between a NAMASTE term and an ICD-11 term
    Stores the AI-generated mapping with similarity scores and validation status
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    # Link to the mapping configuration
    mapping = models.ForeignKey(
        TerminologyMapping, on_delete=models.CASCADE, related_name="concept_mappings"
    )

    # Source concept (polymorphic - can be Ayurveda, Siddha, or Unani)
    source_content_type = models.ForeignKey(ContentType, on_delete=models.CASCADE)
    source_object_id = models.PositiveIntegerField()
    source_concept = GenericForeignKey("source_content_type", "source_object_id")

    # Target concept (always ICD-11 from terminologies app)
    target_concept = models.ForeignKey(
        "terminologies.ICD11Term",
        on_delete=models.CASCADE,
        related_name="concept_mappings",
    )

    # FHIR-compatible mapping relationships
    RELATIONSHIP_CHOICES = [
        ("equivalent", "Equivalent"),  # 1:1 exact match
        ("related-to", "Related To"),  # General relationship
        (
            "source-is-narrower-than-target",
            "Source Is Narrower",
        ),  # Source more specific
        ("source-is-broader-than-target", "Source Is Broader"),  # Source more general
        ("not-related-to", "Not Related"),  # Negative mapping
    ]
    relationship = models.CharField(
        max_length=35, choices=RELATIONSHIP_CHOICES, default="related-to"
    )

    # TinyBioBERT ONNX results
    similarity_score = models.FloatField(
        help_text="Cosine similarity score from BioBERT embeddings (0.0 to 1.0)"
    )
    confidence_score = models.FloatField(
        help_text="Adjusted confidence score with domain-specific boosts"
    )

    # Store embeddings for performance (avoid re-computation)
    source_embedding = models.JSONField(
        null=True, blank=True, help_text="768-dimensional source embedding from BioBERT"
    )
    target_embedding = models.JSONField(
        null=True, blank=True, help_text="768-dimensional target embedding from BioBERT"
    )

    # Expert validation workflow
    is_validated = models.BooleanField(
        default=False, help_text="Has this mapping been reviewed by a domain expert?"
    )
    validation_notes = models.TextField(
        blank=True, help_text="Expert notes on mapping quality or corrections needed"
    )
    validated_by = models.CharField(
        max_length=255,
        blank=True,
        help_text="Username of expert who validated this mapping",
    )
    validated_at = models.DateTimeField(null=True, blank=True)

    # Quality and provenance tracking
    METHOD_CHOICES = [
        ("onnx_biobert", "ONNX TinyBioBERT Automatic"),
        ("manual", "Manual Entry"),
        ("hybrid", "AI + Manual Review"),
        ("imported", "Imported from External Source"),
    ]
    mapping_method = models.CharField(
        max_length=20, choices=METHOD_CHOICES, default="onnx_biobert"
    )

    # Flags for mapping quality
    is_high_confidence = models.BooleanField(
        default=False, help_text="Automatically set for mappings with confidence >= 0.9"
    )
    needs_review = models.BooleanField(
        default=True, help_text="Flag for mappings that need expert review"
    )
    has_issues = models.BooleanField(
        default=False, help_text="Flag for problematic mappings"
    )

    # Full-text search capabilities
    search_vector = SearchVectorField(null=True, blank=True)

    # Audit and timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "concept_mappings"
        verbose_name = "Concept Mapping"
        verbose_name_plural = "Concept Mappings"

        # Ensure no duplicate mappings
        unique_together = [
            ("mapping", "source_content_type", "source_object_id", "target_concept")
        ]

        indexes = [
            # Performance indexes for common queries
            models.Index(fields=["mapping", "is_validated"]),
            models.Index(fields=["mapping", "relationship"]),
            models.Index(fields=["similarity_score"]),
            models.Index(fields=["confidence_score"]),
            models.Index(fields=["is_high_confidence"]),
            models.Index(fields=["needs_review"]),
            models.Index(fields=["mapping_method"]),
            models.Index(fields=["created_at"]),
            # Composite indexes for complex queries
            models.Index(fields=["mapping", "is_validated", "confidence_score"]),
            models.Index(fields=["source_content_type", "mapping"]),
            # Full-text search
            GinIndex(fields=["search_vector"], name="concept_mapping_search_gin"),
        ]

        ordering = ["-confidence_score", "-similarity_score"]

    def __str__(self):
        return f"{self.get_source_display()} → {self.get_target_display()} ({self.confidence_score:.3f})"

    def save(self, *args, **kwargs):
        """Override save to automatically set quality flags"""
        # Set high confidence flag
        self.is_high_confidence = self.confidence_score >= 0.9

        # Set needs review based on confidence and method
        if self.mapping_method == "manual":
            self.needs_review = False
        elif self.is_validated:
            self.needs_review = False
        elif self.confidence_score >= 0.95:
            self.needs_review = False  # Very high confidence
        else:
            self.needs_review = True

        super().save(*args, **kwargs)

    # Helper methods for display
    def get_source_code(self):
        """Get the source concept code"""
        return getattr(self.source_concept, "code", "") if self.source_concept else ""

    def get_source_display(self):
        """Get human-readable source concept name"""
        if not self.source_concept:
            return "Unknown Source"
        return getattr(self.source_concept, "english_name", str(self.source_concept))

    def get_target_code(self):
        """Get the target ICD-11 code"""
        return (
            self.target_concept.code
            if self.target_concept and self.target_concept.code
            else ""
        )

    def get_target_display(self):
        """Get human-readable target concept name"""
        return self.target_concept.title if self.target_concept else "Unknown Target"

    def get_source_system_display(self):
        """Get the source terminology system name"""
        if not self.source_concept:
            return "Unknown"

        model_name = self.source_content_type.model
        if model_name == "ayurvedha":
            return "Ayurveda"
        elif model_name == "siddha":
            return "Siddha"
        elif model_name == "unani":
            return "Unani"
        else:
            return model_name.title()

    # Validation methods
    def mark_as_validated(self, user_name: str, notes: str = ""):
        """Mark this mapping as validated by an expert"""
        self.is_validated = True
        self.validated_by = user_name
        self.validated_at = timezone.now()
        self.validation_notes = notes
        self.needs_review = False
        self.save(
            update_fields=[
                "is_validated",
                "validated_by",
                "validated_at",
                "validation_notes",
                "needs_review",
            ]
        )

    def flag_for_review(self, reason: str = ""):
        """Flag this mapping for expert review"""
        self.needs_review = True
        self.has_issues = True
        if reason:
            existing_notes = self.validation_notes
            self.validation_notes = (
                f"{existing_notes}\n\nFlagged for review: {reason}".strip()
            )
        self.save(update_fields=["needs_review", "has_issues", "validation_notes"])

    @property
    def confidence_category(self):
        """Categorize mapping confidence for display"""
        if self.confidence_score >= 0.95:
            return "Very High"
        elif self.confidence_score >= 0.85:
            return "High"
        elif self.confidence_score >= 0.75:
            return "Medium"
        elif self.confidence_score >= 0.65:
            return "Low"
        else:
            return "Very Low"

    @property
    def quality_score(self):
        """Combined quality score considering similarity, confidence, and validation"""
        base_score = (self.similarity_score + self.confidence_score) / 2

        if self.is_validated:
            base_score += 0.1  # Boost for expert validation

        if self.mapping_method == "manual":
            base_score += 0.05  # Boost for manual curation

        return min(base_score, 1.0)


class MappingAudit(models.Model):
    """
    Audit trail for mapping changes to meet regulatory requirements
    Tracks all changes to ConceptMappings for compliance and debugging
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    # Link to the mapping that was changed
    concept_mapping = models.ForeignKey(
        ConceptMapping, on_delete=models.CASCADE, related_name="audit_entries"
    )

    # Change details
    ACTION_CHOICES = [
        ("create", "Created"),
        ("update", "Updated"),
        ("validate", "Validated"),
        ("flag", "Flagged for Review"),
        ("delete", "Deleted"),
    ]
    action = models.CharField(max_length=20, choices=ACTION_CHOICES)

    # What changed
    field_changes = models.JSONField(
        default=dict,
        help_text="JSON object showing old_value -> new_value for each changed field",
    )

    # Why it changed
    reason = models.TextField(
        blank=True, help_text="Reason for the change (user input or system note)"
    )

    # Who made the change
    user_name = models.CharField(max_length=255, blank=True)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.TextField(blank=True)

    # When it changed
    timestamp = models.DateTimeField(auto_now_add=True)

    # System context
    system_version = models.CharField(max_length=100, blank=True)
    biobert_model_version = models.CharField(max_length=200, blank=True)

    class Meta:
        db_table = "mapping_audit"
        verbose_name = "Mapping Audit Entry"
        verbose_name_plural = "Mapping Audit Entries"
        indexes = [
            models.Index(fields=["concept_mapping", "timestamp"]),
            models.Index(fields=["action", "timestamp"]),
            models.Index(fields=["user_name", "timestamp"]),
        ]
        ordering = ["-timestamp"]

    def __str__(self):
        return f"{self.action.title()} mapping {self.concept_mapping.id} at {self.timestamp}"
