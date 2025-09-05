from django.db import models

class BaseNamasteModel(models.Model):
    """
    Abstract base class for NAMASTE terminology entries.
    Shared fields across Siddha, Ayurveda, and Unani datasets.
    """
    sr_no = models.IntegerField()
    entry_id = models.IntegerField()  # NAMC_ID / NUMC_ID (row identifier in XLS)
    medical_code = models.CharField(max_length=50, unique=True)  # NAMC_CODE / NUMC_CODE
    term = models.CharField(max_length=255)
    short_definition = models.TextField(null=True, blank=True)
    long_definition = models.TextField(null=True, blank=True)

    class Meta:
        abstract = True

    def __str__(self):
        return f"{self.medical_code} - {self.term}"


class SiddhaModel(BaseNamasteModel):
    tamil_term = models.CharField(max_length=255, null=True, blank=True)
    reference = models.TextField(null=True, blank=True)

    class Meta:
        db_table = "siddha_terms"


class AyurvedhaModel(BaseNamasteModel):
    diacritical_term = models.CharField(max_length=255, null=True, blank=True)
    devanagari_term = models.CharField(max_length=255, null=True, blank=True)
    ontology_branches = models.TextField(null=True, blank=True)

    class Meta:
        db_table = "ayurvedha_terms"


class UnaniModel(BaseNamasteModel):
    arabic_term = models.CharField(max_length=255, null=True, blank=True)

    class Meta:
        db_table = "unani_terms"
