from django.db import models

class BaseNamasteModel(models.Model):
    code = models.CharField(max_length=50, unique=True)  # Unique code (e.g., NAMC_CODE)
    description = models.TextField(null=True, blank=True)
    english_name = models.CharField(max_length=255, null=True, blank=True)
    

    class Meta:
        abstract = True
        
    def __str__(self):
        return f"{self.code} - {self.term}"


class AyurvedhaModel(BaseNamasteModel):
    hindi_name = models.CharField(max_length=255, null=True, blank=True)
    diacritical_name = models.CharField(max_length=255, null=True, blank=True)
    

class SiddhaModel(BaseNamasteModel):
    tamil_name = models.CharField(max_length=255, null=True, blank=True)
    romanized_name = models.CharField(max_length=255, null=True, blank=True)
    reference = models.TextField(null=True, blank=True)

class UnaniModel(BaseNamasteModel):
    arabic_name = models.CharField(max_length=255, null=True, blank=True)
    romanized_name = models.CharField(max_length=255, null=True, blank=True)
    reference = models.TextField(null=True, blank=True)
    
from django.db import models

class ICDClassKind(models.Model):
    """
    Represents the type of the ICD entity (chapter, block, category, etc.).
    """
    name = models.CharField(max_length=50, unique=True)  # e.g., chapter, block, category
    description = models.TextField(null=True, blank=True)

    def __str__(self):
        return self.name


class ICD11Term(models.Model):
    """
    Represents a single ICD-11 entity (chapter, block, or category) without block/grouping details.
    """
    foundation_uri = models.URLField(max_length=500, unique=True)
    linearization_uri = models.URLField(max_length=500, null=True, blank=True)
    code = models.CharField(max_length=50, null=True, blank=True)  # e.g., 1.0A00
    title = models.CharField(max_length=255)
    class_kind = models.ForeignKey(ICDClassKind, on_delete=models.PROTECT)
    depth_in_kind = models.PositiveIntegerField(null=True, blank=True)
    is_residual = models.BooleanField(default=False)
    primary_location = models.CharField(max_length=255, null=True, blank=True)
    chapter_no = models.CharField(max_length=50, null=True, blank=True)
    browser_link = models.URLField(max_length=500, null=True, blank=True)
    icat_link = models.URLField(max_length=500, null=True, blank=True)
    is_leaf = models.BooleanField(default=False)
    no_of_non_residual_children = models.PositiveIntegerField(null=True, blank=True)
    version_date = models.DateField(null=True, blank=True)

    class Meta:
        db_table = "icd11_terms"

    def __str__(self):
        return f"{self.code} - {self.title}" if self.code else self.title


