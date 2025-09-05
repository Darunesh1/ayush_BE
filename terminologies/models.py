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

