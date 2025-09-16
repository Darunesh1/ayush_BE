from rest_framework import serializers

from .models import Ayurvedha, ICD11Synonym, ICD11Term, Siddha, TermMapping, Unani


class ICD11SynonymSerializer(serializers.ModelSerializer):
    """
    Serializer for ICD11Synonym model representing alternative labels.
    """

    class Meta:
        model = ICD11Synonym
        fields = [
            "id",
            "label",
        ]
        read_only_fields = ["id"]


class ICD11TermSerializer(serializers.ModelSerializer):
    """
    Complete serializer for ICD11Term model.
    Includes nested read-only synonyms.
    """

    synonyms = ICD11SynonymSerializer(many=True, read_only=True)
    display_name = serializers.SerializerMethodField()

    class Meta:
        model = ICD11Term
        fields = [
            "id",
            "foundation_uri",
            "code",
            "title",
            "synonyms",
            "display_name",
        ]
        read_only_fields = ["id", "display_name"]

    def get_display_name(self, obj):
        """
        Get formatted display name for ICD11Term instance.
        """
        if obj.code:
            return f"{obj.code} - {obj.title}"
        return obj.title

    def validate_foundation_uri(self, value):
        """
        Validate that foundation_uri is required and unique.
        """
        if not value:
            raise serializers.ValidationError("Foundation URI is required.")
        if self.instance and self.instance.foundation_uri == value:
            return value
        if ICD11Term.objects.filter(foundation_uri=value).exists():
            raise serializers.ValidationError(
                "An ICD-11 term with this foundation URI already exists."
            )
        return value

    def validate_title(self, value):
        """
        Validate that title is not empty or whitespace only.
        """
        if not value or not value.strip():
            raise serializers.ValidationError("Title cannot be empty.")
        return value.strip()

    def validate_code(self, value):
        """
        Normalize code to None if empty string.
        """
        if value and not value.strip():
            return None
        return value


class ICD11TermListSerializer(serializers.ModelSerializer):
    """
    Lightweight serializer for list views of ICD11Term.
    """

    class Meta:
        model = ICD11Term
        fields = [
            "id",
            "foundation_uri",
            "code",
            "title",
        ]
        read_only_fields = fields


class AyurvedhaSerializer(serializers.ModelSerializer):
    """
    Complete serializer for AyurvedhaModel.
    Handles Ayurveda medicine terminology from NAMASTE codes.
    """

    display_name = serializers.SerializerMethodField()

    class Meta:
        model = Ayurvedha
        fields = [
            "id",
            # BaseNamasteModel fields
            "code",
            "description",
            "english_name",
            # AyurvedhaModel specific fields
            "hindi_name",
            "diacritical_name",
            "display_name",
        ]
        read_only_fields = ["id", "display_name"]

    def get_display_name(self, obj):
        """
        Get formatted display name for Ayurveda term.
        """
        if obj.code:
            return (
                f"{obj.code} - {obj.english_name or obj.hindi_name or 'Unnamed Term'}"
            )
        return obj.english_name or obj.hindi_name or "Unnamed Term"

    def validate_code(self, value):
        """
        Validate code is unique and not empty.
        """
        if not value or len(value.strip()) == 0:
            raise serializers.ValidationError("Code cannot be empty.")

        # Check uniqueness
        if self.instance and self.instance.code == value:
            return value

        if Ayurvedha.objects.filter(code=value).exists():  # type: ignore
            raise serializers.ValidationError("A term with this code already exists.")

        return value.strip()

    def validate_english_name(self, value):
        """
        Validate English name format.
        """
        if value and len(value.strip()) == 0:
            raise serializers.ValidationError(
                "English name cannot be empty if provided."
            )
        return value

    def validate_hindi_name(self, value):
        """
        Validate Hindi name format.
        """
        if value and len(value.strip()) == 0:
            raise serializers.ValidationError("Hindi name cannot be empty if provided.")
        return value

    def validate_diacritical_name(self, value):
        """
        Validate diacritical name format.
        """
        if value and len(value.strip()) == 0:
            raise serializers.ValidationError(
                "Diacritical name cannot be empty if provided."
            )
        return value

    def validate(self, attrs):
        """
        Cross-field validation for Ayurveda terms.
        """
        # At least one name should be provided
        english_name = attrs.get("english_name")
        hindi_name = attrs.get("hindi_name")
        diacritical_name = attrs.get("diacritical_name")

        if not any([english_name, hindi_name, diacritical_name]):
            raise serializers.ValidationError(
                "At least one name (English, Hindi, or Diacritical) must be provided."
            )

        return attrs


class AyurvedhaListSerializer(serializers.ModelSerializer):
    """
    Lightweight serializer for AyurvedhaModel list views.
    Only includes essential fields for listing Ayurveda medicine terminology.
    """

    class Meta:
        model = Ayurvedha
        fields = [
            "id",
            "code",
            "english_name",
            "hindi_name",
            "diacritical_name",
        ]
        read_only_fields = [
            "id",
            "code",
            "english_name",
            "hindi_name",
            "diacritical_name",
        ]


class SiddhaSerializer(serializers.ModelSerializer):
    """
    Complete serializer for SiddhaModel.
    Handles Siddha medicine terminology from NAMASTE codes.
    """

    display_name = serializers.SerializerMethodField()

    class Meta:
        model = Siddha
        fields = [
            "id",
            # BaseNamasteModel fields
            "code",
            "description",
            "english_name",
            # SiddhaModel specific fields
            "tamil_name",
            "romanized_name",
            "reference",
            "display_name",
        ]
        read_only_fields = ["id", "display_name"]

    def get_display_name(self, obj):
        """
        Get formatted display name for Siddha term.
        """
        if obj.code:
            return (
                f"{obj.code} - {obj.english_name or obj.tamil_name or 'Unnamed Term'}"
            )
        return obj.english_name or obj.tamil_name or "Unnamed Term"

    def validate_code(self, value):
        """
        Validate code is unique and not empty.
        """
        if not value or len(value.strip()) == 0:
            raise serializers.ValidationError("Code cannot be empty.")

        # Check uniqueness
        if self.instance and self.instance.code == value:
            return value

        if Siddha.objects.filter(code=value).exists():  # type: ignore
            raise serializers.ValidationError("A term with this code already exists.")

        return value.strip()

    def validate_english_name(self, value):
        """
        Validate English name format.
        """
        if value and len(value.strip()) == 0:
            raise serializers.ValidationError(
                "English name cannot be empty if provided."
            )
        return value

    def validate_tamil_name(self, value):
        """
        Validate Tamil name format.
        """
        if value and len(value.strip()) == 0:
            raise serializers.ValidationError("Tamil name cannot be empty if provided.")
        return value

    def validate_romanized_name(self, value):
        """
        Validate romanized name format.
        """
        if value and len(value.strip()) == 0:
            raise serializers.ValidationError(
                "Romanized name cannot be empty if provided."
            )
        return value

    def validate_reference(self, value):
        """
        Validate reference format.
        """
        if value and len(value.strip()) == 0:
            return None
        return value

    def validate(self, attrs):
        """
        Cross-field validation for Siddha terms.
        """
        # At least one name should be provided
        english_name = attrs.get("english_name")
        tamil_name = attrs.get("tamil_name")

        if not any([english_name, tamil_name]):
            raise serializers.ValidationError(
                "At least one name (English or Tamil) must be provided."
            )

        return attrs


class SiddhaListSerializer(serializers.ModelSerializer):
    """
    Lightweight serializer for Siddha model list views.
    Only includes essential fields for listing Siddha medicine terminology.
    """

    class Meta:
        model = Siddha
        fields = [
            "id",
            "code",
            "english_name",
            "tamil_name",
            "romanized_name",
        ]
        read_only_fields = [
            "id",
            "code",
            "english_name",
            "tamil_name",
            "romanized_name",
        ]


class UnaniSerializer(serializers.ModelSerializer):
    """
    Complete serializer for UnaniModel.
    Handles Unani medicine terminology from NAMASTE codes.
    """

    display_name = serializers.SerializerMethodField()

    class Meta:
        model = Unani
        fields = [
            "id",
            # BaseNamasteModel fields
            "code",
            "description",
            "english_name",
            # UnaniModel specific fields
            "arabic_name",
            "romanized_name",
            "reference",
            "display_name",
        ]
        read_only_fields = ["id", "display_name"]

    def get_display_name(self, obj):
        """
        Get formatted display name for Unani term.
        """
        if obj.code:
            return (
                f"{obj.code} - {obj.english_name or obj.arabic_name or 'Unnamed Term'}"
            )
        return obj.english_name or obj.arabic_name or "Unnamed Term"

    def validate_code(self, value):
        """
        Validate code is unique and not empty.
        """
        if not value or len(value.strip()) == 0:
            raise serializers.ValidationError("Code cannot be empty.")

        # Check uniqueness
        if self.instance and self.instance.code == value:
            return value

        if Unani.objects.filter(code=value).exists():  # type: ignore
            raise serializers.ValidationError("A term with this code already exists.")

        return value.strip()

    def validate_english_name(self, value):
        """
        Validate English name format.
        """
        if value and len(value.strip()) == 0:
            raise serializers.ValidationError(
                "English name cannot be empty if provided."
            )
        return value

    def validate_arabic_name(self, value):
        """
        Validate Arabic name format.
        """
        if value and len(value.strip()) == 0:
            raise serializers.ValidationError(
                "Arabic name cannot be empty if provided."
            )
        return value

    def validate_romanized_name(self, value):
        """
        Validate romanized name format.
        """
        if value and len(value.strip()) == 0:
            raise serializers.ValidationError(
                "Romanized name cannot be empty if provided."
            )
        return value

    def validate_reference(self, value):
        """
        Validate reference format.
        """
        if value and len(value.strip()) == 0:
            return None
        return value

    def validate(self, attrs):
        """
        Cross-field validation for Unani terms.
        """
        # At least one name should be provided
        english_name = attrs.get("english_name")
        arabic_name = attrs.get("arabic_name")

        if not any([english_name, arabic_name]):
            raise serializers.ValidationError(
                "At least one name (English or Arabic) must be provided."
            )

        return attrs


class UnaniListSerializer(serializers.ModelSerializer):
    """
    Lightweight serializer for Unani model list views.
    Only includes essential fields for listing Unani medicine terminology.
    """

    class Meta:
        model = Unani
        fields = [
            "id",
            "code",
            "english_name",
            "arabic_name",
            "romanized_name",
        ]
        read_only_fields = [
            "id",
            "code",
            "english_name",
            "arabic_name",
            "romanized_name",
        ]


class ICD11TermSummarySerializer(serializers.ModelSerializer):
    """Lightweight ICD term for listings"""

    class Meta:
        model = ICD11Term
        fields = ["code", "title", "chapter_no"]


# Dynamic NAMASTE term serializer
class NamasteTermSerializer(serializers.Serializer):
    """Dynamic serializer that works with any NAMASTE system"""

    system = serializers.CharField()
    code = serializers.CharField()
    english_name = serializers.CharField()
    description = serializers.CharField(allow_null=True)

    # System-specific fields (populated dynamically)
    hindi_name = serializers.CharField(allow_null=True, required=False)
    diacritical_name = serializers.CharField(allow_null=True, required=False)
    tamil_name = serializers.CharField(allow_null=True, required=False)
    arabic_name = serializers.CharField(allow_null=True, required=False)
    romanized_name = serializers.CharField(allow_null=True, required=False)
    reference = serializers.CharField(allow_null=True, required=False)


# Cross-system match serializers
class CrossSystemMatchSerializer(serializers.Serializer):
    """Serializer for cross-system matches"""

    code = serializers.CharField()
    english_name = serializers.CharField()
    similarity_score = serializers.FloatField()


class DetailedCrossSystemMatchSerializer(CrossSystemMatchSerializer):
    """Detailed cross-system match with all fields"""

    description = serializers.CharField(allow_null=True)

    # System-specific fields (will be populated based on system)
    hindi_name = serializers.CharField(allow_null=True, required=False)
    diacritical_name = serializers.CharField(allow_null=True, required=False)
    tamil_name = serializers.CharField(allow_null=True, required=False)
    arabic_name = serializers.CharField(allow_null=True, required=False)
    romanized_name = serializers.CharField(allow_null=True, required=False)
    reference = serializers.CharField(allow_null=True, required=False)


# Term mapping serializers
class TermMappingSummarySerializer(serializers.ModelSerializer):
    """Summary serializer for listings"""

    source_term = serializers.SerializerMethodField()
    icd_term = ICD11TermSummarySerializer(read_only=True)
    confidence_score = serializers.FloatField()
    icd_similarity = serializers.FloatField()

    class Meta:
        model = TermMapping
        fields = [
            "id",
            "source_system",
            "source_term",
            "icd_term",
            "confidence_score",
            "icd_similarity",
            "created_at",
        ]

    def get_source_term(self, obj):
        """Get the primary source term dynamically"""
        source_term = (
            obj.primary_ayurveda_term
            or obj.primary_siddha_term
            or obj.primary_unani_term
        )

        if source_term:
            return {"code": source_term.code, "english_name": source_term.english_name}
        return None


class TermMappingDetailSerializer(serializers.ModelSerializer):
    """Detailed mapping with full information"""

    source_term = serializers.SerializerMethodField()
    icd_mapping = serializers.SerializerMethodField()
    cross_system_matches = serializers.SerializerMethodField()
    confidence_score = serializers.FloatField()
    created_at = serializers.DateTimeField()

    class Meta:
        model = TermMapping
        fields = [
            "id",
            "source_system",
            "source_term",
            "icd_mapping",
            "cross_system_matches",
            "confidence_score",
            "created_at",
        ]

    def get_source_term(self, obj):
        """Get detailed source term information"""
        source_term = (
            obj.primary_ayurveda_term
            or obj.primary_siddha_term
            or obj.primary_unani_term
        )

        if not source_term:
            return None

        base_data = {
            "system": obj.source_system,
            "code": source_term.code,
            "english_name": source_term.english_name,
            "description": source_term.description,
        }

        # Add system-specific fields
        if obj.source_system == "ayurveda":
            base_data.update(
                {
                    "hindi_name": source_term.hindi_name,
                    "diacritical_name": source_term.diacritical_name,
                }
            )
        elif obj.source_system == "siddha":
            base_data.update(
                {
                    "tamil_name": source_term.tamil_name,
                    "romanized_name": source_term.romanized_name,
                    "reference": source_term.reference,
                }
            )
        elif obj.source_system == "unani":
            base_data.update(
                {
                    "arabic_name": source_term.arabic_name,
                    "romanized_name": source_term.romanized_name,
                    "reference": source_term.reference,
                }
            )

        return base_data

    def get_icd_mapping(self, obj):
        """Get ICD mapping with similarity score"""
        icd_data = ICD11TermSerializer(obj.icd_term).data
        icd_data["similarity_score"] = round(obj.icd_similarity, 3)
        return icd_data

    def get_cross_system_matches(self, obj):
        """Get cross-system matches"""
        cross_matches = {}

        if obj.cross_ayurveda_term:
            cross_matches["ayurveda"] = self._serialize_cross_match(
                obj.cross_ayurveda_term, obj.cross_ayurveda_similarity, "ayurveda"
            )

        if obj.cross_siddha_term:
            cross_matches["siddha"] = self._serialize_cross_match(
                obj.cross_siddha_term, obj.cross_siddha_similarity, "siddha"
            )

        if obj.cross_unani_term:
            cross_matches["unani"] = self._serialize_cross_match(
                obj.cross_unani_term, obj.cross_unani_similarity, "unani"
            )

        return cross_matches

    def _serialize_cross_match(self, term, similarity, system):
        """Helper to serialize cross-system match"""
        base_data = {
            "code": term.code,
            "english_name": term.english_name,
            "description": term.description,
            "similarity_score": round(similarity, 3),
        }

        # Add system-specific fields
        if system == "ayurveda":
            base_data.update(
                {
                    "hindi_name": term.hindi_name,
                    "diacritical_name": term.diacritical_name,
                }
            )
        elif system == "siddha":
            base_data.update(
                {
                    "tamil_name": term.tamil_name,
                    "romanized_name": term.romanized_name,
                    "reference": term.reference,
                }
            )
        elif system == "unani":
            base_data.update(
                {
                    "arabic_name": term.arabic_name,
                    "romanized_name": term.romanized_name,
                    "reference": term.reference,
                }
            )

        return base_data


class TermMappingSearchSerializer(serializers.ModelSerializer):
    """Serializer for search results"""

    source_term = serializers.SerializerMethodField()
    icd_term = ICD11TermSummarySerializer(read_only=True)
    confidence_score = serializers.FloatField()
    icd_similarity = serializers.FloatField()
    has_cross_matches = serializers.SerializerMethodField()

    class Meta:
        model = TermMapping
        fields = [
            "id",
            "source_system",
            "source_term",
            "icd_term",
            "confidence_score",
            "icd_similarity",
            "has_cross_matches",
            "created_at",
        ]

    def get_source_term(self, obj):
        source_term = (
            obj.primary_ayurveda_term
            or obj.primary_siddha_term
            or obj.primary_unani_term
        )

        if source_term:
            return {
                "code": source_term.code,
                "english_name": source_term.english_name,
                "description": source_term.description,
            }
        return None

    def get_has_cross_matches(self, obj):
        return any(
            [obj.cross_ayurveda_term, obj.cross_siddha_term, obj.cross_unani_term]
        )


# Statistics serializers
class MappingStatsSerializer(serializers.Serializer):
    """Serializer for mapping statistics"""

    total_mappings = serializers.IntegerField()
    by_system = serializers.DictField()
    confidence_distribution = serializers.DictField()
    top_icd_matches = serializers.ListField()
    recent_mappings = serializers.ListField()


class TopICDMatchSerializer(serializers.Serializer):
    """Serializer for top ICD matches statistics"""

    icd_term__code = serializers.CharField()
    icd_term__title = serializers.CharField()
    mapping_count = serializers.IntegerField()


class RecentMappingSerializer(serializers.Serializer):
    """Serializer for recent mappings statistics"""

    source_system = serializers.CharField()
    source_term = serializers.CharField()
    icd_title = serializers.CharField()
    confidence_score = serializers.FloatField()
    created_at = serializers.DateTimeField()
