from rest_framework import serializers

from .models import AyurvedhaModel, ICD11Term, ICDClassKind, SiddhaModel, UnaniModel


class ICDClassKindSerializer(serializers.ModelSerializer):
    """
    Complete serializer for ICDClassKind model.
    Handles ICD-11 classification categories.
    """

    class Meta:
        model = ICDClassKind
        fields = [
            "id",
            "name",
            "description",
        ]
        read_only_fields = ["id"]

    def validate_name(self, value):
        """
        Ensure name is unique and not empty.
        """
        if not value or len(value.strip()) == 0:
            raise serializers.ValidationError("Name cannot be empty.")

        # Check uniqueness only if this is a new instance or name is being changed
        if self.instance and self.instance.name == value:
            return value

        if ICDClassKind.objects.filter(name=value).exists():  # type: ignore
            raise serializers.ValidationError(
                "A class kind with this name already exists."
            )

        return value.strip()

    def validate_description(self, value):
        """
        Validate description field.
        """
        if value and len(value.strip()) == 0:
            return None
        return value


class ICD11TermSerializer(serializers.ModelSerializer):
    """
    Complete serializer for ICD11Term model.
    Handles ICD-11 Traditional Medicine Module 2 (TM2) and Biomedicine terms.
    """

    # Nested serializer for read operations
    class_kind = ICDClassKindSerializer(read_only=True)

    # Separate field for write operations
    class_kind_id = serializers.PrimaryKeyRelatedField(
        queryset=ICDClassKind.objects.all(),  # type: ignore
        source="class_kind",
        write_only=True,
        required=False,
        allow_null=True,
        help_text="ID of the ICD class kind",
    )

    # Display field for string representation
    display_name = serializers.SerializerMethodField()

    class Meta:
        model = ICD11Term
        fields = [
            "id",
            "foundation_uri",
            "linearization_uri",
            "code",
            "title",
            "class_kind",
            "class_kind_id",
            "depth_in_kind",
            "is_residual",
            "primary_location",
            "chapter_no",
            "browser_link",
            "icat_link",
            "is_leaf",
            "no_of_non_residual_children",
            "version_date",
            "display_name",
        ]
        read_only_fields = ["id", "display_name"]

    def get_display_name(self, obj):
        """
        Get formatted display name for the term.
        """
        if obj.code:
            return f"{obj.code} - {obj.title}"
        return obj.title

    def validate_foundation_uri(self, value):
        """
        Validate foundation URI is unique and properly formatted.
        """
        if not value:
            raise serializers.ValidationError("Foundation URI is required.")

        # Check uniqueness
        if self.instance and self.instance.foundation_uri == value:
            return value

        if ICD11Term.objects.filter(foundation_uri=value).exists():  # type: ignore
            raise serializers.ValidationError(
                "An ICD-11 term with this foundation URI already exists."
            )

        return value

    def validate_title(self, value):
        """
        Validate title is not empty.
        """
        if not value or len(value.strip()) == 0:
            raise serializers.ValidationError("Title cannot be empty.")
        return value.strip()

    def validate_code(self, value):
        """
        Validate ICD-11 code format.
        """
        if value and len(value.strip()) == 0:
            return None
        return value

    def validate_chapter_no(self, value):
        """
        Validate chapter number format.
        """
        if value and not value.isdigit() and value != "26":
            raise serializers.ValidationError(
                "Chapter number should be numeric or '26' for TM2."
            )
        return value

    def validate_depth_in_kind(self, value):
        """
        Validate depth is non-negative.
        """
        if value is not None and value < 0:
            raise serializers.ValidationError("Depth in kind cannot be negative.")
        return value

    def validate_no_of_non_residual_children(self, value):
        """
        Validate number of children is non-negative.
        """
        if value is not None and value < 0:
            raise serializers.ValidationError(
                "Number of non-residual children cannot be negative."
            )
        return value

    def validate(self, attrs):
        """
        Cross-field validation for ICD-11 terms.
        """
        # TM2 terms (Chapter 26) should have codes
        chapter_no = attrs.get("chapter_no")
        code = attrs.get("code")

        if chapter_no == "26" and not code:
            raise serializers.ValidationError(
                {"code": "TM2 terms (Chapter 26) must have a code."}
            )

        # Leaf nodes should not have non-residual children
        is_leaf = attrs.get("is_leaf", False)
        no_of_children = attrs.get("no_of_non_residual_children", 0)

        if is_leaf and no_of_children and no_of_children > 0:
            raise serializers.ValidationError(
                {
                    "no_of_non_residual_children": "Leaf nodes cannot have non-residual children."
                }
            )

        return attrs


class AyurvedhaModelSerializer(serializers.ModelSerializer):
    """
    Complete serializer for AyurvedhaModel.
    Handles Ayurveda medicine terminology from NAMASTE codes.
    """

    display_name = serializers.SerializerMethodField()

    class Meta:
        model = AyurvedhaModel
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

        if AyurvedhaModel.objects.filter(code=value).exists():  # type: ignore
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


class SiddhaModelSerializer(serializers.ModelSerializer):
    """
    Complete serializer for SiddhaModel.
    Handles Siddha medicine terminology from NAMASTE codes.
    """

    display_name = serializers.SerializerMethodField()

    class Meta:
        model = SiddhaModel
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

        if SiddhaModel.objects.filter(code=value).exists():  # type: ignore
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


class UnaniModelSerializer(serializers.ModelSerializer):
    """
    Complete serializer for UnaniModel.
    Handles Unani medicine terminology from NAMASTE codes.
    """

    display_name = serializers.SerializerMethodField()

    class Meta:
        model = UnaniModel
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

        if UnaniModel.objects.filter(code=value).exists():  # type: ignore
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
