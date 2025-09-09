from rest_framework import serializers

from .models import AyurvedhaModel, ICD11Term, ICDClassKind, SiddhaModel


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

        if ICDClassKind.objects.filter(name=value).exists():
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
        queryset=ICDClassKind.objects.all(),
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

        if ICD11Term.objects.filter(foundation_uri=value).exists():
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


class AyurvedaModelSerializer(serializers.ModelSerializer):
    """
    Complete serializer for AyurvedaModel.
    Handles Ayurveda medicine terminology from NAMASTE codes.
    """

    # Assuming BaseNamasteModel fields - adjust based on your actual model
    display_name = serializers.SerializerMethodField()

    class Meta:
        model = AyurvedhaModel
        fields = [
            "id",
            # BaseNamasteModel fields (adjust as per your actual model)
            "namaste_code",
            "english_name",
            "description",
            "severity_level",
            "category",
            "created_at",
            "updated_at",
            # AyurvedaModel specific fields
            "sanskrit_name",
            "devanagari_name",
            "romanized_name",
            "reference",
            "classical_reference",
            "dosha_involvement",
            "prakruti_relation",
            "display_name",
        ]
        read_only_fields = ["id", "created_at", "updated_at", "display_name"]

    def get_display_name(self, obj):
        """
        Get formatted display name for Ayurveda term.
        """
        if hasattr(obj, "namaste_code") and obj.namaste_code:
            return f"{obj.namaste_code} - {obj.english_name or obj.sanskrit_name}"
        return obj.english_name or obj.sanskrit_name or "Unnamed Term"

    def validate_sanskrit_name(self, value):
        """
        Validate Sanskrit name format.
        """
        if value and len(value.strip()) == 0:
            raise serializers.ValidationError(
                "Sanskrit name cannot be empty if provided."
            )
        return value

    def validate_devanagari_name(self, value):
        """
        Validate Devanagari name format.
        """
        if value and len(value.strip()) == 0:
            raise serializers.ValidationError(
                "Devanagari name cannot be empty if provided."
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

    def validate_dosha_involvement(self, value):
        """
        Validate dosha involvement according to Ayurvedic principles.
        """
        if value:
            valid_doshas = [
                "vata",
                "pitta",
                "kapha",
                "vata-pitta",
                "pitta-kapha",
                "vata-kapha",
                "tridosha",
                "sannipata",
            ]
            dosha_lower = value.lower().strip()
            if dosha_lower not in valid_doshas:
                raise serializers.ValidationError(
                    f"Invalid dosha involvement '{value}'. Valid options: {', '.join(valid_doshas)}"
                )
            return dosha_lower
        return value

    def validate_prakruti_relation(self, value):
        """
        Validate Prakruti (constitution) relation.
        """
        if value:
            valid_prakruti = [
                "vata-prakruti",
                "pitta-prakruti",
                "kapha-prakruti",
                "vata-pitta-prakruti",
                "pitta-kapha-prakruti",
                "vata-kapha-prakruti",
                "sama-prakruti",
            ]
            prakruti_lower = value.lower().strip()
            if prakruti_lower not in valid_prakruti:
                raise serializers.ValidationError(
                    f"Invalid Prakruti relation '{value}'. Valid options: {', '.join(valid_prakruti)}"
                )
            return prakruti_lower
        return value

    def validate_severity_level(self, value):
        """
        Validate severity level.
        """
        if value:
            valid_levels = ["mild", "moderate", "severe", "critical"]
            if value.lower() not in valid_levels:
                raise serializers.ValidationError(
                    f"Invalid severity level. Valid options: {', '.join(valid_levels)}"
                )
        return value

    def validate(self, attrs):
        """
        Cross-field validation for Ayurveda terms.
        """
        # At least one name should be provided
        sanskrit_name = attrs.get("sanskrit_name")
        english_name = attrs.get("english_name")
        devanagari_name = attrs.get("devanagari_name")

        if not any([sanskrit_name, english_name, devanagari_name]):
            raise serializers.ValidationError(
                "At least one name (Sanskrit, English, or Devanagari) must be provided."
            )

        return attrs


class SiddhaModelSerializer(serializers.ModelSerializer):
    """
    Complete serializer for SiddhaModel.
    Handles Siddha medicine terminology from NAMASTE codes.
    """

    # Display field for formatted name
    display_name = serializers.SerializerMethodField()

    class Meta:
        model = SiddhaModel
        fields = [
            "id",
            # BaseNamasteModel fields (adjust as per your actual model)
            "namaste_code",
            "english_name",
            "description",
            "severity_level",
            "category",
            "created_at",
            "updated_at",
            # SiddhaModel specific fields
            "tamil_name",
            "sanskrit_name",
            "romanized_name",
            "reference",
            "classical_reference",
            "mukkutram_involvement",
            "udal_kattugal_relation",
            "envagai_thervugal_signs",
            "display_name",
        ]
        read_only_fields = ["id", "created_at", "updated_at", "display_name"]

    def get_display_name(self, obj):
        """
        Get formatted display name for Siddha term.
        """
        if hasattr(obj, "namaste_code") and obj.namaste_code:
            return f"{obj.namaste_code} - {obj.english_name or obj.tamil_name}"
        return obj.english_name or obj.tamil_name or "Unnamed Term"

    def validate_tamil_name(self, value):
        """
        Validate Tamil name format.
        """
        if value and len(value.strip()) == 0:
            raise serializers.ValidationError("Tamil name cannot be empty if provided.")
        return value

    def validate_sanskrit_name(self, value):
        """
        Validate Sanskrit name format.
        """
        if value and len(value.strip()) == 0:
            raise serializers.ValidationError(
                "Sanskrit name cannot be empty if provided."
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

    def validate_mukkutram_involvement(self, value):
        """
        Validate Mukkutram (three humors) involvement in Siddha system.
        Vatham, Pitham, Kapham correspond to Vata, Pitta, Kapha in Ayurveda.
        """
        if value:
            valid_mukkutram = [
                "vatham",
                "pitham",
                "kapham",
                "vatham-pitham",
                "pitham-kapham",
                "vatham-kapham",
                "mukkutram",
                "thannila-mukkutram",
            ]
            mukkutram_lower = value.lower().strip()
            if mukkutram_lower not in valid_mukkutram:
                raise serializers.ValidationError(
                    f"Invalid Mukkutram involvement '{value}'. Valid options: {', '.join(valid_mukkutram)}"
                )
            return mukkutram_lower
        return value

    def validate_udal_kattugal_relation(self, value):
        """
        Validate Udal Kattugal (seven body constituents) relation.
        """
        if value:
            valid_udal_kattugal = [
                "saaram",
                "senneer",
                "oon",
                "kozhuppu",
                "enbu",
                "moolai",
                "sukkilam",
                "suronitham",
            ]

            # Allow multiple constituents separated by comma or semicolon
            constituents = [
                c.strip().lower()
                for c in value.replace(";", ",").split(",")
                if c.strip()
            ]

            for constituent in constituents:
                if constituent not in valid_udal_kattugal:
                    raise serializers.ValidationError(
                        f"Invalid Udal Kattugal constituent '{constituent}'. "
                        f"Valid options: {', '.join(valid_udal_kattugal)}"
                    )

            return ", ".join(constituents)
        return value

    def validate_envagai_thervugal_signs(self, value):
        """
        Validate Envagai Thervugal (eight diagnostic methods) signs.
        """
        if value:
            valid_envagai = [
                "naadi",
                "sparisam",
                "naa",
                "niram",
                "mozhi",
                "vizhi",
                "malam",
                "moothiram",
            ]

            # Allow multiple signs separated by comma or semicolon
            signs = [
                s.strip().lower()
                for s in value.replace(";", ",").split(",")
                if s.strip()
            ]

            for sign in signs:
                if sign not in valid_envagai:
                    raise serializers.ValidationError(
                        f"Invalid Envagai Thervugal sign '{sign}'. "
                        f"Valid options: {', '.join(valid_envagai)}"
                    )

            return ", ".join(signs)
        return value

    def validate_severity_level(self, value):
        """
        Validate severity level.
        """
        if value:
            valid_levels = ["mild", "moderate", "severe", "critical"]
            if value.lower() not in valid_levels:
                raise serializers.ValidationError(
                    f"Invalid severity level. Valid options: {', '.join(valid_levels)}"
                )
        return value

    def validate(self, attrs):
        """
        Cross-field validation for Siddha terms.
        """
        # At least one name should be provided
        tamil_name = attrs.get("tamil_name")
        english_name = attrs.get("english_name")
        sanskrit_name = attrs.get("sanskrit_name")

        if not any([tamil_name, english_name, sanskrit_name]):
            raise serializers.ValidationError(
                "At least one name (Tamil, English, or Sanskrit) must be provided."
            )

        return attrs
