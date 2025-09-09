from django.db import migrations


class Migration(migrations.Migration):
    atomic = False  # Because we will use CONCURRENTLY in SQL (can’t run inside a transaction)

    dependencies = [
        (
            "terminologies",
            "0006_alter_ayurvedhamodel_options_alter_icd11term_options_and_more",
        ),  # Your last migration
    ]

    operations = [
        # AyurvedhaModel trigram indexes
        migrations.RunSQL(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS ayurveda_code_trgm_idx ON ayurveda_terms USING GIN(code gin_trgm_ops);",
            reverse_sql="DROP INDEX IF EXISTS ayurveda_code_trgm_idx;",
        ),
        migrations.RunSQL(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS ayurveda_english_trgm_idx ON ayurveda_terms USING GIN(english_name gin_trgm_ops);",
            reverse_sql="DROP INDEX IF EXISTS ayurveda_english_trgm_idx;",
        ),
        migrations.RunSQL(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS ayurveda_hindi_trgm_idx ON ayurveda_terms USING GIN(hindi_name gin_trgm_ops);",
            reverse_sql="DROP INDEX IF EXISTS ayurveda_hindi_trgm_idx;",
        ),
        migrations.RunSQL(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS ayurveda_diacritical_trgm_idx ON ayurveda_terms USING GIN(diacritical_name gin_trgm_ops);",
            reverse_sql="DROP INDEX IF EXISTS ayurveda_diacritical_trgm_idx;",
        ),
        # SiddhaModel trigram indexes
        migrations.RunSQL(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS siddha_code_trgm_idx ON siddha_terms USING GIN(code gin_trgm_ops);",
            reverse_sql="DROP INDEX IF EXISTS siddha_code_trgm_idx;",
        ),
        migrations.RunSQL(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS siddha_english_trgm_idx ON siddha_terms USING GIN(english_name gin_trgm_ops);",
            reverse_sql="DROP INDEX IF EXISTS siddha_english_trgm_idx;",
        ),
        migrations.RunSQL(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS siddha_tamil_trgm_idx ON siddha_terms USING GIN(tamil_name gin_trgm_ops);",
            reverse_sql="DROP INDEX IF EXISTS siddha_tamil_trgm_idx;",
        ),
        migrations.RunSQL(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS siddha_romanized_trgm_idx ON siddha_terms USING GIN(romanized_name gin_trgm_ops);",
            reverse_sql="DROP INDEX IF EXISTS siddha_romanized_trgm_idx;",
        ),
        # UnaniModel trigram indexes
        migrations.RunSQL(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS unani_code_trgm_idx ON unani_terms USING GIN(code gin_trgm_ops);",
            reverse_sql="DROP INDEX IF EXISTS unani_code_trgm_idx;",
        ),
        migrations.RunSQL(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS unani_english_trgm_idx ON unani_terms USING GIN(english_name gin_trgm_ops);",
            reverse_sql="DROP INDEX IF EXISTS unani_english_trgm_idx;",
        ),
        migrations.RunSQL(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS unani_arabic_trgm_idx ON unani_terms USING GIN(arabic_name gin_trgm_ops);",
            reverse_sql="DROP INDEX IF EXISTS unani_arabic_trgm_idx;",
        ),
        migrations.RunSQL(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS unani_romanized_trgm_idx ON unani_terms USING GIN(romanized_name gin_trgm_ops);",
            reverse_sql="DROP INDEX IF EXISTS unani_romanized_trgm_idx;",
        ),
        # ICD11Term trigram indexes
        migrations.RunSQL(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS icd11_code_trgm_idx ON icd11_terms USING GIN(code gin_trgm_ops);",
            reverse_sql="DROP INDEX IF EXISTS icd11_code_trgm_idx;",
        ),
        migrations.RunSQL(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS icd11_title_trgm_idx ON icd11_terms USING GIN(title gin_trgm_ops);",
            reverse_sql="DROP INDEX IF EXISTS icd11_title_trgm_idx;",
        ),
        migrations.RunSQL(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS icd11_location_trgm_idx ON icd11_terms USING GIN(primary_location gin_trgm_ops);",
            reverse_sql="DROP INDEX IF EXISTS icd11_location_trgm_idx;",
        ),
    ]
