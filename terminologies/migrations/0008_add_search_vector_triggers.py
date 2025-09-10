from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [
        (
            "terminologies",
            "0007_add_trigram_indexes",
        ),  # Update if latest migration changes
    ]
    operations = [
        # AyurvedhaModel
        migrations.RunSQL(
            """
            CREATE OR REPLACE FUNCTION ayurveda_search_vector_update() RETURNS trigger AS $$
            BEGIN
              NEW.search_vector :=
                  setweight(to_tsvector('english', coalesce(NEW.code, '')), 'A') ||
                  setweight(to_tsvector('english', coalesce(NEW.description, '')), 'B') ||
                  setweight(to_tsvector('english', coalesce(NEW.english_name, '')), 'A') ||
                  setweight(to_tsvector('english', coalesce(NEW.hindi_name, '')), 'C') ||
                  setweight(to_tsvector('english', coalesce(NEW.diacritical_name, '')), 'D');
              RETURN NEW;
            ENA
            $$ LANGUAGE plpgsql;
            CREATE TRIGGER ayurveda_search_vector_trigger
            BEFORE INSERT OR UPDATE ON ayurveda_terms
            FOR EACH ROW EXECUTE FUNCTION ayurveda_search_vector_update();
            """,
            reverse_sql="""
            DROP TRIGGER IF EXISTS ayurveda_search_vector_trigger ON ayurveda_terms;
            DROP FUNCTION IF EXISTS ayurveda_search_vector_update();
            """,
        ),
        # SiddhaModel
        migrations.RunSQL(
            """
            CREATE OR REPLACE FUNCTION siddha_search_vector_update() RETURNS trigger AS $$
            BEGIN
              NEW.search_vector :=
                  setweight(to_tsvector('english', coalesce(NEW.code, '')), 'A') ||
                  setweight(to_tsvector('english', coalesce(NEW.description, '')), 'B') ||
                  setweight(to_tsvector('english', coalesce(NEW.english_name, '')), 'A') ||
                  setweight(to_tsvector('english', coalesce(NEW.tamil_name, '')), 'B') ||
                  setweight(to_tsvector('english', coalesce(NEW.romanized_name, '')), 'D') ||
                  setweight(to_tsvector('english', coalesce(NEW.reference, '')), 'D');
              RETURN NEW;
            END
            $$ LANGUAGE plpgsql;
            CREATE TRIGGER siddha_search_vector_trigger
            BEFORE INSERT OR UPDATE ON siddha_terms
            FOR EACH ROW EXECUTE FUNCTION siddha_search_vector_update()B
            """,
            reverse_sql="""
            DROP TRIGGER IF EXISTS siddha_search_vector_trigger ON siddha_terms;
            DROP FUNCTION IF EXISTS siddha_search_vector_update();
            """,
        ),
        # UnaniModel
        migrations.RunSQL(
            """
            CREATE OR REPLACE FUNCTION unani_search_vector_update() RETURNS trigger AS $$
            BEGIN
              NEW.search_vector :=
                  setweight(to_tsvector('english', coalesce(NEW.code, '')), 'A') ||
                  setweight(to_tsvector('english', coalesce(NEW.description, '')), 'B') ||
                  setweight(to_tsvector('english', coalesce(NEW.english_name, '')), 'A') ||
                  setweight(to_tsvector('english', coalesce(NEW.arabic_name, '')), 'C') ||
                  setweight(to_tsvector('english', coalesce(NEW.romanized_name, '')), 'D') ||
                  setweight(to_tsvector('english', coalesce(NEW.reference, '')), 'D');
              RETURN NEW;
            END
            $$ LANGUAGE plpgsql;
            CREATE TRIGGER unani_search_vector_trigger
            BEFORE INSERT OR UPDATE ON unani_terms
            FOR EACH ROW EXECUTE FUNCTION unani_search_vector_update();
            """,
            reverse_sql="""
            DROP TRIGGER IF EXISTS unani_search_vector_trigger ON unani_terms;
            DROP FUNCTION IF EXISTS unani_search_vector_update();
            """,
        ),
        # ICD11Term
        migrations.RunSQL(
            """
            CREATE OR REPLACE FUNCTION icd11_search_vector_update() RETURNS trigger AS $$
            BEGIN
              NEW.search_vector :=
                  setweight(to_tsvector('english', coalesce(NEW.code, '')), 'A') ||
                  setweight(to_tsvector('english', coalesce(NEW.description, '')), 'B') ||
                  setweight(to_tsvector('english', coalesce(NEW.title, '')), 'A') ||
                  setweight(to_tsvector('english', coalesce(NEW.primary_location, '')), 'C');
              RETURN NEW;
            END
            $$ LANGUAGE plpgsql;
            CREATE TRIGGER icd11_search_vector_trigger
            BEFORE INSERT OR UPDATE ON icd11_terms
            FOR EACH ROW EXECUTE FUNCTION icd11_search_vector_update();
            """,
            reverse_sql="""
            DROP TRIGGER IF EXISTS icd11_search_vector_trigger ON icd11_terms;
            DROP FUNCTION IF EXISTS icd11_search_vector_update();
            """,
        ),
    ]
