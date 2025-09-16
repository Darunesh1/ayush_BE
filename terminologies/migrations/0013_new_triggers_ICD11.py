from django.db import migrations


class Migration(migrations.Migration):
    atomic = False  # Triggers and index creation should not run inside transaction

    dependencies = [
        ("terminologies", "0012_auto_20250917_0155"),  # update to your latest migration
    ]

    operations = [
        migrations.RunSQL(
            sql="""
            -- Drop old trigger/function on icd11_terms
            DROP TRIGGER IF EXISTS icd11_search_vector_trigger ON icd11_terms;
            DROP FUNCTION IF EXISTS icd11_search_vector_update();

            -- Create updated trigger/function on icd11_terms (code + title)
            CREATE OR REPLACE FUNCTION icd11_search_vector_update() RETURNS trigger AS $$
            BEGIN
              NEW.search_vector :=
                  setweight(to_tsvector('english', coalesce(NEW.code, '')), 'A') ||
                  setweight(to_tsvector('english', coalesce(NEW.title, '')), 'A');
              RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;

            CREATE TRIGGER icd11_search_vector_trigger
            BEFORE INSERT OR UPDATE ON icd11_terms
            FOR EACH ROW EXECUTE FUNCTION icd11_search_vector_update();


            -- Add search_vector column to icd11_synonyms if missing
            ALTER TABLE icd11_synonyms
            ADD COLUMN IF NOT EXISTS search_vector tsvector;

            -- Drop old trigger/function on icd11_synonyms
            DROP TRIGGER IF EXISTS icd11synonym_search_vector_trigger ON icd11_synonyms;
            DROP FUNCTION IF EXISTS icd11synonym_search_vector_update();

            -- Create trigger/function for icd11_synonyms on label
            CREATE OR REPLACE FUNCTION icd11synonym_search_vector_update() RETURNS trigger AS $$
            BEGIN
              NEW.search_vector := setweight(to_tsvector('english', coalesce(NEW.label, '')), 'A');
              RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;

            CREATE TRIGGER icd11synonym_search_vector_trigger
            BEFORE INSERT OR UPDATE ON icd11_synonyms
            FOR EACH ROW EXECUTE FUNCTION icd11synonym_search_vector_update();
            """,
            reverse_sql="""
            DROP TRIGGER IF EXISTS icd11_search_vector_trigger ON icd11_terms;
            DROP FUNCTION IF EXISTS icd11_search_vector_update();
            DROP TRIGGER IF EXISTS icd11synonym_search_vector_trigger ON icd11_synonyms;
            DROP FUNCTION IF EXISTS icd11synonym_search_vector_update();
            ALTER TABLE icd11_synonyms DROP COLUMN IF EXISTS search_vector;
            """,
        ),
    ]
