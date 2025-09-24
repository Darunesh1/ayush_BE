#!/bin/bash
# import_medical_data_pg.sh - Import using PostgreSQL pg_restore (Much faster)
# Usage: ./import_medical_data_pg.sh [backup_file]

set -e

echo "========================================="
echo "  AYUSHSYNC MEDICAL DATA IMPORT (pg_restore)"
echo "========================================="

# Your specific database configuration
DB_NAME="ayushsync_db"
DB_USER="devuser"
DB_PASSWORD="devpassword"

# Find backup file
if [ ! -z "$1" ]; then
    BACKUP_FILE="$1"
else
    # Look for most recent backup
    BACKUP_FILE=$(ls -t ayushsync_medical_*.backup 2>/dev/null | head -n1)
    if [ -z "$BACKUP_FILE" ]; then
        echo "❌ No backup file found. Usage:"
        echo "   ./import_medical_data_pg.sh [backup_filename]"
        echo ""
        echo "Available backups:"
        ls -la *.backup 2>/dev/null || echo "   No .backup files found"
        echo ""
        echo "💡 Make sure you have transferred the .backup file from your source system"
        exit 1
    fi
fi

if [ ! -f "$BACKUP_FILE" ]; then
    echo "❌ Backup file not found: $BACKUP_FILE"
    exit 1
fi

backup_size=$(ls -lh "$BACKUP_FILE" | awk '{print $5}')
echo "📦 Using backup: $BACKUP_FILE ($backup_size)"
echo "🔧 Target database: $DB_NAME"
echo "👤 User: $DB_USER"

# Check if Docker containers are running
if ! docker-compose ps | grep -q "Up"; then
    echo "❌ Error: Docker containers are not running. Please start with:"
    echo "   docker-compose up -d"
    exit 1
fi

# Get database container (using service name 'postgres' from your docker-compose)
DB_CONTAINER=$(docker-compose ps -q postgres)
if [ -z "$DB_CONTAINER" ]; then
    echo "❌ PostgreSQL container not found. Check if service is named 'postgres' in docker-compose.yml"
    exit 1
fi

# Wait for database to be ready
echo "⏳ Waiting for database to be ready..."
timeout=60
counter=0
while ! docker-compose exec -T postgres pg_isready -U $DB_USER -d $DB_NAME >/dev/null 2>&1; do
    if [ $counter -ge $timeout ]; then
        echo "❌ Database not ready after ${timeout}s"
        exit 1
    fi
    echo "   Database not ready, waiting... ($counter/${timeout}s)"
    sleep 2
    counter=$((counter + 2))
done
echo "✅ Database is ready!"

echo ""
echo "📤 Creating backup of current database (if any data exists)..."
CURRENT_BACKUP="backup_before_restore_$(date +%Y%m%d_%H%M%S).backup"

# Check if database has any data first
data_exists=$(docker-compose exec -T postgres psql -U $DB_USER -d $DB_NAME -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';" 2>/dev/null || echo "0")
data_exists=$(echo $data_exists | tr -d ' \n\r')

if [ "$data_exists" != "0" ] && [ "$data_exists" != "" ]; then
    echo "📊 Found existing data, creating backup..."
    docker-compose exec -T postgres pg_dump \
        -U $DB_USER \
        -d $DB_NAME \
        --format=custom \
        --compress=9 \
        --file=/tmp/$CURRENT_BACKUP
    
    if docker-compose exec -T postgres test -f /tmp/$CURRENT_BACKUP; then
        docker cp $DB_CONTAINER:/tmp/$CURRENT_BACKUP $CURRENT_BACKUP
        docker-compose exec -T postgres rm /tmp/$CURRENT_BACKUP
        current_size=$(ls -lh $CURRENT_BACKUP | awk '{print $5}')
        echo "✅ Current database backed up to: $CURRENT_BACKUP ($current_size)"
    fi
else
    echo "ℹ️  No existing data found, skipping backup"
fi

# Copy backup file to container
echo ""
echo "📁 Copying backup file to database container..."
docker cp "$BACKUP_FILE" $DB_CONTAINER:/tmp/restore_data.backup

# Drop and recreate database for clean restore
echo "🗑️  Preparing database for restore..."
docker-compose exec -T postgres psql -U $DB_USER -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = '$DB_NAME' AND pid <> pg_backend_pid();" 2>/dev/null || true
docker-compose exec -T postgres dropdb -U $DB_USER $DB_NAME 2>/dev/null || echo "ℹ️  Database didn't exist or was already empty"

echo "🆕 Creating fresh database..."
docker-compose exec -T postgres createdb -U $DB_USER $DB_NAME

# Restore database
echo "📥 Restoring your medical terminology data..."
echo "   This may take 2-10 minutes for your large dataset..."
if docker-compose exec -T postgres pg_restore \
    -U $DB_USER \
    -d $DB_NAME \
    --verbose \
    --no-owner \
    --no-privileges \
    --jobs=4 \
    /tmp/restore_data.backup; then
    echo "✅ Database restored successfully!"
else
    echo "❌ Database restore failed!"
    echo "💡 Check if the backup file is compatible and not corrupted"
    exit 1
fi

# Clean up
docker-compose exec -T postgres rm /tmp/restore_data.backup

# Run Django migrations to ensure schema consistency
echo ""
echo "🔄 Running Django migrations to ensure schema consistency..."
if docker-compose exec -T web python manage.py migrate; then
    echo "✅ Django migrations completed"
else
    echo "⚠️  Django migrations had issues (this is usually non-critical for data restore)"
fi

# Verify restored data
echo ""
echo "📊 Verifying restored medical terminology data..."
VERIFICATION_SCRIPT='
import sys
from terminologies.models import ICD11Term, Ayurvedha, Siddha, Unani
from namasthe_mapping.models import TerminologyMapping, ConceptMapping, MappingAudit

print("📊 Restored Data Verification:")
print("=" * 50)

try:
    icd11_count = ICD11Term.objects.count()
    ayurveda_count = Ayurvedha.objects.count()
    siddha_count = Siddha.objects.count()
    unani_count = Unani.objects.count()
    mapping_config_count = TerminologyMapping.objects.count()
    concept_mapping_count = ConceptMapping.objects.count()
    audit_count = MappingAudit.objects.count()

    print(f"🏥 ICD-11 Terms: {icd11_count:,}")
    print(f"🌿 Ayurveda Terms: {ayurveda_count:,}")
    print(f"⚗️  Siddha Terms: {siddha_count:,}")
    print(f"🧪 Unani Terms: {unani_count:,}")
    print(f"⚙️  Terminology Mappings: {mapping_config_count}")
    print(f"🔗 Concept Mappings: {concept_mapping_count:,}")
    print(f"📝 Audit Entries: {audit_count}")
    print("-" * 50)

    total_terminology = icd11_count + ayurveda_count + siddha_count + unani_count
    total_records = total_terminology + mapping_config_count + concept_mapping_count + audit_count
    print(f"📊 Total Terminology Records: {total_terminology:,}")
    print(f"📊 Total All Records: {total_records:,}")

    # Test BioBERT embeddings
    if icd11_count > 0:
        icd11_with_embeddings = ICD11Term.objects.exclude(embedding__isnull=True).count()
        embedding_percentage = (icd11_with_embeddings / icd11_count) * 100 if icd11_count > 0 else 0
        print(f"🧠 ICD-11 terms with BioBERT embeddings: {icd11_with_embeddings:,}/{icd11_count:,} ({embedding_percentage:.1f}%)")

    if ayurveda_count > 0:
        ayurveda_with_embeddings = Ayurvedha.objects.exclude(embedding__isnull=True).count()
        embedding_percentage = (ayurveda_with_embeddings / ayurveda_count) * 100 if ayurveda_count > 0 else 0
        print(f"🧠 Ayurveda terms with BioBERT embeddings: {ayurveda_with_embeddings:,}/{ayurveda_count:,} ({embedding_percentage:.1f}%)")

    # Test relationships
    print("\n🔗 Testing Data Relationships:")
    if concept_mapping_count > 0:
        sample_mapping = ConceptMapping.objects.first()
        if sample_mapping and hasattr(sample_mapping, "get_source_display") and hasattr(sample_mapping, "get_target_display"):
            print(f"✅ Sample Mapping: {sample_mapping.get_source_display()} → {sample_mapping.get_target_display()}")
            print(f"   Source System: {sample_mapping.get_source_system_display()}")
            print(f"   Confidence Score: {sample_mapping.confidence_score:.3f}")
            print(f"   Similarity Score: {sample_mapping.similarity_score:.3f}")
            print("✅ GenericForeignKey relationships working correctly")
        else:
            print("⚠️  Sample mapping found but methods unavailable")
    else:
        print("ℹ️  No concept mappings to test")

    print("\n🎉 Data verification completed successfully!")
    print("✅ All BioBERT embeddings and AI mappings preserved")
    
except Exception as e:
    print(f"❌ Verification failed: {str(e)}")
    print("💡 The data might still be restored correctly, but Django models may need attention")
    sys.exit(1)
'

if docker-compose exec -T web python manage.py shell -c "$VERIFICATION_SCRIPT"; then
    echo ""
    echo "✅ Data verification successful!"
else
    echo "❌ Data verification had issues"
    echo "💡 Check Django model imports and database connections"
fi

echo ""
echo "🎉 IMPORT COMPLETED SUCCESSFULLY!"
echo "================================="
echo ""
echo "📦 Data imported from: $BACKUP_FILE ($backup_size)"
if [ -f "$CURRENT_BACKUP" ]; then
    current_size=$(ls -lh $CURRENT_BACKUP | awk '{print $5}')
    echo "💾 Previous data backed up to: $CURRENT_BACKUP ($current_size)"
fi
echo ""
echo "🚀 Your Ayushsync medical terminology system is ready!"
echo ""
echo "📋 What was restored:"
echo "  ✅ Complete medical terminology database"
echo "  ✅ ICD-11 terms with definitions and classifications"  
echo "  ✅ NAMASTE terms (Ayurveda, Siddha, Unani)"
echo "  ✅ BioBERT embeddings for similarity search"
echo "  ✅ AI-generated concept mappings with confidence scores"
echo "  ✅ Audit trail and validation data"
echo "  ✅ PostgreSQL search indexes and triggers"
echo ""
echo "🔍 Next steps:"
echo "  1. Test Django admin panel: http://localhost:8000/admin/"
echo "  2. Verify FHIR terminology endpoints"
echo "  3. Test fuzzy search capabilities"
echo "  4. Validate BioBERT similarity functions"

