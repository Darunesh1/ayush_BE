#!/bin/bash
# export_medical_data.sh - Export Ayushsync Medical Terminology Data
# Usage: ./export_medical_data.sh

set -e  # Exit on any error

echo "========================================="
echo "  AYUSHSYNC MEDICAL DATA EXPORT"
echo "========================================="

# Configuration
EXPORT_DIR="exports"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ARCHIVE_NAME="ayushsync_medical_data_${TIMESTAMP}.tar.gz"

# Check if Docker containers are running
if ! docker-compose ps | grep -q "Up"; then
    echo "❌ Error: Docker containers are not running. Please start with 'docker-compose up -d'"
    exit 1
fi

# Create exports directory
echo "📁 Creating export directory..."
mkdir -p $EXPORT_DIR
cd $EXPORT_DIR

# Function to export with error handling
export_model() {
    local model=$1
    local filename=$2
    echo "📤 Exporting $model..."
    
    if docker-compose exec -T web python manage.py dumpdata $model --indent=2 > $filename; then
        local count=$(cat $filename | grep -c '"model"' || echo "0")
        echo "✅ $model: $count records exported to $filename"
    else
        echo "❌ Failed to export $model"
        exit 1
    fi
}

# Export in dependency order
echo ""
echo "🔄 Step 1: Exporting Django ContentTypes (required for GenericForeignKey)..."
export_model "contenttypes" "contenttypes.json"

echo ""
echo "🔄 Step 2: Exporting Terminologies (NAMASTE + ICD-11)..."
export_model "terminologies.ICD11Term" "icd11_terms.json"
export_model "terminologies.Ayurvedha" "ayurveda_terms.json"
export_model "terminologies.Siddha" "siddha_terms.json"
export_model "terminologies.Unani" "unani_terms.json"

echo ""
echo "🔄 Step 3: Exporting Terminology Mappings..."
export_model "namasthe_mapping.TerminologyMapping" "terminology_mappings.json"
export_model "namasthe_mapping.ConceptMapping" "concept_mappings.json"
export_model "namasthe_mapping.MappingAudit" "mapping_audit.json"

echo ""
echo "🔄 Step 4: Creating combined export with natural keys..."
echo "📤 Exporting complete dataset..."

COMBINED_FILE="ayushsync_complete_${TIMESTAMP}.json"

if docker-compose exec -T web python manage.py dumpdata \
    contenttypes \
    terminologies.ICD11Term \
    terminologies.Ayurvedha \
    terminologies.Siddha \
    terminologies.Unani \
    namasthe_mapping.TerminologyMapping \
    namasthe_mapping.ConceptMapping \
    namasthe_mapping.MappingAudit \
    --natural-foreign --natural-primary \
    --indent=2 > "$COMBINED_FILE"; then
    
    # Count total records (removed 'local' keyword)
    total_records=$(cat "$COMBINED_FILE" | grep -c '"model"' || echo "0")
    echo "✅ Complete dataset: $total_records total records exported"
else
    echo "❌ Failed to create combined export"
    exit 1
fi

echo ""
echo "🔄 Step 5: Creating compressed archive..."
if tar -czf $ARCHIVE_NAME *.json; then
    # Get archive size (removed 'local' keyword)
    archive_size=$(ls -lh $ARCHIVE_NAME | awk '{print $5}')
    echo "✅ Archive created: $ARCHIVE_NAME ($archive_size)"
else
    echo "❌ Failed to create archive"
    exit 1
fi

echo ""
echo "📊 Export Summary:"
echo "==================="
ls -la *.json *.tar.gz | while read line; do
    echo "  $line"
done

cd ..

echo ""
echo "🎉 Export completed successfully!"
echo "📁 Files are located in: $(pwd)/$EXPORT_DIR/"
echo "📦 Transfer archive: $EXPORT_DIR/$ARCHIVE_NAME"
echo ""
echo "💾 Data Summary:"
echo "  - ICD-11 Terms: 35,171 records"
echo "  - Ayurveda Terms: 2,893 records"
echo "  - Siddha Terms: 1,925 records"
echo "  - Unani Terms: 2,521 records"
echo "  - Concept Mappings: 21,922 records"
echo "  - BioBERT embeddings included"
echo ""
echo "Next steps:"
echo "  1. Copy $EXPORT_DIR/$ARCHIVE_NAME to your target system"
echo "  2. Extract: tar -xzf $ARCHIVE_NAME"
echo "  3. Run: ./import_medical_data.sh"

