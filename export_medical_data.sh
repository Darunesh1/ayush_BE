#!/bin/bash
# export_medical_data_pg.sh - Export using PostgreSQL pg_dump (Much faster and smaller)
# Usage: ./export_medical_data_pg.sh

set -e

echo "========================================="
echo "  AYUSHSYNC MEDICAL DATA EXPORT (pg_dump)"
echo "========================================="

# Configuration from your docker-compose.yml
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="ayushsync_medical_${TIMESTAMP}"

# Your specific database configuration
DB_NAME="ayushsync_db"
DB_USER="devuser"
DB_PASSWORD="devpassword"

# Check if Docker containers are running
if ! docker-compose ps | grep -q "Up"; then
    echo "❌ Error: Docker containers are not running. Please start with 'docker-compose up -d'"
    exit 1
fi

echo "📁 Starting PostgreSQL database export..."
echo "🔧 Database: $DB_NAME"
echo "👤 User: $DB_USER"

# Get database container (using service name 'postgres' from your docker-compose)
DB_CONTAINER=$(docker-compose ps -q postgres)

if [ -z "$DB_CONTAINER" ]; then
    echo "❌ PostgreSQL container not found. Check if service is named 'postgres' in docker-compose.yml"
    exit 1
fi

echo "📤 Exporting with pg_dump (custom format - highly compressed)..."

# Custom format (binary, compressed) - RECOMMENDED for large datasets
echo "Creating binary backup (recommended for restore)..."
docker-compose exec -T postgres pg_dump \
    -U $DB_USER \
    -d $DB_NAME \
    --format=custom \
    --compress=9 \
    --verbose \
    --file=/tmp/${BACKUP_NAME}.backup

# Copy from container to host
docker cp $DB_CONTAINER:/tmp/${BACKUP_NAME}.backup ${BACKUP_NAME}.backup

echo "📤 Creating additional SQL backup (for manual inspection if needed)..."
# Compressed SQL dump
docker-compose exec -T postgres pg_dump \
    -U $DB_USER \
    -d $DB_NAME \
    --format=plain \
    --compress=9 \
    --verbose \
    --file=/tmp/${BACKUP_NAME}.sql.gz

# Copy from container to host  
docker cp $DB_CONTAINER:/tmp/${BACKUP_NAME}.sql.gz ${BACKUP_NAME}.sql.gz

# Clean up container files
docker-compose exec -T postgres rm /tmp/${BACKUP_NAME}.backup /tmp/${BACKUP_NAME}.sql.gz

# Get file sizes and show summary
custom_size=$(ls -lh ${BACKUP_NAME}.backup | awk '{print $5}')
sql_size=$(ls -lh ${BACKUP_NAME}.sql.gz | awk '{print $5}')

echo ""
echo "🎉 PostgreSQL export completed successfully!"
echo "=================================="
echo "📁 Files created:"
echo "  📦 ${BACKUP_NAME}.backup ($custom_size) - Binary format (RECOMMENDED)"
echo "  📄 ${BACKUP_NAME}.sql.gz ($sql_size) - SQL format (for inspection)"
echo ""
echo "💾 Benefits over JSON export:"
echo "  ✅ 70-90% smaller file size"
echo "  ✅ 10-50x faster export/import"  
echo "  ✅ Preserves all PostgreSQL data types perfectly"
echo "  ✅ Includes BioBERT embeddings in efficient binary format"
echo "  ✅ No Django dependency needed for restore"
echo ""
echo "📊 Your dataset summary:"
echo "  - Database: ayushsync_db"
echo "  - ~35,000 ICD-11 terms with embeddings"
echo "  - ~7,000 NAMASTE terms (Ayurveda/Siddha/Unani)"
echo "  - ~22,000 AI-generated concept mappings"
echo ""
echo "🚚 Next steps:"
echo "  1. Transfer ${BACKUP_NAME}.backup to your target system"
echo "  2. Run: ./import_medical_data_pg.sh ${BACKUP_NAME}.backup"
echo ""
echo "💡 Tip: Use the .backup file (not .sql.gz) for fastest restore"

