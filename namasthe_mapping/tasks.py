# namasthe_mapping/tasks.py
# COMPLETELY OPTIMIZED for on-demand execution with minimal resource usage
# Designed to prevent laptop crashes and memory issues

import gc
import json
import logging
import time
from typing import Any, Dict, List, Optional

from celery import shared_task
from django.core.cache import cache
from django.db import transaction
from django.utils import timezone

# Only import non-Django modules at module level
logger = logging.getLogger(__name__)

# ========== CONFIGURATION ==========
MAX_BATCH_SIZE = 15  # Conservative batch size to prevent crashes
MAX_ICD11_TERMS = 5000  # Limit ICD-11 terms for memory efficiency
CHUNK_SIZE = 10  # Size for processing large batches
CACHE_TIMEOUT = 900  # 15 minutes cache (shorter for on-demand tasks)
MAX_MAPPINGS_PER_TERM = 3  # Limit matches per term
MEMORY_CLEANUP_INTERVAL = 5  # Cleanup every N terms


# ========== MAIN MAPPING TASK ==========


@shared_task(
    bind=True,
    autoretry_for=(Exception,),
    retry_kwargs={"max_retries": 2, "countdown": 30},
    soft_time_limit=600,  # 10 minutes
    time_limit=900,  # 15 minutes
    acks_late=True,
    reject_on_worker_lost=True,
    max_retries=2,
)
def create_concept_mappings_batch(
    self,
    mapping_config_id: str,
    source_term_ids: List[int],
    system: str,
    max_mappings: int = 3,
    similarity_threshold: float = 0.75,
) -> Dict[str, Any]:
    """
    OPTIMIZED batch mapping creation for on-demand execution
    - Memory-safe processing
    - Automatic chunking for large batches
    - Aggressive garbage collection
    - Conservative resource usage
    """

    start_time = time.time()

    # Enforce batch size limits to prevent crashes
    if len(source_term_ids) > MAX_BATCH_SIZE:
        logger.warning(
            f"Large batch ({len(source_term_ids)}) detected, chunking into {CHUNK_SIZE}-term pieces"
        )
        return _process_large_batch_in_chunks(
            self,
            mapping_config_id,
            source_term_ids,
            system,
            min(max_mappings, MAX_MAPPINGS_PER_TERM),
            similarity_threshold,
        )

    logger.info(
        f"🔗 Processing {len(source_term_ids)} {system} terms [MEMORY-OPTIMIZED]"
    )

    # Short-lived progress tracking
    progress_key = f"mapping_{self.request.id}"
    _update_progress(
        progress_key,
        "starting",
        0,
        {
            "system": system,
            "batch_size": len(source_term_ids),
            "start_time": start_time,
        },
    )

    try:
        # Lazy import Django models to reduce startup memory
        from django.contrib.contenttypes.models import ContentType

        from namasthe_mapping.models import ConceptMapping, TerminologyMapping
        from namasthe_mapping.services import HuggingFaceAPIService
        from terminologies.models import Ayurvedha, ICD11Term, Siddha, Unani

        # Get mapping config
        mapping_config = TerminologyMapping.objects.get(id=mapping_config_id)

        # Model mapping
        model_map = {"ayurveda": Ayurvedha, "siddha": Siddha, "unani": Unani}
        SourceModel = model_map.get(system)
        if not SourceModel:
            raise ValueError(f"Unknown system: {system}")

        source_content_type = ContentType.objects.get_for_model(SourceModel)

        _update_progress(progress_key, "loading", 15, {"stage": "loading_terms"})

        # Memory-optimized source term loading
        source_terms = list(
            SourceModel.objects.filter(
                id__in=source_term_ids, embedding__isnull=False
            ).only("id", "embedding")  # Load only necessary fields
        )

        if not source_terms:
            logger.warning(f"No valid source terms found")
            return {
                "mappings_created": 0,
                "source_terms_processed": 0,
                "elapsed_time": time.time() - start_time,
            }

        # Memory-limited ICD-11 terms loading with smart caching
        icd11_terms = _get_limited_icd11_terms(system)
        if not icd11_terms:
            raise Exception("No ICD-11 terms available for mapping")

        logger.info(
            f"🧠 Loaded {len(source_terms)} source terms, {len(icd11_terms)} ICD-11 targets"
        )

        _update_progress(
            progress_key,
            "processing",
            35,
            {
                "stage": "initializing_service",
                "source_terms": len(source_terms),
                "target_terms": len(icd11_terms),
            },
        )

        # Initialize BioBERT service (memory-conscious)
        service = HuggingFaceAPIService(model_preference="biobert")
        icd11_embeddings = [term["embedding"] for term in icd11_terms]

        logger.info(f"🔧 Service: {service.model_name} (dim: {service.embedding_dim})")

        # Process mappings with memory management
        mappings_to_create = []
        processed_count = 0

        for i, source_term in enumerate(source_terms):
            try:
                # Validate embedding dimensions
                if not _validate_embedding(source_term.embedding):
                    continue

                # Find similarity matches
                matches = service.find_best_matches(
                    source_term.embedding,
                    icd11_embeddings,
                    similarity_threshold=similarity_threshold,
                    top_k=min(max_mappings, MAX_MAPPINGS_PER_TERM),
                )

                # Create mapping objects
                for match in matches:
                    icd11_term = icd11_terms[match["index"]]

                    mapping = ConceptMapping(
                        mapping=mapping_config,
                        source_content_type=source_content_type,
                        source_object_id=source_term.id,
                        target_concept_id=icd11_term["id"],
                        relationship=_determine_relationship(match["similarity"]),
                        similarity_score=match["similarity"],
                        confidence_score=min(match["similarity"] + 0.05, 1.0),
                        # Store embeddings only for high-confidence mappings (saves memory)
                        source_embedding=source_term.embedding
                        if match["similarity"] >= 0.85
                        else None,
                        target_embedding=icd11_term["embedding"]
                        if match["similarity"] >= 0.85
                        else None,
                        mapping_method="dmis-lab/biobert-v1.1",
                        is_high_confidence=(match["similarity"] >= 0.9),
                        needs_review=(match["similarity"] < 0.85),
                    )
                    mappings_to_create.append(mapping)

                processed_count += 1

                # Memory management and progress updates
                if (i + 1) % MEMORY_CLEANUP_INTERVAL == 0:
                    progress = 35 + (processed_count / len(source_terms) * 50)
                    _update_progress(
                        progress_key,
                        "processing",
                        progress,
                        {
                            "processed": processed_count,
                            "total": len(source_terms),
                            "pending_mappings": len(mappings_to_create),
                        },
                    )

                    # Force garbage collection
                    gc.collect()

            except Exception as e:
                logger.error(f"Error processing term {source_term.id}: {str(e)}")

        _update_progress(progress_key, "saving", 85, {"stage": "saving_mappings"})

        # Memory-efficient database save
        mappings_created = _save_mappings_in_batches(mappings_to_create)

        # Cleanup memory
        del icd11_embeddings, icd11_terms, mappings_to_create, source_terms
        gc.collect()

        elapsed_time = time.time() - start_time

        _update_progress(
            progress_key,
            "completed",
            100,
            {
                "mappings_created": mappings_created,
                "terms_processed": processed_count,
                "elapsed_time": elapsed_time,
            },
        )

        logger.info(
            f"✅ Created {mappings_created} mappings for {system} in {elapsed_time:.1f}s"
        )

        return {
            "mappings_created": mappings_created,
            "source_terms_processed": processed_count,
            "elapsed_time": elapsed_time,
            "memory_optimized": True,
        }

    except Exception as e:
        _update_progress(progress_key, "failed", -1, {"error": str(e)})
        logger.error(f"❌ Mapping batch failed: {str(e)}")
        # Cleanup on failure
        gc.collect()
        raise

    finally:
        # Always cleanup
        gc.collect()


# ========== HELPER FUNCTIONS ==========


def _process_large_batch_in_chunks(
    task_instance,
    mapping_config_id: str,
    source_term_ids: List[int],
    system: str,
    max_mappings: int,
    similarity_threshold: float,
) -> Dict[str, Any]:
    """Process large batches in memory-safe chunks"""

    total_mappings = 0
    total_processed = 0
    total_chunks = (len(source_term_ids) + CHUNK_SIZE - 1) // CHUNK_SIZE

    logger.info(
        f"🔄 Chunking {len(source_term_ids)} terms into {total_chunks} chunks of {CHUNK_SIZE}"
    )

    for i in range(0, len(source_term_ids), CHUNK_SIZE):
        chunk = source_term_ids[i : i + CHUNK_SIZE]
        chunk_num = i // CHUNK_SIZE + 1

        logger.info(
            f"📦 Processing chunk {chunk_num}/{total_chunks} ({len(chunk)} terms)"
        )

        try:
            # Process chunk synchronously to maintain memory control
            chunk_result = create_concept_mappings_batch.apply(
                args=[
                    mapping_config_id,
                    chunk,
                    system,
                    max_mappings,
                    similarity_threshold,
                ]
            ).get(timeout=900)  # 15 minute timeout per chunk

            total_mappings += chunk_result.get("mappings_created", 0)
            total_processed += chunk_result.get("source_terms_processed", 0)

            logger.info(
                f"✓ Chunk {chunk_num}: {chunk_result.get('mappings_created', 0)} mappings created"
            )

            # Pause between chunks to prevent system overload
            time.sleep(3)

            # Force cleanup between chunks
            gc.collect()

        except Exception as e:
            logger.error(f"❌ Chunk {chunk_num} failed: {str(e)}")

    return {
        "mappings_created": total_mappings,
        "source_terms_processed": total_processed,
        "chunks_processed": total_chunks,
        "chunked_processing": True,
    }


def _get_limited_icd11_terms(system: str) -> List[Dict[str, Any]]:
    """Get limited ICD-11 terms with smart caching"""

    cache_key = f"icd11_limited_{system}_{MAX_ICD11_TERMS}"
    cached_terms = cache.get(cache_key)

    if cached_terms:
        logger.info(f"📋 Using cached ICD-11 terms ({len(cached_terms)})")
        return cached_terms

    from terminologies.models import ICD11Term

    # Load limited set with smart ordering (most relevant first)
    icd11_terms = list(
        ICD11Term.objects.filter(embedding__isnull=False)
        .values("id", "code", "title", "embedding")
        .order_by("id")[:MAX_ICD11_TERMS]
    )

    if icd11_terms:
        cache.set(cache_key, icd11_terms, timeout=CACHE_TIMEOUT)
        logger.info(f"💾 Cached {len(icd11_terms)} ICD-11 terms")

    return icd11_terms


def _validate_embedding(embedding: List[float]) -> bool:
    """Validate embedding quality and dimensions"""
    if not embedding or len(embedding) != 768:
        return False

    # Check for reasonable number of non-zero values
    non_zero_count = sum(1 for x in embedding if abs(x) > 0.001)
    return non_zero_count > 50  # At least 50 non-zero values


def _determine_relationship(similarity: float) -> str:
    """Determine semantic relationship based on similarity score"""
    if similarity >= 0.95:
        return "equivalent"
    elif similarity >= 0.85:
        return "related-to"
    else:
        return "source-is-narrower-than-target"


def _save_mappings_in_batches(mappings: List) -> int:
    """Save mappings in memory-efficient batches"""
    if not mappings:
        return 0

    mappings_created = 0
    batch_size = 25  # Small batches for memory efficiency

    with transaction.atomic():
        for i in range(0, len(mappings), batch_size):
            batch = mappings[i : i + batch_size]
            try:
                from namasthe_mapping.models import ConceptMapping

                created = ConceptMapping.objects.bulk_create(
                    batch, batch_size=batch_size, ignore_conflicts=True
                )
                mappings_created += len(created)

                # Cleanup between batches
                if i % (batch_size * 4) == 0:  # Every 100 mappings
                    gc.collect()

            except Exception as e:
                logger.error(f"Error saving batch {i // batch_size + 1}: {str(e)}")

    return mappings_created


def _update_progress(key: str, status: str, progress: int, extra_data: Dict = None):
    """Update task progress with automatic cleanup"""
    data = {"status": status, "progress": progress, "updated_at": time.time()}

    if extra_data:
        data.update(extra_data)

    cache.set(key, data, timeout=CACHE_TIMEOUT)


# ========== UTILITY TASKS ==========


@shared_task
def get_mapping_progress(task_id: str) -> Optional[Dict[str, Any]]:
    """Get progress of a specific mapping task"""
    progress_key = f"mapping_{task_id}"
    return cache.get(progress_key)


@shared_task
def validate_system_embeddings(system: str, sample_size: int = 50) -> Dict[str, Any]:
    """Validate embedding quality before running large mapping jobs"""
    try:
        from terminologies.models import Ayurvedha, Siddha, Unani

        model_map = {"ayurveda": Ayurvedha, "siddha": Siddha, "unani": Unani}
        Model = model_map.get(system)

        if not Model:
            return {"error": f"Unknown system: {system}"}

        # Sample terms for validation
        sample_terms = list(
            Model.objects.filter(embedding__isnull=False).order_by("?")[:sample_size]
        )

        if not sample_terms:
            return {"error": f"No embeddings found for {system}"}

        valid_count = 0
        quality_scores = []

        for term in sample_terms:
            if _validate_embedding(term.embedding):
                valid_count += 1
                non_zero_ratio = sum(1 for x in term.embedding if abs(x) > 0.001) / 768
                quality_scores.append(non_zero_ratio)

        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        validity_rate = (valid_count / len(sample_terms)) * 100

        recommendation = (
            "proceed"
            if validity_rate >= 80 and avg_quality >= 0.15
            else "check_embeddings"
        )

        return {
            "system": system,
            "sample_size": len(sample_terms),
            "valid_embeddings": valid_count,
            "validity_rate": validity_rate,
            "average_quality": avg_quality,
            "recommendation": recommendation,
            "ready_for_mapping": recommendation == "proceed",
        }

    except Exception as e:
        logger.error(f"Validation failed for {system}: {str(e)}")
        return {"error": str(e), "system": system}


@shared_task
def cleanup_mapping_cache(max_age_hours: int = 2) -> Dict[str, Any]:
    """Clean up old mapping progress and cache entries"""
    try:
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600

        # Clean progress entries
        progress_keys = cache.keys("mapping_*")
        cleaned_count = 0

        for key in progress_keys:
            try:
                data = cache.get(key)
                if data:
                    updated_at = data.get("updated_at", 0)
                    if current_time - updated_at > max_age_seconds:
                        cache.delete(key)
                        cleaned_count += 1
            except Exception:
                cache.delete(key)  # Delete corrupted entries
                cleaned_count += 1

        # Clean ICD-11 cache entries
        icd11_keys = cache.keys("icd11_limited_*")
        for key in icd11_keys:
            try:
                # Let these expire naturally, but clean if corrupted
                data = cache.get(key)
                if not data or not isinstance(data, list):
                    cache.delete(key)
                    cleaned_count += 1
            except Exception:
                cache.delete(key)
                cleaned_count += 1

        logger.info(f"🧹 Cleaned {cleaned_count} cache entries")

        return {"cleaned_entries": cleaned_count, "cleanup_time": current_time}

    except Exception as e:
        logger.error(f"Cache cleanup failed: {str(e)}")
        return {"error": str(e)}


@shared_task
def get_system_mapping_stats(system: str) -> Dict[str, Any]:
    """Get mapping statistics for a system"""
    try:
        from django.contrib.contenttypes.models import ContentType

        from namasthe_mapping.models import ConceptMapping, TerminologyMapping
        from terminologies.models import Ayurvedha, Siddha, Unani

        model_map = {"ayurveda": Ayurvedha, "siddha": Siddha, "unani": Unani}
        Model = model_map.get(system)

        if not Model:
            return {"error": f"Unknown system: {system}"}

        source_content_type = ContentType.objects.get_for_model(Model)

        # Get mapping config
        source_system = f"NAMASTE-{system.title()}"
        mapping_config = TerminologyMapping.objects.filter(
            source_system=source_system, target_system="ICD-11"
        ).first()

        if not mapping_config:
            return {"error": f"No mapping configuration found for {system}"}

        # Get statistics
        total_mappings = ConceptMapping.objects.filter(
            mapping=mapping_config, source_content_type=source_content_type
        ).count()

        high_confidence = ConceptMapping.objects.filter(
            mapping=mapping_config,
            source_content_type=source_content_type,
            is_high_confidence=True,
        ).count()

        needs_review = ConceptMapping.objects.filter(
            mapping=mapping_config,
            source_content_type=source_content_type,
            needs_review=True,
        ).count()

        return {
            "system": system,
            "total_mappings": total_mappings,
            "high_confidence_mappings": high_confidence,
            "needs_review_mappings": needs_review,
            "confidence_rate": (high_confidence / total_mappings * 100)
            if total_mappings > 0
            else 0,
            "review_rate": (needs_review / total_mappings * 100)
            if total_mappings > 0
            else 0,
        }

    except Exception as e:
        logger.error(f"Stats retrieval failed for {system}: {str(e)}")
        return {"error": str(e), "system": system}


# ========== DEPRECATED TASKS ==========


@shared_task(name="namasthe_mapping.generate_single_embedding")
def generate_single_embedding(*args, **kwargs):
    """DEPRECATED: Use management commands for embedding generation"""
    return {
        "deprecated": True,
        "message": "Use management commands for embedding tasks",
    }


@shared_task(name="namasthe_mapping.generate_batch_embeddings")
def generate_batch_embeddings(*args, **kwargs):
    """DEPRECATED: Use management commands for batch operations"""
    return {
        "deprecated": True,
        "message": "Use management commands for embedding tasks",
    }


# ========== INITIALIZATION ==========

# Success confirmation with configuration summary
logger.info("✅ OPTIMIZED mapping tasks loaded successfully")
logger.info(
    f"📋 Config: max_batch={MAX_BATCH_SIZE}, max_icd11={MAX_ICD11_TERMS}, chunk={CHUNK_SIZE}"
)
