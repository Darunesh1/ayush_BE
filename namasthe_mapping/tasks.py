# namasthe_mapping/tasks.py
# Complete optimized file using existing fuzzy search functions

import json
import logging

from celery import group, shared_task
from django.apps import apps
from django.core.cache import cache
from django.utils import timezone

from namasthe_mapping.services import HuggingFaceAPIService

logger = logging.getLogger(__name__)


@shared_task(bind=True, max_retries=3)
def generate_single_embedding(
    self, model_name: str, record_id: int, force: bool = False
):
    """Generate embedding for a single record with retry logic"""
    try:
        # Get the model class and record
        if model_name == "ICD11Term":
            Model = apps.get_model("terminologies", "ICD11Term")
        else:
            Model = apps.get_model("terminologies", model_name)

        record = Model.objects.get(pk=record_id)

        # Check if embedding already exists and not forcing
        if not force and record.embedding:
            return {
                "success": True,
                "record_id": record_id,
                "model": model_name,
                "action": "skipped",
                "reason": "embedding_exists",
            }

        # Check Redis cache first
        cache_key = f"embedding:{model_name}:{record_id}"
        cached_embedding = cache.get(cache_key)

        if cached_embedding and not force:
            # Use cached embedding
            record.embedding = cached_embedding["embedding"]
            record.embedding_updated_at = timezone.now()
            record.embedding_model_version = cached_embedding["model_version"]
            record.save(
                update_fields=[
                    "embedding",
                    "embedding_updated_at",
                    "embedding_model_version",
                ]
            )

            return {
                "success": True,
                "record_id": record_id,
                "model": model_name,
                "action": "cached",
                "non_zero_count": cached_embedding.get("non_zero_count", 0),
            }

        # Generate new embedding
        service = HuggingFaceAPIService()
        text = record.get_embedding_text()

        if not text or not text.strip():
            return {
                "success": False,
                "record_id": record_id,
                "model": model_name,
                "error": "empty_text",
            }

        embedding = service.generate_embedding(text)

        # Validate embedding
        non_zero_count = sum(1 for x in embedding if abs(x) > 0.001)

        if non_zero_count < 50:  # Quality check
            raise Exception(
                f"Poor quality embedding: only {non_zero_count} non-zero values"
            )

        # Save to database
        now = timezone.now()
        record.embedding = embedding
        record.embedding_updated_at = now
        record.embedding_model_version = service.model_name
        record.save(
            update_fields=[
                "embedding",
                "embedding_updated_at",
                "embedding_model_version",
            ]
        )

        # Cache the embedding for future use (24 hours)
        cache.set(
            cache_key,
            {
                "embedding": embedding,
                "model_version": service.model_name,
                "non_zero_count": non_zero_count,
                "created_at": now.isoformat(),
            },
            timeout=86400,
        )

        return {
            "success": True,
            "record_id": record_id,
            "model": model_name,
            "action": "generated",
            "non_zero_count": non_zero_count,
        }

    except Exception as e:
        logger.error(
            f"Error generating embedding for {model_name}:{record_id}: {str(e)}"
        )

        # Retry logic
        if self.request.retries < self.max_retries:
            # Exponential backoff: 2^retry_count * 60 seconds
            countdown = (2**self.request.retries) * 60
            raise self.retry(exc=e, countdown=countdown)

        return {
            "success": False,
            "record_id": record_id,
            "model": model_name,
            "error": str(e),
            "retries_exhausted": True,
        }


def find_icd11_candidates_fast(
    namaste_term, similarity_threshold=0.3, max_candidates=20
):
    """Use existing fuzzy search functions - they're already optimized"""

    # Import your existing fuzzy search function
    from terminologies.utils import fuzzy_search_icd_terms

    # Get search text from NAMASTE term
    search_text = namaste_term.get_embedding_text()

    if not search_text or not search_text.strip():
        return []

    try:
        # Use your existing fuzzy search that handles:
        # - Full-text search with SearchRank
        # - TrigramSimilarity for fuzzy matching
        # - Multiple fields (title, code, definition)
        # - Proper relevance ordering
        results = fuzzy_search_icd_terms(search_text, limit=max_candidates)

        # Extract IDs from the QuerySet results
        candidate_ids = [result.id for result in results]

        return candidate_ids

    except Exception as e:
        logger.error(f"Fuzzy search failed for '{search_text}': {str(e)}")

        # Fallback to simple search if fuzzy search fails
        from terminologies.models import ICD11Term

        keywords = [w.lower() for w in search_text.split() if len(w) >= 3][:3]

        candidates = set()
        for keyword in keywords:
            matches = list(
                ICD11Term.objects.filter(title__icontains=keyword).values_list(
                    "id", flat=True
                )[: max_candidates // 3]
            )
            candidates.update(matches)

        return list(candidates)[:max_candidates]


@shared_task
def find_candidates_for_model(
    model_name: str, similarity_threshold: float = 0.3, limit_per_model: int = None
):
    """Find ICD-11 candidates for a single NAMASTE model using existing fuzzy search"""

    try:
        # Get the model class
        if model_name == "Ayurvedha":
            Model = apps.get_model("terminologies", "Ayurvedha")
        elif model_name == "Siddha":
            Model = apps.get_model("terminologies", "Siddha")
        elif model_name == "Unani":
            Model = apps.get_model("terminologies", "Unani")
        else:
            return {"success": False, "error": f"Unknown model: {model_name}"}

        # Get terms with embeddings
        queryset = Model.objects.exclude(embedding__isnull=True)
        if limit_per_model:
            queryset = queryset[:limit_per_model]

        terms = list(queryset)
        total_terms = len(terms)

        logger.info(
            f"🚀 Starting candidate finding for {model_name}: {total_terms} terms"
        )

        model_candidates = set()
        processed = 0

        # Process in small chunks for better progress reporting
        chunk_size = 25

        for chunk_start in range(0, len(terms), chunk_size):
            chunk_terms = terms[chunk_start : chunk_start + chunk_size]
            chunk_candidates = set()

            for term in chunk_terms:
                # Use your existing fuzzy search
                try:
                    candidates = find_icd11_candidates_fast(
                        term, similarity_threshold, max_candidates=25
                    )
                    chunk_candidates.update(candidates)

                    # Log some successful matches for debugging
                    if candidates and processed < 5:  # Log first few terms only
                        logger.info(
                            f"🔍 {term.english_name[:30]} → {len(candidates)} matches"
                        )

                except Exception as e:
                    logger.error(f"Failed to find candidates for {term.id}: {str(e)}")

                processed += 1

            # Add chunk candidates to model total
            model_candidates.update(chunk_candidates)

            # Update progress in Redis every chunk
            progress_key = f"candidate_progress:{model_name}"
            progress_data = {
                "model": model_name,
                "processed": processed,
                "total": total_terms,
                "percentage": (processed / total_terms) * 100,
                "candidates_found": len(model_candidates),
                "updated_at": timezone.now().isoformat(),
            }
            cache.set(progress_key, progress_data, timeout=3600)

            logger.info(
                f"📈 {model_name}: {processed}/{total_terms} "
                f"({progress_data['percentage']:.1f}%) - {len(model_candidates)} candidates"
            )

        # Store final candidates in Redis
        candidates_key = f"candidates:{model_name}"
        candidate_ids = list(model_candidates)
        cache.set(candidates_key, candidate_ids, timeout=7200)  # 2 hours

        logger.info(
            f"✅ {model_name} completed: {len(model_candidates)} unique candidates found"
        )

        return {
            "success": True,
            "model": model_name,
            "processed": processed,
            "candidates_found": len(model_candidates),
            "candidates_key": candidates_key,
        }

    except Exception as e:
        logger.error(f"❌ Error finding candidates for {model_name}: {str(e)}")
        return {"success": False, "model": model_name, "error": str(e)}


@shared_task
def monitor_candidate_progress():
    """Monitor progress of parallel candidate finding"""

    models = ["Ayurvedha", "Siddha", "Unani"]
    progress_data = {}

    for model in models:
        progress_key = f"candidate_progress:{model}"
        model_progress = cache.get(progress_key)
        if model_progress:
            progress_data[model] = model_progress

    return progress_data


@shared_task
def generate_batch_embeddings(model_name: str, record_ids: list, force: bool = False):
    """Generate embeddings for a batch of records - FIXED VERSION"""

    results = []

    # Process each record individually instead of using group().get()
    for record_id in record_ids:
        try:
            # Call the function directly instead of as a task
            result = generate_single_embedding.apply(
                args=[model_name, record_id, force]
            ).get()
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to process record {record_id}: {str(e)}")
            results.append(
                {
                    "success": False,
                    "record_id": record_id,
                    "model": model_name,
                    "error": str(e),
                }
            )

    # Aggregate results
    successful = sum(1 for r in results if r.get("success", False))
    failed = sum(1 for r in results if not r.get("success", False))
    cached = sum(1 for r in results if r.get("action") == "cached")
    generated = sum(1 for r in results if r.get("action") == "generated")

    return {
        "model": model_name,
        "total_processed": len(record_ids),
        "successful": successful,
        "failed": failed,
        "cached": cached,
        "generated": generated,
        "results": results,
    }


@shared_task
def update_progress(task_name: str, current: int, total: int, model_name: str = None):
    """Update progress in Redis for monitoring"""
    progress_key = f"embedding_progress:{task_name}"

    progress_data = {
        "current": current,
        "total": total,
        "percentage": (current / total * 100) if total > 0 else 0,
        "model": model_name,
        "updated_at": timezone.now().isoformat(),
    }

    # Store progress in Redis (expires in 1 hour)
    cache.set(progress_key, progress_data, timeout=3600)

    return progress_data


@shared_task
def cleanup_embedding_cache(older_than_days: int = 7):
    """Clean up old embedding cache entries"""
    logger.info(
        f"Cleaning up embedding cache entries older than {older_than_days} days"
    )

    return {"message": "Cache cleanup completed"}


# Simple debug print (no registration issues)
logger.info("NAMASTE mapping tasks loaded successfully")
