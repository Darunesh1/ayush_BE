# namasthe_mapping/tasks.py
# Celery tasks for optimized embedding generation

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


@shared_task
def generate_batch_embeddings(model_name: str, record_ids: list, force: bool = False):
    """Generate embeddings for a batch of records"""

    # Create a group of individual embedding tasks
    job = group(
        generate_single_embedding.s(model_name, record_id, force)
        for record_id in record_ids
    )

    # Execute the group
    result = job.apply_async()

    # Wait for all tasks to complete and collect results
    results = result.get()

    # Aggregate results
    successful = sum(1 for r in results if r["success"])
    failed = sum(1 for r in results if not r["success"])
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

    # This would typically use Redis pattern matching
    # Implementation depends on your Redis setup

    logger.info(
        f"Cleaning up embedding cache entries older than {older_than_days} days"
    )

    # Example cleanup logic (adapt to your Redis configuration)
    # This is a simplified version
    return {"message": "Cache cleanup completed"}
