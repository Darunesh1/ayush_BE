# namasthe_mapping/management/commands/smart_embed_icd11.py
# Enhanced with robust error handling and tensor size prevention

import logging
import os
import time
from typing import Dict, List, Optional, Set, Tuple

from celery import group
from django.core.cache import cache
from django.core.management.base import BaseCommand
from django.utils import timezone

from namasthe_mapping.services import HuggingFaceAPIService
from namasthe_mapping.tasks import find_candidates_for_model, monitor_candidate_progress
from terminologies.models import ICD11Term

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Enhanced smart ICD-11 embedding with comprehensive error prevention"

    def add_arguments(self, parser):
        parser.add_argument(
            "--batch-size",
            type=int,
            default=5,
            help="Batch size (reduced for stability)",
        )
        parser.add_argument("--similarity-threshold", type=float, default=0.3)
        parser.add_argument(
            "--limit-per-model", type=int, help="Limit terms per model for testing"
        )
        parser.add_argument(
            "--retry-failed",
            action="store_true",
            help="Retry previously failed embeddings",
        )
        parser.add_argument(
            "--skip-candidates",
            action="store_true",
            help="Skip candidate finding phase",
        )
        parser.add_argument(
            "--model-preference",
            type=str,
            default="biobert",
            choices=["biobert", "tinybert", "minibert"],
            help="Model to use",
        )
        parser.add_argument(
            "--validate-only",
            action="store_true",
            help="Only validate texts without embedding",
        )

    def handle(self, *args, **options):
        self.stdout.write(
            self.style.SUCCESS(
                "🚀 Starting ENHANCED smart ICD-11 embedding generation..."
            )
        )

        # Show configuration
        self.show_configuration(options)

        # Skip candidate finding if requested
        if options.get("skip_candidates"):
            self.stdout.write("⏭️ Skipping candidate finding phase...")
            # Use all ICD-11 terms that need embeddings
            candidates = list(
                ICD11Term.objects.filter(embedding__isnull=True).values_list(
                    "id", flat=True
                )[:500]
            )
            if not candidates:
                self.stdout.write("✅ All terms already have embeddings!")
                return
        else:
            # Step 1: Find candidates
            candidates = self.find_candidates_parallel(
                options["similarity_threshold"], options.get("limit_per_model")
            )

            if not candidates:
                self.stdout.write("❌ No candidates found!")
                return

        # Step 2: Validate texts if requested
        if options.get("validate_only"):
            self.validate_embedding_texts(candidates)
            return

        # Step 3: Embed candidates
        self.embed_candidates_enhanced(
            candidates,
            options["batch_size"],
            options.get("retry_failed", False),
            options.get("model_preference", "biobert"),
        )

    def show_configuration(self, options: Dict) -> None:
        """Display current configuration"""
        self.stdout.write("\n🔧 CONFIGURATION:")
        self.stdout.write("=" * 50)
        self.stdout.write(f"📊 Batch size: {options['batch_size']}")
        self.stdout.write(f"🎯 Similarity threshold: {options['similarity_threshold']}")
        self.stdout.write(
            f"🤖 Model preference: {options.get('model_preference', 'biobert')}"
        )
        self.stdout.write(f"🔄 Retry failed: {options.get('retry_failed', False)}")
        self.stdout.write(f"⏭️ Skip candidates: {options.get('skip_candidates', False)}")

        if options.get("limit_per_model"):
            self.stdout.write(f"📏 Limit per model: {options['limit_per_model']}")

    def validate_embedding_texts(self, candidate_ids: List[int]) -> None:
        """Validate texts without actually generating embeddings"""
        self.stdout.write(f"\n🔍 VALIDATING {len(candidate_ids)} candidate texts...")
        self.stdout.write("=" * 60)

        candidates = ICD11Term.objects.filter(id__in=candidate_ids)
        service = HuggingFaceAPIService()

        valid_count = 0
        invalid_count = 0
        problematic_texts = []

        for i, term in enumerate(candidates, 1):
            try:
                text = term.get_embedding_text()
                cleaned_text, status = service.validate_and_clean_text(text)

                if status == "valid":
                    truncated_text = service.truncate_text_precise(cleaned_text)

                    # Estimate final token count
                    estimated_tokens = (
                        len(truncated_text.split()) * service.tokens_per_word
                    )

                    if estimated_tokens <= service.max_length * 0.85:
                        valid_count += 1
                        self.stdout.write(
                            f"   ✅ {term.code}: Valid ({len(text)} → {len(truncated_text)} chars, ~{estimated_tokens:.0f} tokens)"
                        )
                    else:
                        invalid_count += 1
                        problematic_texts.append(
                            (term.code, len(text), estimated_tokens)
                        )
                        self.stdout.write(
                            f"   ⚠️ {term.code}: Too long (~{estimated_tokens:.0f} tokens)"
                        )
                else:
                    invalid_count += 1
                    problematic_texts.append((term.code, len(text) if text else 0, 0))
                    self.stdout.write(f"   ❌ {term.code}: Invalid - {status}")

            except Exception as e:
                invalid_count += 1
                self.stdout.write(f"   ❌ {term.code}: Error - {str(e)}")

        # Summary
        self.stdout.write(f"\n📊 VALIDATION SUMMARY:")
        self.stdout.write(f"   ✅ Valid: {valid_count}")
        self.stdout.write(f"   ❌ Invalid: {invalid_count}")
        self.stdout.write(
            f"   📊 Success rate: {(valid_count / (valid_count + invalid_count)) * 100:.1f}%"
        )

        if problematic_texts:
            self.stdout.write(f"\n⚠️ Most problematic texts:")
            for code, orig_len, tokens in sorted(
                problematic_texts, key=lambda x: x[2], reverse=True
            )[:5]:
                self.stdout.write(f"   {code}: {orig_len} chars, ~{tokens:.0f} tokens")

    def manual_candidate_retrieval(self) -> List[int]:
        """Enhanced manual candidate retrieval with better error handling"""
        self.stdout.write("\n🔧 MANUAL CANDIDATE RETRIEVAL:")

        models = ["Ayurvedha", "Siddha", "Unani"]
        all_candidates = set()

        for model in models:
            # Try multiple possible key formats
            keys_to_try = [
                f"candidates:{model}",
                f":1:candidates:{model}",  # Django cache prefix
                f"candidate_progress:{model}",
                f":1:candidate_progress:{model}",
            ]

            found = False
            for key in keys_to_try:
                try:
                    data = cache.get(key)
                    if data:
                        if isinstance(data, dict):
                            # This might be progress data with candidates info
                            if "candidates_found" in data:
                                self.stdout.write(
                                    f"📊 {model} progress: Found {data['candidates_found']} candidates"
                                )
                            continue
                        elif isinstance(data, (list, set, tuple)):
                            # Validate candidate IDs
                            valid_candidates = [
                                cid for cid in data if isinstance(cid, int) and cid > 0
                            ]
                            all_candidates.update(valid_candidates)
                            self.stdout.write(
                                f"✅ {model}: {len(valid_candidates)} valid candidates from {key}"
                            )
                            found = True
                            break
                        else:
                            self.stdout.write(
                                f"⚠️ {model}: Unexpected data type {type(data)} in {key}"
                            )

                except Exception as exc:
                    self.stdout.write(f"❌ Error reading {key}: {exc}")

            if not found:
                self.stdout.write(f"❌ {model}: No candidates found in any key")

        self.stdout.write(
            f"📊 Total candidates from manual retrieval: {len(all_candidates)}"
        )
        return list(all_candidates)

    def direct_redis_check(self) -> List[int]:
        """Enhanced direct Redis connection with better parsing"""
        self.stdout.write("\n🔍 DIRECT REDIS CONNECTION CHECK:")

        try:
            import json
            import pickle

            import redis

            # Connect directly to Redis
            r = redis.Redis(host="db", port=6379, db=0, decode_responses=False)

            # Get all keys with candidates
            all_keys = r.keys("*candidate*")
            self.stdout.write(f"📋 Direct Redis keys with 'candidate': {len(all_keys)}")

            all_candidates = set()

            for key in all_keys:
                try:
                    # Get raw data
                    raw_data = r.get(key)
                    if raw_data:
                        data = None

                        # Try JSON first
                        try:
                            data = json.loads(raw_data.decode("utf-8"))
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            # Try pickle (Django cache format)
                            try:
                                data = pickle.loads(raw_data)
                            except Exception:
                                self.stdout.write(
                                    f"❌ {key.decode()}: Could not decode data"
                                )
                                continue

                        # Extract candidates
                        if isinstance(data, (list, set, tuple)):
                            valid_candidates = [
                                cid for cid in data if isinstance(cid, int) and cid > 0
                            ]
                            all_candidates.update(valid_candidates)
                            self.stdout.write(
                                f"✅ {key.decode()}: {len(valid_candidates)} candidates"
                            )
                        elif isinstance(data, dict) and "candidates_found" in data:
                            self.stdout.write(
                                f"📊 {key.decode()}: Progress data ({data.get('candidates_found', 0)} found)"
                            )
                        else:
                            self.stdout.write(f"📊 {key.decode()}: {type(data)} data")

                except Exception as exc:
                    self.stdout.write(f"❌ Error processing {key}: {exc}")

            return list(all_candidates)

        except Exception as exc:
            self.stdout.write(f"❌ Direct Redis connection failed: {exc}")
            return []

    def find_candidates_parallel(
        self, threshold: float, limit_per_model: Optional[int]
    ) -> List[int]:
        """Enhanced parallel candidate finding with better monitoring"""
        self.stdout.write("\n🔍 PHASE 1: Finding ICD-11 candidates (PARALLEL)...")
        self.stdout.write("=" * 60)

        models = ["Ayurvedha", "Siddha", "Unani"]

        # Start parallel tasks for all models
        self.stdout.write("🚀 Starting parallel candidate finding for all models...")

        job = group(
            find_candidates_for_model.s(model, threshold, limit_per_model)
            for model in models
        )

        result = job.apply_async()

        # Enhanced monitoring with better exit conditions
        self.stdout.write("📊 Monitoring progress with enhanced detection...\n")

        wait_time = 0
        max_wait = 900  # 15 minutes max
        check_interval = 3  # Check every 3 seconds

        stable_completion_count = 0  # Track stable completions
        last_total_candidates = 0

        while wait_time < max_wait:
            try:
                # Monitor progress
                progress_task = monitor_candidate_progress.apply_async()
                progress_info = progress_task.get(timeout=10)

                if progress_info:
                    # Clear screen for better visibility
                    os.system("clear" if os.name == "posix" else "cls")

                    self.stdout.write("📊 PARALLEL CANDIDATE FINDING PROGRESS")
                    self.stdout.write("=" * 60)

                    completed_count = 0
                    total_candidates_found = 0

                    for model, progress in progress_info.items():
                        percentage = progress.get("percentage", 0)
                        processed = progress.get("processed", 0)
                        total = progress.get("total", 1)
                        candidates_found = progress.get("candidates_found", 0)

                        # Enhanced progress bar
                        bar_length = 40
                        filled_length = int(bar_length * percentage / 100)
                        bar = "█" * filled_length + "░" * (bar_length - filled_length)

                        self.stdout.write(
                            f"{model:10} [{bar}] {percentage:6.1f}% "
                            f"({processed}/{total}) - {candidates_found} candidates"
                        )

                        if percentage >= 100:
                            completed_count += 1
                        total_candidates_found += candidates_found

                    self.stdout.write(
                        f"\n🔄 Status: {completed_count}/{len(models)} models completed"
                    )
                    self.stdout.write(f"🕐 Wait time: {wait_time} seconds")
                    self.stdout.write(
                        f"📊 Total candidates found so far: {total_candidates_found}"
                    )

                    # Enhanced exit condition: stability check
                    if completed_count >= len(models) and total_candidates_found > 0:
                        if total_candidates_found == last_total_candidates:
                            stable_completion_count += 1
                        else:
                            stable_completion_count = 0

                        last_total_candidates = total_candidates_found

                        if stable_completion_count >= 3:  # Stable for 3 checks
                            self.stdout.write(
                                "\n✅ All models completed with stable results!"
                            )
                            break
                    else:
                        stable_completion_count = 0

            except Exception as e:
                self.stdout.write(f"⚠️ Monitoring error: {str(e)}")

            # Wait and increment counter
            time.sleep(check_interval)
            wait_time += check_interval

        # Extended wait for Redis writes
        self.stdout.write("\n⏳ Waiting 20 seconds for Redis write completion...")
        time.sleep(20)

        # Enhanced candidate retrieval
        return self.retrieve_candidates_comprehensive()

    def retrieve_candidates_comprehensive(self) -> List[int]:
        """Comprehensive candidate retrieval with multiple strategies"""
        self.stdout.write("\n🔍 COMPREHENSIVE CANDIDATE RETRIEVAL:")
        self.stdout.write("=" * 50)

        all_candidates = set()

        # Strategy 1: Manual retrieval from Redis
        manual_candidates = self.manual_candidate_retrieval()
        if manual_candidates:
            all_candidates.update(manual_candidates)

        # Strategy 2: Direct Redis if needed
        if len(all_candidates) < 50:  # If we got very few candidates
            self.stdout.write("🔧 Trying direct Redis connection...")
            direct_candidates = self.direct_redis_check()
            if direct_candidates:
                all_candidates.update(direct_candidates)

        # Strategy 3: Extended retry with different wait
        if len(all_candidates) < 20:
            self.stdout.write("🔧 Extended retry after longer wait...")
            time.sleep(15)
            retry_candidates = self.manual_candidate_retrieval()
            if retry_candidates:
                all_candidates.update(retry_candidates)

        # Final validation and results
        if all_candidates:
            # Verify candidates exist in database and need embeddings
            valid_candidates = list(
                ICD11Term.objects.filter(
                    id__in=list(all_candidates), embedding__isnull=True
                ).values_list("id", flat=True)
            )

            total_found = len(all_candidates)
            valid_count = len(valid_candidates)

            self.stdout.write("\n" + "=" * 60)
            self.stdout.write(
                self.style.SUCCESS(
                    f"✅ PHASE 1 COMPLETE: Found {total_found} candidates ({valid_count} need embedding)"
                )
            )

            # Show sample
            sample_candidates = list(all_candidates)[:5]
            self.stdout.write(f"🔍 Sample candidates: {sample_candidates}")

            return valid_candidates
        else:
            self.stdout.write("\n" + "=" * 60)
            self.stdout.write(
                self.style.ERROR("❌ PHASE 1 FAILED: No candidates retrieved")
            )
            return []

    def embed_candidates_enhanced(
        self,
        candidate_ids: List[int],
        batch_size: int,
        retry_failed: bool = False,
        model_preference: str = "biobert",
    ) -> None:
        """Enhanced embedding with comprehensive error prevention"""

        self.stdout.write(
            f"\n🎯 PHASE 2: Enhanced embedding for {len(candidate_ids)} candidates..."
        )
        self.stdout.write("=" * 60)

        # Get candidate objects
        candidates_query = ICD11Term.objects.filter(id__in=candidate_ids)

        if retry_failed:
            # Include terms that failed previously (have null or poor embeddings)
            candidates_needing_embedding = candidates_query.filter(
                models.Q(embedding__isnull=True)
                | models.Q(embedding__exact=[0.0] * 768)
            )
        else:
            candidates_needing_embedding = candidates_query.filter(
                embedding__isnull=True
            )

        total_to_embed = candidates_needing_embedding.count()
        already_embedded = len(candidate_ids) - total_to_embed

        self.stdout.write(f"📊 Total candidates: {len(candidate_ids)}")
        self.stdout.write(f"📊 Already embedded: {already_embedded}")
        self.stdout.write(f"📊 Need embedding: {total_to_embed}")

        if retry_failed:
            self.stdout.write("🔄 Including previously failed embeddings")

        # Show efficiency calculation
        total_icd11 = ICD11Term.objects.count()
        efficiency_gain = (
            (total_icd11 - total_to_embed) / total_icd11 * 100 if total_icd11 > 0 else 0
        )
        self.stdout.write(
            f"📊 💰 Efficiency gain: {efficiency_gain:.1f}% less work than embedding all ICD-11 terms!"
        )

        if total_to_embed == 0:
            self.stdout.write("✅ All candidates already have embeddings!")
            return

        # Initialize enhanced service
        service = HuggingFaceAPIService(model_preference=model_preference)

        # Show model information
        model_info = service.get_model_info()
        self.stdout.write(
            f"🤖 Using model: {model_info['model_name']} (dim: {model_info['embedding_dimension']})"
        )

        # Health check
        health = service.health_check()
        if health["status"] != "healthy":
            self.stdout.write(
                f"⚠️ Model health: {health['status']} - {health.get('error', 'degraded performance')}"
            )

        # Process candidates with enhanced error handling
        self.process_candidates_safely(
            service, candidates_needing_embedding, batch_size
        )

    def process_candidates_safely(
        self, service: HuggingFaceAPIService, candidates_query, batch_size: int
    ) -> None:
        """Process candidates with comprehensive safety measures"""

        processed = 0
        successful = 0
        failed = 0
        skipped = 0
        tensor_errors = 0

        candidates_list = list(candidates_query)
        total_candidates = len(candidates_list)

        # Use smaller batch size for safety
        safe_batch_size = min(batch_size, 3)
        total_batches = (total_candidates + safe_batch_size - 1) // safe_batch_size

        self.stdout.write(
            f"\n🎯 SAFE EMBEDDING PROCESSING ({total_batches} batches, size {safe_batch_size})"
        )
        self.stdout.write("=" * 60)

        for i in range(0, total_candidates, safe_batch_size):
            batch = candidates_list[i : i + safe_batch_size]
            batch_num = (i // safe_batch_size) + 1

            self.stdout.write(
                f"\n⚙️ Batch {batch_num}/{total_batches}: Processing {len(batch)} terms"
            )

            # Pre-validate batch
            valid_batch, validation_stats = self.validate_batch_texts(service, batch)

            skipped += validation_stats["skipped"]

            if not valid_batch:
                self.stdout.write("❌ No valid terms in batch - skipping")
                failed += len(batch)
                processed += len(batch)
                continue

            # Process each term individually for maximum safety
            batch_successful = 0
            batch_failed = 0
            batch_tensor_errors = 0

            for j, (term, validated_text) in enumerate(valid_batch, 1):
                try:
                    self.stdout.write(
                        f"   🔄 {j}/{len(valid_batch)}: {term.code} ({len(validated_text)} chars)"
                    )

                    # Generate embedding with comprehensive error handling
                    embedding = service.generate_embedding(validated_text)

                    # Quality validation
                    non_zero_count = sum(1 for x in embedding if abs(x) > 0.001)
                    quality_threshold = (
                        service.embedding_dim * 0.1
                    )  # At least 10% non-zero

                    if non_zero_count >= quality_threshold:
                        # Save successful embedding
                        now = timezone.now()
                        term.embedding = embedding
                        term.embedding_updated_at = now
                        term.embedding_model_version = service.model_name
                        term.save(
                            update_fields=[
                                "embedding",
                                "embedding_updated_at",
                                "embedding_model_version",
                            ]
                        )

                        batch_successful += 1
                        successful += 1
                        self.stdout.write(
                            f"   ✅ {term.code}: SUCCESS ({non_zero_count}/{service.embedding_dim} values)"
                        )

                    else:
                        batch_failed += 1
                        failed += 1
                        self.stdout.write(
                            f"   ❌ {term.code}: Poor quality ({non_zero_count}/{service.embedding_dim} values)"
                        )

                except Exception as e:
                    error_msg = str(e)
                    batch_failed += 1
                    failed += 1

                    # Track tensor-specific errors
                    if "tensor" in error_msg.lower() and "size" in error_msg.lower():
                        batch_tensor_errors += 1
                        tensor_errors += 1
                        self.stdout.write(
                            f"   🚨 {term.code}: TENSOR ERROR - {error_msg}"
                        )
                    else:
                        self.stdout.write(f"   ❌ {term.code}: ERROR - {error_msg}")

                # Small delay between terms for API stability
                time.sleep(1)

            processed += len(batch)

            # Batch summary
            self.stdout.write(f"\n📈 BATCH {batch_num} COMPLETE:")
            self.stdout.write(f"   ✅ Successful: {batch_successful}")
            self.stdout.write(f"   ❌ Failed: {batch_failed}")
            self.stdout.write(f"   🚨 Tensor errors: {batch_tensor_errors}")
            self.stdout.write(f"   ⏭️ Skipped: {validation_stats['skipped']}")

            # Overall progress
            overall_progress = (processed / total_candidates) * 100
            success_rate = (successful / processed) * 100 if processed > 0 else 0

            self.stdout.write(
                f"   📊 Overall: {processed}/{total_candidates} ({overall_progress:.1f}%), Success: {success_rate:.1f}%"
            )

            # Rate limiting between batches
            if batch_num < total_batches:
                self.stdout.write("   ⏳ Cooling down...")
                time.sleep(5)

        # Final comprehensive summary
        self.display_final_summary(
            total_candidates,
            processed,
            successful,
            failed,
            skipped,
            tensor_errors,
            service,
        )

    def validate_batch_texts(
        self, service: HuggingFaceAPIService, batch: List
    ) -> Tuple[List[Tuple], Dict]:
        """Validate texts in batch before processing"""

        valid_batch = []
        validation_stats = {
            "valid": 0,
            "skipped": 0,
            "empty": 0,
            "too_long": 0,
            "invalid_chars": 0,
        }

        for term in batch:
            try:
                # Get and validate embedding text
                text = term.get_embedding_text()
                cleaned_text, status = service.validate_and_clean_text(text)

                if status == "valid":
                    # Additional length validation
                    truncated_text = service.truncate_text_precise(cleaned_text)
                    estimated_tokens = (
                        len(truncated_text.split()) * service.tokens_per_word
                    )

                    if (
                        estimated_tokens <= service.max_length * 0.8
                    ):  # Conservative limit
                        valid_batch.append((term, truncated_text))
                        validation_stats["valid"] += 1
                        self.stdout.write(
                            f"   ✅ {term.code}: Valid (~{estimated_tokens:.0f} tokens)"
                        )
                    else:
                        validation_stats["too_long"] += 1
                        self.stdout.write(
                            f"   ⚠️ {term.code}: Too long (~{estimated_tokens:.0f} tokens) - SKIPPED"
                        )
                else:
                    validation_stats[status] = validation_stats.get(status, 0) + 1
                    validation_stats["skipped"] += 1
                    self.stdout.write(
                        f"   ❌ {term.code}: Invalid ({status}) - SKIPPED"
                    )

            except Exception as e:
                validation_stats["skipped"] += 1
                self.stdout.write(f"   ❌ {term.code}: Validation error - {str(e)}")

        return valid_batch, validation_stats

    def display_final_summary(
        self,
        total_candidates: int,
        processed: int,
        successful: int,
        failed: int,
        skipped: int,
        tensor_errors: int,
        service: HuggingFaceAPIService,
    ) -> None:
        """Display comprehensive final summary"""

        self.stdout.write("\n" + "=" * 60)
        self.stdout.write(self.style.SUCCESS("✅ ENHANCED SMART EMBEDDING COMPLETED!"))
        self.stdout.write("📊 COMPREHENSIVE STATISTICS:")
        self.stdout.write(f"   🎯 Total candidates: {total_candidates}")
        self.stdout.write(f"   🔄 Processed: {processed}")
        self.stdout.write(f"   ✅ Successful: {successful}")
        self.stdout.write(f"   ❌ Failed: {failed}")
        self.stdout.write(f"   ⏭️ Skipped (validation): {skipped}")
        self.stdout.write(f"   🚨 Tensor errors: {tensor_errors}")

        # Calculate rates
        if processed > 0:
            success_rate = (successful / processed) * 100
            failure_rate = (failed / processed) * 100
            tensor_error_rate = (tensor_errors / processed) * 100

            self.stdout.write(f"\n📈 PERFORMANCE METRICS:")
            self.stdout.write(f"   ✅ Success rate: {success_rate:.1f}%")
            self.stdout.write(f"   ❌ Failure rate: {failure_rate:.1f}%")
            self.stdout.write(f"   🚨 Tensor error rate: {tensor_error_rate:.1f}%")

        # Model information
        model_info = service.get_model_info()
        self.stdout.write(f"\n🤖 MODEL USED:")
        self.stdout.write(f"   📝 Model: {model_info['model_name']}")
        self.stdout.write(f"   📏 Dimension: {model_info['embedding_dimension']}")
        self.stdout.write(f"   🏥 Medical optimized: {model_info['medical_optimized']}")

        # Recommendations
        if tensor_errors > 0:
            self.stdout.write(f"\n💡 RECOMMENDATIONS:")
            self.stdout.write(
                f"   - Consider using --model-preference tinybert for problematic texts"
            )
            self.stdout.write(f"   - Run --validate-only first to identify issues")
            self.stdout.write(f"   - Use smaller --batch-size (current: was larger)")

        self.stdout.write("=" * 60)
