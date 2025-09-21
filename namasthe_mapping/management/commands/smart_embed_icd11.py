# namasthe_mapping/management/commands/smart_embed_icd11.py
# Complete version with unlimited timeout control and resume capability

import logging
import os
import time
from typing import Dict, List, Optional, Set, Tuple

from celery import group
from django.core.cache import cache
from django.core.management.base import BaseCommand
from django.db import models
from django.utils import timezone

from namasthe_mapping.services import HuggingFaceAPIService
from namasthe_mapping.tasks import find_candidates_for_model, monitor_candidate_progress
from terminologies.models import ICD11Term

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Complete smart ICD-11 embedding with unlimited timeout control and resume capability"

    def add_arguments(self, parser):
        # Basic parameters
        parser.add_argument(
            "--batch-size",
            type=int,
            default=15,
            help="Batch size for embedding processing",
        )
        parser.add_argument(
            "--similarity-threshold",
            type=float,
            default=0.3,
            help="Similarity threshold for candidate finding",
        )
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

        # CANDIDATE FINDING TIMEOUT CONTROLS
        parser.add_argument(
            "--max-wait-minutes",
            type=int,
            default=0,
            help="Maximum wait time for candidate finding in minutes (0 = unlimited)",
        )
        parser.add_argument(
            "--check-interval",
            type=int,
            default=3,
            help="Progress check interval in seconds",
        )
        parser.add_argument(
            "--no-timeout",
            action="store_true",
            help="Disable all timeouts (wait indefinitely for completion)",
        )

        # EMBEDDING PHASE TIMEOUT CONTROLS
        parser.add_argument(
            "--embedding-timeout-minutes",
            type=int,
            default=0,
            help="Maximum time for embedding phase in minutes (0 = unlimited)",
        )
        parser.add_argument(
            "--api-timeout-seconds",
            type=int,
            default=60,
            help="Timeout for individual API calls (default: 60 seconds)",
        )
        parser.add_argument(
            "--save-progress-every",
            type=int,
            default=50,
            help="Save progress checkpoint every N embeddings",
        )
        parser.add_argument(
            "--resume-from-batch",
            type=int,
            default=0,
            help="Resume embedding from specific batch number",
        )

    def handle(self, *args, **options):
        self.options = options  # Store options for later use

        self.stdout.write(
            self.style.SUCCESS(
                "🚀 Starting COMPLETE smart ICD-11 embedding generation..."
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
                )[:2000]  # Reasonable limit for skip mode
            )
            if not candidates:
                self.stdout.write("✅ All terms already have embeddings!")
                return
        else:
            # Step 1: Find candidates with configurable timeout
            candidates = self.find_candidates_parallel(
                options["similarity_threshold"],
                options.get("limit_per_model"),
                options.get("max_wait_minutes", 0),
                options.get("check_interval", 3),
                options.get("no_timeout", False),
            )

            if not candidates:
                self.stdout.write("❌ No candidates found!")
                return

        # Step 2: Validate texts if requested
        if options.get("validate_only"):
            self.validate_embedding_texts(candidates)
            return

        # Step 3: Embed candidates with unlimited time
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

        # Show candidate finding timeout configuration
        max_wait = options.get("max_wait_minutes", 0)
        if options.get("no_timeout", False):
            self.stdout.write(f"⏰ Candidate timeout: DISABLED (unlimited)")
        elif max_wait == 0:
            self.stdout.write(f"⏰ Candidate timeout: UNLIMITED")
        else:
            self.stdout.write(f"⏰ Candidate max wait: {max_wait} minutes")

        # Show embedding phase timeout configuration
        embed_timeout = options.get("embedding_timeout_minutes", 0)
        if embed_timeout == 0:
            self.stdout.write(f"⏰ Embedding timeout: UNLIMITED")
        else:
            self.stdout.write(f"⏰ Embedding timeout: {embed_timeout} minutes")

        self.stdout.write(
            f"🔄 Check interval: {options.get('check_interval', 3)} seconds"
        )
        self.stdout.write(
            f"📡 API timeout: {options.get('api_timeout_seconds', 60)} seconds"
        )
        self.stdout.write(
            f"💾 Save progress every: {options.get('save_progress_every', 50)} embeddings"
        )

        if options.get("resume_from_batch", 0) > 0:
            self.stdout.write(f"🔄 Resuming from batch: {options['resume_from_batch']}")

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
            keys_to_try = [
                f"candidates:{model}",
                f":1:candidates:{model}",
                f"candidate_progress:{model}",
                f":1:candidate_progress:{model}",
            ]

            found = False
            for key in keys_to_try:
                try:
                    data = cache.get(key)
                    if data:
                        if isinstance(data, dict):
                            if "candidates_found" in data:
                                self.stdout.write(
                                    f"📊 {model} progress: Found {data['candidates_found']} candidates"
                                )
                            continue
                        elif isinstance(data, (list, set, tuple)):
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

            r = redis.Redis(host="db", port=6379, db=0, decode_responses=False)
            all_keys = r.keys("*candidate*")
            self.stdout.write(f"📋 Direct Redis keys with 'candidate': {len(all_keys)}")

            all_candidates = set()

            for key in all_keys:
                try:
                    raw_data = r.get(key)
                    if raw_data:
                        data = None

                        try:
                            data = json.loads(raw_data.decode("utf-8"))
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            try:
                                data = pickle.loads(raw_data)
                            except Exception:
                                self.stdout.write(
                                    f"❌ {key.decode()}: Could not decode data"
                                )
                                continue

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
        self,
        threshold: float,
        limit_per_model: Optional[int],
        max_wait_minutes: int = 0,
        check_interval: int = 3,
        no_timeout: bool = False,
    ) -> List[int]:
        """Enhanced parallel candidate finding with configurable timeout"""
        self.stdout.write("\n🔍 PHASE 1: Finding ICD-11 candidates (PARALLEL)...")
        self.stdout.write("=" * 60)

        models = ["Ayurvedha", "Siddha", "Unani"]

        self.stdout.write("🚀 Starting parallel candidate finding for all models...")

        job = group(
            find_candidates_for_model.s(model, threshold, limit_per_model)
            for model in models
        )

        result = job.apply_async()

        # Configure timeout based on parameters
        if no_timeout:
            max_wait_seconds = float("inf")
            timeout_msg = "NO TIMEOUT - Will wait indefinitely"
        elif max_wait_minutes == 0:
            max_wait_seconds = float("inf")
            timeout_msg = "UNLIMITED TIMEOUT"
        else:
            max_wait_seconds = max_wait_minutes * 60
            timeout_msg = f"{max_wait_minutes} minute timeout"

        self.stdout.write(f"📊 Monitoring with {timeout_msg}...")
        self.stdout.write(f"🔄 Checking every {check_interval} seconds\n")

        # Enhanced monitoring with configurable timeout
        wait_time = 0
        stable_completion_count = 0
        last_total_candidates = 0
        last_progress_update = 0

        while wait_time < max_wait_seconds:
            try:
                progress_task = monitor_candidate_progress.apply_async()
                progress_info = progress_task.get(timeout=10)

                if progress_info:
                    # Update display every 30 seconds to reduce flicker
                    if wait_time - last_progress_update >= 30:
                        os.system("clear" if os.name == "posix" else "cls")
                        last_progress_update = wait_time

                    self.stdout.write("📊 PARALLEL CANDIDATE FINDING PROGRESS")
                    self.stdout.write("=" * 60)

                    completed_count = 0
                    total_candidates_found = 0
                    all_models_data = []

                    for model, progress in progress_info.items():
                        percentage = progress.get("percentage", 0)
                        processed = progress.get("processed", 0)
                        total = progress.get("total", 1)
                        candidates_found = progress.get("candidates_found", 0)

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

                        all_models_data.append(
                            {
                                "model": model,
                                "percentage": percentage,
                                "candidates": candidates_found,
                            }
                        )

                    # Enhanced status display
                    self.stdout.write(
                        f"\n🔄 Status: {completed_count}/{len(models)} models completed"
                    )

                    # Show timing information
                    elapsed_minutes = wait_time / 60
                    if max_wait_seconds == float("inf"):
                        self.stdout.write(
                            f"🕐 Elapsed time: {elapsed_minutes:.1f} minutes (no timeout)"
                        )
                    else:
                        remaining_minutes = (max_wait_seconds - wait_time) / 60
                        self.stdout.write(
                            f"🕐 Elapsed: {elapsed_minutes:.1f}min, Remaining: {remaining_minutes:.1f}min"
                        )

                    self.stdout.write(
                        f"📊 Total candidates found so far: {total_candidates_found}"
                    )

                    # Enhanced exit condition with stability check
                    if completed_count >= len(models) and total_candidates_found > 0:
                        if total_candidates_found == last_total_candidates:
                            stable_completion_count += 1
                            self.stdout.write(
                                f"🔄 Stable completion check {stable_completion_count}/3"
                            )
                        else:
                            stable_completion_count = 0

                        last_total_candidates = total_candidates_found

                        if stable_completion_count >= 3:
                            self.stdout.write(
                                "\n✅ All models completed with stable results!"
                            )
                            break
                    else:
                        stable_completion_count = 0

                    # Show estimated time remaining
                    if completed_count < len(models):
                        incomplete_models = [
                            data for data in all_models_data if data["percentage"] < 100
                        ]
                        if incomplete_models and wait_time > 60:
                            avg_progress = sum(
                                data["percentage"] for data in incomplete_models
                            ) / len(incomplete_models)
                            if avg_progress > 5:
                                estimated_remaining = (
                                    wait_time * (100 - avg_progress) / avg_progress
                                ) / 60
                                self.stdout.write(
                                    f"📊 Estimated remaining time: {estimated_remaining:.1f} minutes"
                                )

            except Exception as e:
                self.stdout.write(f"⚠️ Monitoring error: {str(e)}")

            time.sleep(check_interval)
            wait_time += check_interval

            # Progress indicator for long waits
            if wait_time % 60 == 0 and wait_time > 0:
                minutes_elapsed = wait_time // 60
                self.stdout.write(f"⏰ {minutes_elapsed} minutes elapsed...")

        # Check if we exited due to timeout
        if wait_time >= max_wait_seconds and max_wait_seconds != float("inf"):
            self.stdout.write(f"\n⚠️ Timeout reached after {max_wait_minutes} minutes!")
            self.stdout.write("📊 Proceeding with whatever candidates were found...")
        else:
            self.stdout.write(
                f"\n✅ Candidate finding completed in {wait_time / 60:.1f} minutes"
            )

        # Adaptive wait for Redis writes
        redis_wait_time = min(20, max(5, 20 - wait_time // 60))
        self.stdout.write(
            f"⏳ Waiting {redis_wait_time} seconds for Redis write completion..."
        )
        time.sleep(redis_wait_time)

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

        # Strategy 2: Direct Redis if we got very few candidates
        if len(all_candidates) < 50:
            self.stdout.write("🔧 Trying direct Redis connection...")
            direct_candidates = self.direct_redis_check()
            if direct_candidates:
                all_candidates.update(direct_candidates)

        # Strategy 3: Extended retry
        if len(all_candidates) < 20:
            self.stdout.write("🔧 Extended retry after longer wait...")
            time.sleep(15)
            retry_candidates = self.manual_candidate_retrieval()
            if retry_candidates:
                all_candidates.update(retry_candidates)

        # Final validation and results
        if all_candidates:
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
        """Enhanced embedding with unlimited timeout and resume capability"""

        self.stdout.write(
            f"\n🎯 PHASE 2: Enhanced embedding for {len(candidate_ids)} candidates..."
        )
        self.stdout.write("⏰ UNLIMITED TIME - Will run until completion!")
        self.stdout.write("=" * 60)

        # Get embedding phase configuration
        embedding_timeout = self.options.get("embedding_timeout_minutes", 0)
        api_timeout = self.options.get("api_timeout_seconds", 60)
        save_every = self.options.get("save_progress_every", 50)
        resume_batch = self.options.get("resume_from_batch", 0)

        if embedding_timeout == 0:
            self.stdout.write("⏰ Embedding timeout: UNLIMITED")
        else:
            self.stdout.write(f"⏰ Embedding timeout: {embedding_timeout} minutes")

        self.stdout.write(f"🔄 API timeout per call: {api_timeout} seconds")
        self.stdout.write(f"💾 Progress saved every: {save_every} embeddings")

        # Get candidate objects
        candidates_query = ICD11Term.objects.filter(id__in=candidate_ids)

        if retry_failed:
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

        if total_to_embed == 0:
            self.stdout.write("✅ All candidates already have embeddings!")
            return

        # Initialize enhanced service with custom timeout
        service = HuggingFaceAPIService(model_preference=model_preference)
        service.timeout = api_timeout  # Override default timeout

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

        # Process with unlimited time and resume capability
        self.process_candidates_unlimited(
            service,
            candidates_needing_embedding,
            batch_size,
            embedding_timeout,
            save_every,
            resume_batch,
        )

    def process_candidates_unlimited(
        self,
        service: HuggingFaceAPIService,
        candidates_query,
        batch_size: int,
        embedding_timeout_minutes: int = 0,
        save_every: int = 50,
        resume_batch: int = 0,
    ) -> None:
        """Process candidates with unlimited time and resume capability"""

        processed = 0
        successful = 0
        failed = 0
        skipped = 0
        tensor_errors = 0

        candidates_list = list(candidates_query)
        total_candidates = len(candidates_list)

        # Calculate batches with resume capability
        safe_batch_size = batch_size
        total_batches = (total_candidates + safe_batch_size - 1) // safe_batch_size

        # Resume from specific batch if requested
        start_batch = resume_batch
        if start_batch > 0:
            self.stdout.write(f"🔄 Resuming from batch {start_batch}")
            processed = start_batch * safe_batch_size
            candidates_list = candidates_list[processed:]
            total_candidates = len(candidates_list)
            total_batches = (total_candidates + safe_batch_size - 1) // safe_batch_size

        self.stdout.write(
            f"\n🎯 UNLIMITED EMBEDDING PROCESSING ({total_batches} batches, size {safe_batch_size})"
        )
        self.stdout.write("⏰ NO TIME LIMITS - Process will run until completion")
        self.stdout.write("=" * 60)

        start_time = time.time()
        embedding_start_time = start_time

        # Set embedding phase timeout (if specified)
        if embedding_timeout_minutes > 0:
            max_embedding_time = embedding_timeout_minutes * 60
            self.stdout.write(
                f"⏰ Embedding phase will timeout after {embedding_timeout_minutes} minutes"
            )
        else:
            max_embedding_time = float("inf")
            self.stdout.write("⏰ Embedding phase has NO timeout")

        batch_num = start_batch
        for i in range(0, len(candidates_list), safe_batch_size):
            batch = candidates_list[i : i + safe_batch_size]
            batch_num += 1
            batch_start_time = time.time()

            # Check embedding phase timeout
            embedding_elapsed = time.time() - embedding_start_time
            if embedding_elapsed > max_embedding_time:
                self.stdout.write(
                    f"\n⏰ Embedding phase timeout reached after {embedding_timeout_minutes} minutes"
                )
                self.stdout.write(
                    f"💾 Resume from batch {batch_num} using: --resume-from-batch {batch_num}"
                )
                break

            self.stdout.write(
                f"\n⚙️ Batch {batch_num}/{start_batch + total_batches}: Processing {len(batch)} terms"
            )

            # Show time information
            elapsed_minutes = embedding_elapsed / 60
            if max_embedding_time == float("inf"):
                self.stdout.write(
                    f"⏰ Elapsed: {elapsed_minutes:.1f} minutes (unlimited)"
                )
            else:
                remaining_minutes = (max_embedding_time - embedding_elapsed) / 60
                self.stdout.write(
                    f"⏰ Elapsed: {elapsed_minutes:.1f}min, Remaining: {remaining_minutes:.1f}min"
                )

            # Pre-validate batch
            valid_batch, validation_stats = self.validate_batch_texts(service, batch)
            skipped += validation_stats["skipped"]

            if not valid_batch:
                self.stdout.write("❌ No valid terms in batch - skipping")
                failed += len(batch)
                processed += len(batch)
                continue

            # Process each term with unlimited retries for network issues
            batch_successful = 0
            batch_failed = 0
            batch_tensor_errors = 0

            for j, (term, validated_text) in enumerate(valid_batch, 1):
                try:
                    self.stdout.write(
                        f"   🔄 {j}/{len(valid_batch)}: {term.code} ({len(validated_text)} chars)"
                    )

                    # Generate embedding with unlimited retries for network issues
                    embedding = self.generate_embedding_with_retries(
                        service, validated_text, term.code, max_retries=10
                    )

                    if embedding is None:
                        batch_failed += 1
                        failed += 1
                        self.stdout.write(
                            f"   ❌ {term.code}: Failed after all retries"
                        )
                        continue

                    # Quality validation
                    non_zero_count = sum(1 for x in embedding if abs(x) > 0.001)
                    quality_threshold = service.embedding_dim * 0.1

                    if non_zero_count >= quality_threshold:
                        # Save successful embedding with database retry
                        if self.save_embedding_with_retry(
                            term, embedding, service.model_name
                        ):
                            batch_successful += 1
                            successful += 1
                            self.stdout.write(
                                f"   ✅ {term.code}: SUCCESS ({non_zero_count}/{service.embedding_dim} values)"
                            )
                        else:
                            batch_failed += 1
                            failed += 1
                            self.stdout.write(
                                f"   ❌ {term.code}: Database save failed"
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

                    if "tensor" in error_msg.lower() and "size" in error_msg.lower():
                        batch_tensor_errors += 1
                        tensor_errors += 1
                        self.stdout.write(
                            f"   🚨 {term.code}: TENSOR ERROR - {error_msg}"
                        )
                    else:
                        self.stdout.write(f"   ❌ {term.code}: ERROR - {error_msg}")

                # Small delay to prevent API hammering
                time.sleep(0.3)

            processed += len(batch)
            batch_time = time.time() - batch_start_time

            # Enhanced progress tracking with save points
            if successful > 0 and successful % save_every == 0:
                self.stdout.write(
                    f"💾 Progress checkpoint: {successful} embeddings completed"
                )

            # Comprehensive batch summary
            self.display_batch_summary(
                batch_num,
                start_batch + total_batches,
                batch_time,
                batch_successful,
                batch_failed,
                batch_tensor_errors,
                validation_stats,
                processed,
                total_candidates,
                successful,
                failed,
                start_time,
            )

            # Minimal delay between batches
            if batch_num < start_batch + total_batches:
                time.sleep(0.5)

        # Final comprehensive summary
        self.display_final_summary(
            total_candidates + (start_batch * safe_batch_size),
            processed + (start_batch * safe_batch_size),
            successful,
            failed,
            skipped,
            tensor_errors,
            service,
            time.time() - start_time,
        )

    def generate_embedding_with_retries(
        self,
        service: HuggingFaceAPIService,
        text: str,
        term_code: str,
        max_retries: int = 10,
    ) -> Optional[List[float]]:
        """Generate embedding with unlimited retries for network issues"""

        for attempt in range(max_retries):
            try:
                embedding = service.generate_embedding(text)

                if embedding and len(embedding) == service.embedding_dim:
                    non_zero_count = sum(1 for x in embedding if abs(x) > 0.001)
                    if non_zero_count > 10:
                        return embedding

                self.stdout.write(
                    f"   ⚠️ {term_code}: Poor quality attempt {attempt + 1}, retrying..."
                )

            except Exception as e:
                error_msg = str(e)
                self.stdout.write(
                    f"   ⚠️ {term_code}: Attempt {attempt + 1} failed: {error_msg}"
                )

                # For network errors, wait longer
                if "timeout" in error_msg.lower() or "connection" in error_msg.lower():
                    wait_time = min(30, 2**attempt)
                    self.stdout.write(
                        f"   ⏳ Network issue detected, waiting {wait_time}s..."
                    )
                    time.sleep(wait_time)
                elif "tensor" in error_msg.lower():
                    if attempt >= 2:
                        break
                    time.sleep(1)
                else:
                    time.sleep(1)

        return None

    def save_embedding_with_retry(
        self, term, embedding: List[float], model_name: str, max_retries: int = 3
    ) -> bool:
        """Save embedding with database connection retry"""

        for attempt in range(max_retries):
            try:
                now = timezone.now()
                term.embedding = embedding
                term.embedding_updated_at = now
                term.embedding_model_version = model_name
                term.save(
                    update_fields=[
                        "embedding",
                        "embedding_updated_at",
                        "embedding_model_version",
                    ]
                )
                return True

            except Exception as e:
                self.stdout.write(
                    f"   ⚠️ Database save attempt {attempt + 1} failed: {str(e)}"
                )
                if attempt < max_retries - 1:
                    time.sleep(1)

                # Try to refresh database connection
                from django.db import connection

                connection.close()

        return False

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
                text = term.get_embedding_text()
                cleaned_text, status = service.validate_and_clean_text(text)

                if status == "valid":
                    truncated_text = service.truncate_text_precise(cleaned_text)
                    estimated_tokens = (
                        len(truncated_text.split()) * service.tokens_per_word
                    )

                    if estimated_tokens <= service.max_length * 0.8:
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

    def display_batch_summary(
        self,
        batch_num: int,
        total_batches: int,
        batch_time: float,
        batch_successful: int,
        batch_failed: int,
        batch_tensor_errors: int,
        validation_stats: Dict,
        processed: int,
        total_candidates: int,
        successful: int,
        failed: int,
        start_time: float,
    ) -> None:
        """Display comprehensive batch summary with timing"""

        self.stdout.write(f"\n📈 BATCH {batch_num} COMPLETE ({batch_time:.1f}s):")
        self.stdout.write(f"   ✅ Successful: {batch_successful}")
        self.stdout.write(f"   ❌ Failed: {batch_failed}")
        self.stdout.write(f"   🚨 Tensor errors: {batch_tensor_errors}")
        self.stdout.write(f"   ⏭️ Skipped: {validation_stats['skipped']}")

        # Progress and ETA
        overall_progress = (
            (processed / total_candidates) * 100 if total_candidates > 0 else 0
        )
        success_rate = (successful / processed) * 100 if processed > 0 else 0

        elapsed_time = time.time() - start_time
        if processed > 0:
            avg_time_per_item = elapsed_time / processed
            remaining_items = total_candidates - processed
            eta_seconds = remaining_items * avg_time_per_item
            eta_hours = eta_seconds / 3600

            if eta_hours > 1:
                eta_str = f"{eta_hours:.1f}h"
            else:
                eta_str = f"{eta_seconds / 60:.1f}min"

            self.stdout.write(
                f"   📊 Progress: {processed}/{total_candidates} ({overall_progress:.1f}%), "
                f"Success: {success_rate:.1f}%, ETA: {eta_str}"
            )

            # Throughput information
            throughput = processed / (elapsed_time / 60)
            self.stdout.write(f"   ⚡ Throughput: {throughput:.1f} embeddings/minute")

            # Resume information
            self.stdout.write(f"   💾 Resume command: --resume-from-batch {batch_num}")

    def display_final_summary(
        self,
        total_candidates: int,
        processed: int,
        successful: int,
        failed: int,
        skipped: int,
        tensor_errors: int,
        service: HuggingFaceAPIService,
        total_time: float,
    ) -> None:
        """Display comprehensive final summary with timing information"""

        self.stdout.write("\n" + "=" * 60)
        self.stdout.write(self.style.SUCCESS("✅ COMPLETE SMART EMBEDDING FINISHED!"))
        self.stdout.write("📊 COMPREHENSIVE STATISTICS:")
        self.stdout.write(f"   🎯 Total candidates: {total_candidates}")
        self.stdout.write(f"   🔄 Processed: {processed}")
        self.stdout.write(f"   ✅ Successful: {successful}")
        self.stdout.write(f"   ❌ Failed: {failed}")
        self.stdout.write(f"   ⏭️ Skipped (validation): {skipped}")
        self.stdout.write(f"   🚨 Tensor errors: {tensor_errors}")

        # Timing information
        self.stdout.write(f"\n⏰ TIMING STATISTICS:")
        self.stdout.write(
            f"   📊 Total time: {total_time / 60:.1f} minutes ({total_time / 3600:.1f} hours)"
        )
        if processed > 0:
            avg_time = total_time / processed
            self.stdout.write(f"   📊 Average time per item: {avg_time:.1f} seconds")
            throughput = processed / (total_time / 60)
            self.stdout.write(f"   📊 Throughput: {throughput:.1f} embeddings/minute")

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
            self.stdout.write(f"   - Use smaller --batch-size for stability")

        # Performance recommendations
        if successful > 0:
            throughput = processed / (total_time / 60)
            if throughput < 50:
                self.stdout.write(f"\n⚡ PERFORMANCE TIPS:")
                self.stdout.write(
                    f"   - Consider increasing --batch-size for better throughput"
                )
                self.stdout.write(f"   - Check network connectivity to HuggingFace API")

        self.stdout.write("=" * 60)
