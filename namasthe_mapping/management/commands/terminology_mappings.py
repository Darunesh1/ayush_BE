# management/commands/terminology_mappings.py

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from celery import group
from celery.result import GroupResult
from django.contrib.contenttypes.models import ContentType
from django.core.cache import cache
from django.core.management.base import BaseCommand
from django.db import transaction
from django.db.models import Avg, Count, Q
from django.utils import timezone

# Import only what we need at module level
logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = (
        "Create terminology mappings with advanced progress tracking and optimization"
    )

    def add_arguments(self, parser):
        # Basic parameters
        parser.add_argument(
            "--systems",
            nargs="+",
            default=["ayurveda", "siddha", "unani"],
            help="Source terminology systems to map to ICD-11",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=25,
            help="Batch size for processing (default: 25)",
        )
        parser.add_argument(
            "--similarity-threshold",
            type=float,
            default=0.75,
            help="Minimum similarity threshold (default: 0.75)",
        )
        parser.add_argument(
            "--max-mappings-per-term",
            type=int,
            default=3,
            help="Maximum mappings per term (default: 3)",
        )

        # Progress tracking
        parser.add_argument(
            "--max-wait-minutes",
            type=int,
            default=0,
            help="Maximum wait time in minutes (0 = unlimited)",
        )
        parser.add_argument(
            "--check-interval",
            type=int,
            default=3,
            help="Progress check interval in seconds (default: 3)",
        )
        parser.add_argument(
            "--no-timeout",
            action="store_true",
            help="Disable all timeouts - wait indefinitely",
        )
        parser.add_argument(
            "--save-progress-every",
            type=int,
            default=100,
            help="Save progress every N mappings (default: 100)",
        )

        # Processing options
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Show what would be created without execution",
        )
        parser.add_argument(
            "--force-recreate",
            action="store_true",
            help="Delete existing mappings and recreate",
        )
        parser.add_argument(
            "--workers",
            type=int,
            default=4,
            help="Number of parallel workers (default: 4)",
        )

    def handle(self, *args, **options):
        self.options = options
        start_time = time.time()

        self.stdout.write(
            self.style.SUCCESS(
                "🚀 Starting OPTIMIZED terminology mapping with progress tracking"
            )
        )

        # Show configuration
        self.show_configuration(options)

        # Validate prerequisites
        if not self.validate_prerequisites():
            return

        # Process all systems
        total_results = self.process_all_systems(options, start_time)

        # Display final summary
        self.display_final_summary(total_results, time.time() - start_time)

    def show_configuration(self, options: Dict) -> None:
        """Display comprehensive configuration"""
        self.stdout.write("\n📋 CONFIGURATION")
        self.stdout.write("=" * 60)
        self.stdout.write(f"  • Systems: {', '.join(options['systems'])}")
        self.stdout.write(f"  • Batch size: {options['batch_size']}")
        self.stdout.write(
            f"  • Similarity threshold: {options['similarity_threshold']}"
        )
        self.stdout.write(
            f"  • Max mappings per term: {options['max_mappings_per_term']}"
        )
        self.stdout.write(f"  • Workers: {options['workers']}")

        # Timeout configuration
        max_wait = options.get("max_wait_minutes", 0)
        if options.get("no_timeout"):
            self.stdout.write("  • Timeout: DISABLED (unlimited)")
        elif max_wait == 0:
            self.stdout.write("  • Timeout: UNLIMITED")
        else:
            self.stdout.write(f"  • Max wait: {max_wait} minutes")

        self.stdout.write(f"  • Check interval: {options['check_interval']} seconds")
        self.stdout.write(
            f"  • Progress saves: every {options['save_progress_every']} mappings"
        )
        self.stdout.write(f"  • Dry run: {options['dry_run']}")
        self.stdout.write(f"  • Force recreate: {options['force_recreate']}")

    def validate_prerequisites(self) -> bool:
        """Validate prerequisites"""
        # Import models inside function
        from terminologies.models import Ayurvedha, ICD11Term, Siddha, Unani

        self.stdout.write("\n🔍 VALIDATING PREREQUISITES")
        self.stdout.write("=" * 50)

        # Check ICD-11 embeddings
        icd11_count = ICD11Term.objects.filter(embedding__isnull=False).count()
        if icd11_count == 0:
            self.stdout.write(
                self.style.ERROR("❌ No ICD-11 terms with embeddings found")
            )
            return False

        self.stdout.write(f"✅ ICD-11 terms with embeddings: {icd11_count}")

        # Check NAMASTE embeddings
        model_map = {"ayurveda": Ayurvedha, "siddha": Siddha, "unani": Unani}

        for system in self.options["systems"]:
            Model = model_map[system]

            total = Model.objects.count()
            embedded = Model.objects.filter(embedding__isnull=False).count()

            self.stdout.write(
                f"  • {system.upper()}: {embedded}/{total} "
                f"({embedded / total * 100:.1f}%) embedded"
            )

        # Test Redis connection
        try:
            cache.set("test_mapping_connection", "ok", timeout=10)
            if cache.get("test_mapping_connection") == "ok":
                self.stdout.write("✅ Redis connection verified")
                cache.delete("test_mapping_connection")
            else:
                self.stdout.write("⚠️  Redis connection issue")
        except Exception as e:
            self.stdout.write(f"⚠️  Redis error: {str(e)}")

        return True

    def process_all_systems(self, options: Dict, start_time: float) -> Dict[str, Any]:
        """Process all terminology systems"""

        self.stdout.write(f"\n🔄 PROCESSING {len(options['systems'])} SYSTEMS")
        self.stdout.write("=" * 60)

        # Cache ICD-11 embeddings for performance
        self.cache_icd11_embeddings()

        results = {
            "configs_created": 0,
            "total_mappings": 0,
            "successful_systems": [],
            "failed_systems": [],
            "system_stats": {},
            "processing_times": {},
        }

        for i, system in enumerate(options["systems"], 1):
            system_start = time.time()

            self.stdout.write(
                f"\n{'=' * 20} SYSTEM {i}/{len(options['systems'])}: {system.upper()} {'=' * 20}"
            )

            try:
                # Create mapping configuration
                config = self.create_mapping_config(system, options)
                if not config:
                    results["failed_systems"].append(system)
                    continue

                results["configs_created"] += 1

                # Process mappings with progress tracking
                system_result = self.process_system_mappings(config, system, options)

                results["total_mappings"] += system_result["mappings_created"]
                results["successful_systems"].append(system)
                results["system_stats"][system] = system_result
                results["processing_times"][system] = time.time() - system_start

                self.stdout.write(
                    self.style.SUCCESS(
                        f"✅ {system.upper()}: {system_result['mappings_created']} mappings created"
                    )
                )

                # Update statistics
                if not options["dry_run"] and hasattr(config, "update_statistics"):
                    config.update_statistics()
                    self.display_system_stats(config)

            except Exception as e:
                results["failed_systems"].append(system)
                self.stdout.write(
                    self.style.ERROR(f"❌ Error processing {system}: {str(e)}")
                )
                logger.error(
                    f"System processing error for {system}: {str(e)}", exc_info=True
                )

        return results

    def cache_icd11_embeddings(self) -> None:
        """Cache ICD-11 embeddings for performance"""
        from terminologies.models import ICD11Term

        cache_key = "icd11_embeddings_mapping"

        if cache.get(cache_key):
            self.stdout.write("✅ Using cached ICD-11 embeddings")
            return

        self.stdout.write("💾 Caching ICD-11 embeddings...")

        icd11_terms = list(
            ICD11Term.objects.filter(embedding__isnull=False).values(
                "id", "code", "title", "embedding"
            )
        )

        if icd11_terms:
            cache.set(cache_key, icd11_terms, timeout=3600)  # 1 hour
            self.stdout.write(f"✅ Cached {len(icd11_terms)} ICD-11 terms")

    def create_mapping_config(self, system: str, options: Dict):
        """Create or get mapping configuration"""
        from namasthe_mapping.models import TerminologyMapping

        source_system = f"NAMASTE-{system.title()}"
        config_name = f"{source_system} to ICD-11 Mapping"

        if options["dry_run"]:
            self.stdout.write(f"  🔍 DRY RUN: Would create '{config_name}'")

            class MockConfig:
                def __init__(self):
                    self.name = config_name
                    self.id = "mock-id"
                    self.similarity_threshold = options["similarity_threshold"]

            return MockConfig()

        try:
            # Check existing
            existing = TerminologyMapping.objects.filter(
                source_system=source_system, target_system="ICD-11"
            ).first()

            if existing and options["force_recreate"]:
                self.stdout.write("  🗑️  Deleting existing mapping...")
                existing.delete()
                existing = None

            if existing:
                self.stdout.write(f"  ♻️  Using existing: {existing.name}")
                # Update parameters
                existing.similarity_threshold = options["similarity_threshold"]
                existing.status = "active"
                existing.save(update_fields=["similarity_threshold", "status"])
                return existing

            # Create new
            with transaction.atomic():
                config = TerminologyMapping.objects.create(
                    name=config_name,
                    description=f"Automated mapping from {source_system} to ICD-11",
                    source_system=source_system,
                    target_system="ICD-11",
                    similarity_threshold=options["similarity_threshold"],
                    status="active",
                    is_active=True,
                    created_by="system-auto",
                )

                self.stdout.write(f"  ✨ Created: {config.name}")
                return config

        except Exception as e:
            self.stdout.write(f"  ❌ Config creation error: {str(e)}")
            return None

    def process_system_mappings(
        self, config, system: str, options: Dict
    ) -> Dict[str, Any]:
        """Process mappings for a system with progress tracking"""

        # Get source term statistics
        stats = self.get_source_stats(config, system, options["dry_run"])

        if stats["to_process"] == 0:
            self.stdout.write(f"  ℹ️  No {system} terms need processing")
            return {"mappings_created": 0, "terms_processed": 0}

        self.stdout.write(f"  📊 Processing {stats['to_process']} {system} terms")

        if options["dry_run"]:
            estimated = stats["to_process"] * options["max_mappings_per_term"]
            self.stdout.write(f"  🔍 DRY RUN: Would create ~{estimated} mappings")
            return {
                "mappings_created": estimated,
                "terms_processed": stats["to_process"],
            }

        # Process with Celery and monitoring
        return self.process_with_celery_tracking(config, stats, system, options)

    def get_source_stats(self, config, system: str, dry_run: bool) -> Dict[str, int]:
        """Get source term statistics"""
        from django.contrib.contenttypes.models import ContentType

        from namasthe_mapping.models import ConceptMapping
        from terminologies.models import Ayurvedha, Siddha, Unani

        model_map = {"ayurveda": Ayurvedha, "siddha": Siddha, "unani": Unani}
        Model = model_map[system]

        # Get existing mappings
        if not dry_run and hasattr(config, "id"):
            existing_ids = set(
                ConceptMapping.objects.filter(
                    mapping=config,
                    source_content_type=ContentType.objects.get_for_model(Model),
                ).values_list("source_object_id", flat=True)
            )
        else:
            existing_ids = set()

        total = Model.objects.count()
        embedded = Model.objects.filter(embedding__isnull=False).count()

        to_process = (
            Model.objects.filter(embedding__isnull=False)
            .exclude(id__in=existing_ids)
            .count()
        )

        return {
            "total": total,
            "embedded": embedded,
            "existing": len(existing_ids),
            "to_process": to_process,
        }

    def process_with_celery_tracking(
        self, config, stats: Dict[str, int], system: str, options: Dict
    ) -> Dict[str, Any]:
        """Process with Celery and advanced progress tracking"""
        from django.contrib.contenttypes.models import ContentType

        from namasthe_mapping.models import ConceptMapping
        from namasthe_mapping.tasks import create_concept_mappings_batch
        from terminologies.models import Ayurvedha, Siddha, Unani

        batch_size = options["batch_size"]
        total_batches = (stats["to_process"] + batch_size - 1) // batch_size

        self.stdout.write(
            f"  📦 Creating {total_batches} batches of ~{batch_size} terms"
        )

        # Get source term IDs
        model_map = {"ayurveda": Ayurvedha, "siddha": Siddha, "unani": Unani}
        Model = model_map[system]

        existing_ids = set(
            ConceptMapping.objects.filter(
                mapping=config,
                source_content_type=ContentType.objects.get_for_model(Model),
            ).values_list("source_object_id", flat=True)
        )

        source_ids = list(
            Model.objects.filter(embedding__isnull=False)
            .exclude(id__in=existing_ids)
            .values_list("id", flat=True)
        )

        # Create Celery tasks
        tasks = []
        for i in range(0, len(source_ids), batch_size):
            batch_ids = source_ids[i : i + batch_size]
            task = create_concept_mappings_batch.s(
                str(config.id),
                batch_ids,
                system,
                options["max_mappings_per_term"],
                options["similarity_threshold"],
            )
            tasks.append(task)

        self.stdout.write(f"  🚀 Launching {len(tasks)} Celery tasks...")

        # Execute and monitor
        job_group = group(tasks)
        group_result = job_group.apply_async()

        return self.monitor_progress_enhanced(
            group_result, total_batches, system, options
        )

    def monitor_progress_enhanced(
        self, group_result: GroupResult, total_batches: int, system: str, options: Dict
    ) -> Dict[str, Any]:
        """Enhanced progress monitoring with real-time updates"""

        max_wait = options.get("max_wait_minutes", 0)
        check_interval = options.get("check_interval", 3)
        no_timeout = options.get("no_timeout", False)

        if no_timeout:
            max_wait_seconds = float("inf")
            timeout_msg = "NO TIMEOUT"
        elif max_wait == 0:
            max_wait_seconds = float("inf")
            timeout_msg = "UNLIMITED"
        else:
            max_wait_seconds = max_wait * 60
            timeout_msg = f"{max_wait}min timeout"

        self.stdout.write(f"  ⏱️  Monitoring with {timeout_msg}")

        start_time = time.time()
        wait_time = 0
        last_update = 0
        stable_count = 0

        total_mappings = 0
        total_terms = 0
        completed_batches = 0

        while wait_time < max_wait_seconds:
            try:
                # Clear screen periodically
                if wait_time - last_update > 30:
                    if os.name == "posix":
                        os.system("clear")
                    last_update = wait_time

                self.stdout.write(f"\n📊 MAPPING PROGRESS - {system.upper()}")
                self.stdout.write("=" * 60)

                # Check task status
                completed = sum(1 for r in group_result.results if r.ready())
                failed = sum(
                    1 for r in group_result.results if r.ready() and r.failed()
                )
                successful = completed - failed
                pending = total_batches - completed

                # Progress calculation
                progress = (completed / total_batches * 100) if total_batches > 0 else 0

                # Progress bar
                bar_length = 50
                filled = int(bar_length * progress / 100)
                bar = "█" * filled + "░" * (bar_length - filled)

                self.stdout.write(f"Progress: [{bar}] {progress:6.1f}%")
                self.stdout.write(
                    f"Batches:  ✅{successful} ❌{failed} ⏳{pending} Total: {total_batches}"
                )

                # Timing information
                elapsed = time.time() - start_time
                if completed > 0:
                    avg_time = elapsed / completed
                    eta_seconds = pending * avg_time

                    if eta_seconds > 3600:
                        eta = f"{eta_seconds / 3600:.1f}h"
                    else:
                        eta = f"{eta_seconds / 60:.1f}min"

                    rate = completed / elapsed * 60  # batches per minute
                    self.stdout.write(
                        f"Timing:   {elapsed / 60:.1f}min elapsed, ETA {eta}"
                    )
                    self.stdout.write(f"Rate:     {rate:.1f} batches/min")

                # Collect results
                current_mappings = 0
                current_terms = 0

                for result in group_result.results:
                    if result.ready() and result.successful():
                        try:
                            task_result = result.get()
                            current_mappings += task_result.get("mappings_created", 0)
                            current_terms += task_result.get(
                                "source_terms_processed", 0
                            )
                        except:
                            pass

                total_mappings = current_mappings
                total_terms = current_terms
                completed_batches = successful

                self.stdout.write(
                    f"Results:  {total_mappings} mappings, {total_terms} terms processed"
                )

                # Timeout warning
                if max_wait_seconds != float("inf"):
                    remaining = (max_wait_seconds - wait_time) / 60
                    self.stdout.write(f"Timeout:  {remaining:.1f}min remaining")

                # Completion check
                if completed == total_batches:
                    stable_count += 1
                    if stable_count >= 3:
                        self.stdout.write("✅ All tasks completed!")
                        break
                else:
                    stable_count = 0

                # Progress checkpoint
                save_every = options.get("save_progress_every", 100)
                if total_mappings > 0 and total_mappings % save_every == 0:
                    cache.set(
                        f"mapping_checkpoint_{system}",
                        {
                            "mappings": total_mappings,
                            "terms": total_terms,
                            "timestamp": time.time(),
                        },
                        timeout=3600,
                    )

            except Exception as e:
                self.stdout.write(f"⚠️  Monitoring error: {str(e)}")

            time.sleep(check_interval)
            wait_time += check_interval

        # Final results
        elapsed = time.time() - start_time

        if wait_time >= max_wait_seconds and max_wait_seconds != float("inf"):
            self.stdout.write(f"⚠️  Timeout reached after {max_wait} minutes")
        else:
            self.stdout.write(f"✅ Completed in {elapsed / 60:.1f} minutes")

        return {
            "mappings_created": total_mappings,
            "terms_processed": total_terms,
            "batches_completed": completed_batches,
            "processing_time": elapsed,
        }

    def display_system_stats(self, config) -> None:
        """Display system statistics"""
        self.stdout.write(f"\n📈 STATISTICS: {config.name}")
        self.stdout.write("-" * 50)
        self.stdout.write(f"  • Total mappings: {config.total_mappings}")
        self.stdout.write(f"  • Validated: {config.validated_mappings}")
        self.stdout.write(f"  • High confidence: {config.high_confidence_mappings}")
        self.stdout.write(f"  • Avg confidence: {config.average_confidence:.3f}")
        self.stdout.write(f"  • Avg similarity: {config.average_similarity:.3f}")

    def display_final_summary(self, results: Dict[str, Any], total_time: float) -> None:
        """Display comprehensive final summary"""

        self.stdout.write("\n" + "=" * 80)
        self.stdout.write(self.style.SUCCESS("🎉 TERMINOLOGY MAPPING COMPLETED!"))
        self.stdout.write("=" * 80)

        # Statistics
        self.stdout.write("📊 FINAL STATISTICS")
        self.stdout.write(f"  • Configurations: {results['configs_created']}")
        self.stdout.write(f"  • Total mappings: {results['total_mappings']}")
        self.stdout.write(
            f"  • Successful systems: {len(results['successful_systems'])}"
        )
        self.stdout.write(f"  • Failed systems: {len(results['failed_systems'])}")
        self.stdout.write(
            f"  • Total time: {total_time / 60:.1f}min ({total_time / 3600:.1f}h)"
        )

        if results["total_mappings"] > 0:
            rate = results["total_mappings"] / total_time
            self.stdout.write(
                f"  • Rate: {rate:.2f} mappings/sec ({rate * 60:.0f}/min)"
            )

        # System breakdown
        if results["successful_systems"]:
            self.stdout.write("\n✅ SUCCESSFUL SYSTEMS:")
            for system in results["successful_systems"]:
                stats = results["system_stats"].get(system, {})
                time_taken = results["processing_times"].get(system, 0)
                self.stdout.write(
                    f"  • {system.upper()}: {stats.get('mappings_created', 0)} "
                    f"mappings in {time_taken:.1f}s"
                )

        if results["failed_systems"]:
            self.stdout.write(
                f"\n❌ FAILED SYSTEMS: {', '.join(results['failed_systems'])}"
            )

        self.stdout.write("\n🏁 Mapping process completed!")
