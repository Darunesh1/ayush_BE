import time

import requests
from django.core.management.base import BaseCommand
from django.db import transaction

from terminologies.models import ICD11Term


class Command(BaseCommand):
    help = "Populate ICD11Term records from local API using existing foundation URIs"

    def add_arguments(self, parser):
        parser.add_argument(
            "--batch-size",
            type=int,
            default=100,
            help="Number of terms to process in each batch (default: 100)",
        )
        parser.add_argument(
            "--delay",
            type=float,
            default=0.1,
            help="Delay between API calls in seconds (default: 0.1)",
        )
        parser.add_argument(
            "--base-url",
            type=str,
            default="http://localhost:8080",
            help="Base URL for the ICD API (default: http://localhost:8080)",
        )

    def handle(self, *args, **options):
        batch_size = options["batch_size"]
        delay = options["delay"]
        base_url = options["base_url"].rstrip("/")

        self.stdout.write(
            self.style.SUCCESS("Starting ICD11Term population from local API...")
        )

        # Filter out invalid URIs and get valid ones
        terms_to_update = (
            ICD11Term.objects.exclude(foundation_uri__isnull=True)
            .exclude(foundation_uri__exact="")
            .exclude(foundation_uri__startswith="missing-foundation-uri")
            .filter(foundation_uri__startswith="http")
            .values_list("foundation_uri", "id")
        )

        total_terms = len(terms_to_update)
        self.stdout.write(f"Found {total_terms} valid terms to update")

        if total_terms == 0:
            self.stdout.write(self.style.WARNING("No valid terms found. Exiting."))
            return

        updated_count = 0
        error_count = 0

        # Process in batches
        for i in range(0, total_terms, batch_size):
            batch = terms_to_update[i : i + batch_size]
            self.stdout.write(
                f"Processing batch {i // batch_size + 1} ({len(batch)} terms)..."
            )

            for foundation_uri, term_id in batch:
                try:
                    # Convert to local API URL
                    local_api_url = self.convert_to_local_api_url(
                        foundation_uri, base_url
                    )

                    if not local_api_url:
                        self.stdout.write(
                            self.style.WARNING(
                                f"Could not convert URI: {foundation_uri}"
                            )
                        )
                        error_count += 1
                        continue

                    result = self.populate_single_term(local_api_url, foundation_uri)

                    if result:
                        updated_count += 1
                    else:
                        error_count += 1

                    # Add delay to avoid overwhelming the API
                    if delay > 0:
                        time.sleep(delay)

                except Exception as e:
                    self.stdout.write(
                        self.style.ERROR(f"Unexpected error for {foundation_uri}: {e}")
                    )
                    error_count += 1

            # Progress update
            processed = min(i + batch_size, total_terms)
            self.stdout.write(
                f"Progress: {processed}/{total_terms} "
                f"(Updated: {updated_count}, Errors: {error_count})"
            )

        # Final summary
        self.stdout.write(
            self.style.SUCCESS(
                f"\nPopulation complete!\n"
                f"Total processed: {total_terms}\n"
                f"Successfully updated: {updated_count}\n"
                f"Errors: {error_count}"
            )
        )

    def convert_to_local_api_url(self, foundation_uri, base_url):
        """Convert foundation URI to local MMS API URL"""
        try:
            # Handle entity URIs: http://id.who.int/icd/entity/1435254666
            if "/icd/entity/" in foundation_uri:
                entity_id = foundation_uri.split("/icd/entity/")[-1]
                # Convert entity ID to MMS API URL
                return f"{base_url}/icd/release/11/2025-01/mms/{entity_id}"

            # Handle MMS URIs: http://id.who.int/icd/release/11/2025-01/mms/250688797
            elif "/mms/" in foundation_uri:
                entity_id = foundation_uri.split("/mms/")[-1]
                return f"{base_url}/icd/release/11/2025-01/mms/{entity_id}"

            # If no recognizable pattern, return None
            return None

        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f"Error converting URL {foundation_uri}: {e}")
            )
            return None

    def populate_single_term(self, api_url, original_foundation_uri):
        """Populate a single ICD11Term from API data"""
        try:
            # Set up headers as specified
            headers = {
                "Accept-Language": "en",
                "API-Version": "v2",
                "Accept": "application/json",
            }

            self.stdout.write(f"Calling: {api_url}")  # Debug output

            # Make API request to LOCAL server
            response = requests.get(api_url, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()

            # Extract and process the data
            processed_data = self.process_api_response(data)

            # Update or create the term
            with transaction.atomic():
                term, created = ICD11Term.objects.update_or_create(
                    foundation_uri=original_foundation_uri, defaults=processed_data
                )

            action = "created" if created else "updated"
            self.stdout.write(f"✓ {action}: {term.code} - {term.title[:50]}...")
            return True

        except requests.exceptions.RequestException as e:
            self.stdout.write(self.style.WARNING(f"API error for {api_url}: {e}"))
            return False
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Processing error for {api_url}: {e}"))
            return False

    def process_api_response(self, data):
        """Process API response data into model fields"""
        # Extract index terms
        index_terms = []
        for term in data.get("indexTerm", []):
            if "label" in term and "@value" in term["label"]:
                index_terms.append(term["label"]["@value"])

        # Extract inclusions
        inclusions = []
        for inc in data.get("inclusion", []):
            label = inc.get("label", {}).get("@value", "")
            if label:
                inclusions.append(
                    {
                        "label": label,
                        "foundation_reference": inc.get("foundationReference", ""),
                    }
                )

        # Extract exclusions
        exclusions = []
        for exc in data.get("exclusion", []):
            label = exc.get("label", {}).get("@value", "")
            if label:
                exclusions.append(
                    {
                        "label": label,
                        "foundation_reference": exc.get("foundationReference", ""),
                        "linearization_reference": exc.get(
                            "linearizationReference", ""
                        ),
                    }
                )

        # Return processed data
        return {
            "code": data.get("code", ""),
            "title": data.get("title", {}).get("@value", ""),
            "definition": data.get("definition", {}).get("@value", ""),
            "long_definition": data.get("longDefinition", {}).get("@value", ""),
            "index_terms": index_terms,
            "parent": data.get("parent", []),
            "inclusions": inclusions,
            "exclusions": exclusions,
            "postcoordination_scales": data.get("postcoordinationScale", []),
            "related_perinatal_entities": data.get(
                "relatedEntitiesInPerinatalChapter", []
            ),
            "browser_url": data.get("browserUrl", ""),
            "source": data.get("source", ""),
            "class_kind": data.get("classKind", ""),
        }
