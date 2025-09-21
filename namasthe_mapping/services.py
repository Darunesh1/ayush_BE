# namasthe_mapping/services.py
# Ultra-lightweight BioBERT via Hugging Face API

import logging
import time
from typing import Dict, List

import numpy as np
import requests
from django.conf import settings

logger = logging.getLogger(__name__)


class HuggingFaceAPIService:
    """Ultra-lightweight TinyBioBERT service via Hugging Face API"""

    def __init__(self):
        self.model_name = "nlpie/tiny-biobert"
        self.api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{self.model_name}"

        # Get API token from settings
        token = getattr(settings, "HUGGINGFACE_API_TOKEN", None)
        if not token:
            raise ValueError(
                "HUGGINGFACE_API_TOKEN not found in settings. "
                "Please add your HF API token to config/settings.py"
            )

        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        # Configuration from settings
        self.timeout = getattr(settings, "BIOBERT_API_TIMEOUT", 30)
        self.rate_limit = getattr(settings, "BIOBERT_RATE_LIMIT", 1)

        # Test API connection on initialization
        self._test_api_connection()

    def _test_api_connection(self):
        """Test API connection on service initialization"""
        try:
            test_response = requests.post(
                self.api_url,
                headers=self.headers,
                json={"inputs": "test connection"},
                timeout=10,
            )

            if test_response.status_code == 200:
                logger.info("✅ Hugging Face API connection successful")
            elif test_response.status_code == 503:
                logger.warning("⚠️  Model is loading, will retry during generation")
            else:
                logger.warning(f"⚠️  API test returned: {test_response.status_code}")

        except Exception as e:
            logger.error(f"❌ API connection test failed: {str(e)}")
            raise

    def normalize_medical_text(self, text: str) -> str:
        """Normalize medical text for better embeddings"""
        if not text:
            return ""

        normalized = text.lower().strip()

        # Medical abbreviations normalization
        replacements = {
            " nos": " not otherwise specified",
            " unspec": " unspecified",
            " w/": " with",
            " w/o": " without",
            "&": " and ",
            "+": " plus ",
            "pts": "patients",
            "dx": "diagnosis",
            "tx": "treatment",
            "sx": "symptoms",
        }

        for old, new in replacements.items():
            normalized = normalized.replace(old, new)

        # Clean whitespace
        return " ".join(normalized.split())

    def generate_embedding(self, text: str, retries: int = 3) -> List[float]:
        """Generate embedding for single text via API"""
        if not text or text.strip() == "":
            return [0.0] * 768

        normalized_text = self.normalize_medical_text(text)

        for attempt in range(retries):
            try:
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json={"inputs": normalized_text},
                    timeout=self.timeout,
                )

                if response.status_code == 200:
                    embedding = response.json()

                    # Validate embedding format
                    if isinstance(embedding, list) and len(embedding) == 768:
                        return embedding
                    else:
                        logger.error(
                            f"Invalid embedding format: {type(embedding)}, length: {len(embedding) if isinstance(embedding, list) else 'N/A'}"
                        )
                        return [0.0] * 768

                elif response.status_code == 503:
                    # Model is loading, wait and retry
                    wait_time = (attempt + 1) * 5
                    logger.warning(
                        f"Model loading, waiting {wait_time}s before retry..."
                    )
                    time.sleep(wait_time)
                    continue

                else:
                    logger.error(f"API error {response.status_code}: {response.text}")
                    if attempt == retries - 1:  # Last attempt
                        return [0.0] * 768
                    time.sleep(2)  # Brief wait before retry

            except requests.RequestException as e:
                logger.error(f"Request error (attempt {attempt + 1}): {str(e)}")
                if attempt == retries - 1:  # Last attempt
                    return [0.0] * 768
                time.sleep(2)

        return [0.0] * 768

    def generate_batch_embeddings(
        self, texts: List[str], batch_size: int = 5
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts with batching and rate limiting"""
        if not texts:
            return []

        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size

        logger.info(f"🚀 Processing {len(texts)} texts in {total_batches} batches")

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_num = i // batch_size + 1

            logger.info(
                f"⚙️  Processing batch {batch_num}/{total_batches} ({len(batch_texts)} texts)"
            )

            # Try batch request first
            batch_embeddings = self._generate_batch_via_api(batch_texts)

            if batch_embeddings:
                all_embeddings.extend(batch_embeddings)
                logger.info(f"✅ Batch {batch_num} completed successfully")
            else:
                # Fallback: individual requests
                logger.warning(
                    f"⚠️  Batch {batch_num} failed, falling back to individual requests"
                )
                for text in batch_texts:
                    embedding = self.generate_embedding(text)
                    all_embeddings.append(embedding)
                    time.sleep(0.2)  # Rate limit individual requests

            # Rate limiting between batches
            if i + batch_size < len(texts):  # Don't wait after last batch
                time.sleep(self.rate_limit)

        logger.info(f"🎯 Completed: {len(all_embeddings)} embeddings generated")
        return all_embeddings

    def _generate_batch_via_api(
        self, texts: List[str], retries: int = 2
    ) -> List[List[float]]:
        """Generate embeddings for a batch via single API call"""
        normalized_texts = [self.normalize_medical_text(text) for text in texts]

        for attempt in range(retries):
            try:
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json={"inputs": normalized_texts},
                    timeout=self.timeout * 2,  # Longer timeout for batches
                )

                if response.status_code == 200:
                    batch_embeddings = response.json()

                    # Validate batch format
                    if (
                        isinstance(batch_embeddings, list)
                        and len(batch_embeddings) == len(texts)
                        and all(
                            isinstance(emb, list) and len(emb) == 768
                            for emb in batch_embeddings
                        )
                    ):
                        return batch_embeddings
                    else:
                        logger.error(f"Invalid batch embedding format")
                        return None

                elif response.status_code == 503:
                    wait_time = (attempt + 1) * 10
                    logger.warning(f"Model loading, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                else:
                    logger.error(
                        f"Batch API error {response.status_code}: {response.text}"
                    )
                    return None

            except requests.RequestException as e:
                logger.error(f"Batch request error: {str(e)}")
                if attempt < retries - 1:
                    time.sleep(5)

        return None

    def calculate_similarity(
        self, embedding1: List[float], embedding2: List[float]
    ) -> float:
        """Calculate cosine similarity between embeddings"""
        try:
            if not embedding1 or not embedding2:
                return 0.0

            # Convert to numpy arrays
            emb1 = np.array(embedding1)
            emb2 = np.array(embedding2)

            # Cosine similarity
            dot_product = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            similarity = dot_product / (norm1 * norm2)
            return float(np.clip(similarity, -1.0, 1.0))

        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}")
            return 0.0

    def find_best_matches(
        self,
        source_embedding: List[float],
        target_embeddings: List[List[float]],
        similarity_threshold: float = 0.75,
        top_k: int = 5,
    ) -> List[Dict]:
        """Find best matching targets using vectorized operations"""
        matches = []

        if not source_embedding or not target_embeddings:
            return matches

        try:
            # Convert to numpy arrays for vectorized operations
            source_emb = np.array(source_embedding)
            target_embs = np.array(target_embeddings)

            # Vectorized cosine similarity
            dot_products = np.dot(target_embs, source_emb)
            source_norm = np.linalg.norm(source_emb)
            target_norms = np.linalg.norm(target_embs, axis=1)

            # Avoid division by zero
            valid_norms = (source_norm > 0) & (target_norms > 0)

            similarities = np.zeros(len(target_embs))
            similarities[valid_norms] = dot_products[valid_norms] / (
                source_norm * target_norms[valid_norms]
            )

            # Find matches above threshold
            above_threshold = similarities >= similarity_threshold

            if not np.any(above_threshold):
                return matches

            # Get top matches
            valid_indices = np.where(above_threshold)[0]
            valid_similarities = similarities[above_threshold]

            # Sort by similarity descending
            sorted_indices = np.argsort(valid_similarities)[::-1]

            for idx in sorted_indices[:top_k]:
                original_idx = valid_indices[idx]
                similarity = valid_similarities[idx]

                matches.append(
                    {
                        "index": int(original_idx),
                        "similarity": float(similarity),
                        "confidence": min(
                            float(similarity) + 0.05, 1.0
                        ),  # Small confidence boost
                    }
                )

            return matches

        except Exception as e:
            logger.error(f"Error finding matches: {str(e)}")
            return matches

    def get_model_info(self) -> Dict:
        """Get service information"""
        return {
            "model_name": self.model_name,
            "framework": "Hugging Face Inference API",
            "embedding_dimension": 768,
            "total_dependencies_size": "~75MB",
            "api_url": self.api_url,
            "timeout": self.timeout,
            "rate_limit": self.rate_limit,
        }
