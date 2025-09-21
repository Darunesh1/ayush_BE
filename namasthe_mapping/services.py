# namasthe_mapping/services.py
# Fixed with a BERT model that supports feature extraction

import logging
import os
import time
from typing import Dict, List

import numpy as np
import requests
from django.conf import settings

logger = logging.getLogger(__name__)


class HuggingFaceAPIService:
    """HF API service using proper BERT model for feature extraction"""

    def __init__(self):
        # Use BioBERT that definitely supports feature extraction
        self.model_name = "dmis-lab/biobert-v1.1"
        self.api_url = f"https://api-inference.huggingface.co/models/{self.model_name}"

        # Get token
        api_key = (
            os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGINGFACE_API_TOKEN")
            or getattr(settings, "HUGGINGFACE_API_TOKEN", None)
        )

        if not api_key:
            raise ValueError("No HF token found")

        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        self.timeout = 30
        self.rate_limit_delay = 2.0  # Slightly slower for stability

        logger.info(f"✅ Service initialized with BioBERT: {self.model_name}")

        # Test connection
        self._test_connection()

    def _test_connection(self):
        """Test API with BioBERT model"""
        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json={"inputs": "test connection", "options": {"wait_for_model": True}},
                timeout=15,
            )

            logger.info(f"Test response status: {response.status_code}")

            if response.status_code == 200:
                logger.info("✅ BioBERT API connection successful")
            elif response.status_code == 503:
                logger.info("⏳ BioBERT model loading (normal)")
            else:
                logger.warning(f"Test response: {response.text}")

        except Exception as e:
            logger.warning(f"Connection test: {str(e)}")

    def normalize_medical_text(self, text: str) -> str:
        """Normalize medical text for BioBERT"""
        if not text:
            return ""

        # Keep more medical terminology intact for BioBERT
        normalized = text.strip()

        # Only basic normalization for BioBERT
        replacements = {
            " w/": " with",
            " w/o": " without",
            "&": " and ",
            "  ": " ",  # Double spaces
        }

        for old, new in replacements.items():
            normalized = normalized.replace(old, new)

        return normalized

    def generate_embedding(self, text: str, retries: int = 3) -> List[float]:
        """Generate embedding using BioBERT"""
        if not text or text.strip() == "":
            return [0.0] * 768  # BioBERT uses 768 dimensions

        normalized_text = self.normalize_medical_text(text)

        for attempt in range(retries):
            try:
                payload = {
                    "inputs": normalized_text,
                    "options": {"wait_for_model": True, "use_cache": False},
                }

                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout,
                )

                logger.info(f"🔍 API call {attempt + 1}: Status {response.status_code}")

                if response.status_code == 200:
                    result = response.json()
                    embedding = self._extract_biobert_embedding(result)

                    if embedding and len(embedding) == 768:
                        # Check if we got real values
                        non_zero_count = sum(1 for x in embedding if abs(x) > 0.001)
                        if (
                            non_zero_count > 50
                        ):  # BioBERT should have many non-zero values
                            logger.info(
                                f"✅ Got BioBERT embedding with {non_zero_count} non-zero values"
                            )
                            return embedding
                        else:
                            logger.warning(
                                f"BioBERT embedding mostly zeros: {non_zero_count}"
                            )

                elif response.status_code == 503:
                    wait_time = 15 + (attempt * 10)
                    logger.info(f"⏳ BioBERT loading, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                else:
                    logger.error(
                        f"❌ API error {response.status_code}: {response.text}"
                    )

                if attempt == retries - 1:
                    logger.error("❌ All retries failed")
                    return [0.0] * 768

                time.sleep(3)

            except Exception as e:
                logger.error(f"❌ Request error (attempt {attempt + 1}): {str(e)}")
                if attempt == retries - 1:
                    return [0.0] * 768
                time.sleep(3)

        return [0.0] * 768

    def _extract_biobert_embedding(self, result) -> List[float]:
        """Extract embedding from BioBERT API response"""
        try:
            logger.info(f"🔍 BioBERT result type: {type(result)}")

            # BioBERT typically returns nested structure
            if isinstance(result, list) and len(result) > 0:
                # First try direct access
                if isinstance(result[0], list) and len(result[0]) == 768:
                    logger.info("✅ Found 768-dim embedding in result[0]")
                    return [float(x) for x in result[0]]

                # Try nested access
                if hasattr(result[0], "__iter__"):
                    for item in result[0]:
                        if hasattr(item, "__len__") and len(item) == 768:
                            logger.info(
                                "✅ Found 768-dim embedding in nested structure"
                            )
                            return [float(x) for x in item]

            # Handle numpy arrays
            if hasattr(result, "tolist"):
                as_list = result.tolist()
                if isinstance(as_list, list):
                    if len(as_list) == 768:
                        logger.info("✅ Found 768-dim embedding from numpy conversion")
                        return [float(x) for x in as_list]
                    elif (
                        len(as_list) > 0
                        and isinstance(as_list[0], list)
                        and len(as_list[0]) == 768
                    ):
                        logger.info("✅ Found nested 768-dim embedding from numpy")
                        return [float(x) for x in as_list[0]]

            # Handle dict response (common with BioBERT)
            if isinstance(result, dict):
                logger.info(f"Dict keys: {list(result.keys())}")
                for key in [
                    "embeddings",
                    "features",
                    "hidden_states",
                    "last_hidden_state",
                ]:
                    if key in result:
                        embedding = result[key]
                        if hasattr(embedding, "__len__") and len(embedding) == 768:
                            logger.info(f"✅ Found 768-dim embedding in {key}")
                            return [float(x) for x in embedding]

            logger.error("❌ Could not extract 768-dim BioBERT embedding")
            if isinstance(result, list):
                logger.error(f"   List structure: length={len(result)}")
                if len(result) > 0:
                    logger.error(f"   First item type: {type(result[0])}")
                    if hasattr(result[0], "__len__"):
                        logger.error(f"   First item length: {len(result[0])}")

            return None

        except Exception as e:
            logger.error(f"❌ BioBERT extraction error: {str(e)}")
            return None

    def generate_batch_embeddings(
        self, texts: List[str], batch_size: int = 3
    ) -> List[List[float]]:
        """Generate BioBERT embeddings for multiple texts (slower but more reliable)"""
        if not texts:
            return []

        all_embeddings = []
        total_texts = len(texts)
        successful_count = 0

        logger.info(f"🚀 Generating BioBERT embeddings for {total_texts} texts")

        for i, text in enumerate(texts):
            embedding = self.generate_embedding(text)
            all_embeddings.append(embedding)

            # Count successful embeddings
            non_zero_count = sum(1 for x in embedding if abs(x) > 0.001)
            if non_zero_count > 50:
                successful_count += 1

            if (i + 1) % 3 == 0 or i == total_texts - 1:
                logger.info(
                    f"📈 Progress: {i + 1}/{total_texts} (successful: {successful_count})"
                )

            # Rate limiting for BioBERT
            if i < total_texts - 1:
                time.sleep(self.rate_limit_delay)

        logger.info(
            f"✅ BioBERT completed: {successful_count}/{total_texts} successful embeddings"
        )
        return all_embeddings

    def calculate_similarity(
        self, embedding1: List[float], embedding2: List[float]
    ) -> float:
        """Calculate cosine similarity"""
        try:
            if not embedding1 or not embedding2:
                return 0.0

            if len(embedding1) != len(embedding2):
                logger.error(
                    f"Dimension mismatch: {len(embedding1)} vs {len(embedding2)}"
                )
                return 0.0

            emb1 = np.array(embedding1, dtype=np.float32)
            emb2 = np.array(embedding2, dtype=np.float32)

            dot_product = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            similarity = dot_product / (norm1 * norm2)
            return float(np.clip(similarity, -1.0, 1.0))

        except Exception as e:
            logger.error(f"Similarity error: {str(e)}")
            return 0.0

    def find_best_matches(
        self,
        source_embedding: List[float],
        target_embeddings: List[List[float]],
        similarity_threshold: float = 0.75,
        top_k: int = 5,
    ) -> List[Dict]:
        """Find best matches using BioBERT embeddings"""
        matches = []

        if not source_embedding or not target_embeddings:
            return matches

        try:
            source_emb = np.array(source_embedding, dtype=np.float32)
            target_embs = np.array(target_embeddings, dtype=np.float32)

            dot_products = np.dot(target_embs, source_emb)
            source_norm = np.linalg.norm(source_emb)
            target_norms = np.linalg.norm(target_embs, axis=1)

            valid_mask = (source_norm > 0) & (target_norms > 0)
            similarities = np.zeros(len(target_embs))

            if source_norm > 0:
                valid_indices = np.where(valid_mask)[0]
                similarities[valid_indices] = dot_products[valid_indices] / (
                    source_norm * target_norms[valid_indices]
                )

            above_threshold = similarities >= similarity_threshold

            if not np.any(above_threshold):
                return matches

            valid_indices = np.where(above_threshold)[0]
            valid_similarities = similarities[above_threshold]
            sorted_indices = np.argsort(valid_similarities)[::-1]

            for idx in sorted_indices[:top_k]:
                original_idx = valid_indices[idx]
                similarity = valid_similarities[idx]

                matches.append(
                    {
                        "index": int(original_idx),
                        "similarity": float(similarity),
                        "confidence": min(float(similarity) + 0.05, 1.0),
                    }
                )

            return matches

        except Exception as e:
            logger.error(f"Error finding matches: {str(e)}")
            return matches

    def get_model_info(self) -> Dict:
        """Get BioBERT model info"""
        return {
            "model_name": self.model_name,
            "framework": "BioBERT via HF API",
            "embedding_dimension": 768,
            "medical_optimized": True,
            "model_type": "BioBERT",
        }


# Keep alias
HuggingFaceInferenceService = HuggingFaceAPIService
