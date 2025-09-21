# namasthe_mapping/services.py
# Enhanced with robust text handling and tensor size error prevention

import logging
import os
import re
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
from django.conf import settings

logger = logging.getLogger(__name__)


class HuggingFaceAPIService:
    """Enhanced HF API service with robust text preprocessing to prevent tensor errors"""

    def __init__(self, model_preference="biobert"):
        # Enhanced model configuration with fallback options
        self.model_configs = {
            "biobert": {
                "name": "dmis-lab/biobert-v1.1",
                "max_length": 512,
                "embedding_dim": 768,
                "tokens_per_word": 1.4,  # Medical text tokenization ratio
            },
            "tinybert": {
                "name": "huawei-noah/TinyBERT_General_4L_312D",
                "max_length": 512,
                "embedding_dim": 312,
                "tokens_per_word": 1.2,
            },
            "minibert": {
                "name": "sentence-transformers/all-MiniLM-L6-v2",
                "max_length": 256,
                "embedding_dim": 384,
                "tokens_per_word": 1.1,
            },
        }

        config = self.model_configs.get(model_preference, self.model_configs["biobert"])
        self.model_name = config["name"]
        self.max_length = config["max_length"]
        self.embedding_dim = config["embedding_dim"]
        self.tokens_per_word = config["tokens_per_word"]

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
        self.rate_limit_delay = 2.5  # Conservative rate limiting

        logger.info(
            f"✅ Service initialized with {self.model_name} (dim: {self.embedding_dim})"
        )

        # Test connection
        self._test_connection()

    def _test_connection(self):
        """Test API connection with simple input"""
        try:
            test_text = "test connection"
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json={
                    "inputs": test_text,
                    "options": {
                        "wait_for_model": True,
                        "use_cache": False,
                        "truncation": True,
                        "max_length": self.max_length,
                    },
                },
                timeout=15,
            )

            logger.info(f"Test response status: {response.status_code}")

            if response.status_code == 200:
                logger.info(f"✅ {self.model_name} API connection successful")
            elif response.status_code == 503:
                logger.info(f"⏳ {self.model_name} model loading (normal)")
            else:
                logger.warning(f"Test response: {response.text}")

        except Exception as e:
            logger.warning(f"Connection test failed: {str(e)}")

    def validate_and_clean_text(self, text: str) -> Tuple[str, str]:
        """Comprehensive text validation and cleaning"""
        if not text or not isinstance(text, str):
            return "", "empty_or_invalid"

        original_length = len(text)

        # Strip and basic validation
        text = text.strip()
        if len(text) < 2:
            return "", "too_short"

        # Remove problematic characters that cause tokenization issues
        problematic_patterns = [
            r"[^\x00-\x7F]+",  # Non-ASCII characters that might cause issues
            r"[†‡§¶©®™]+",  # Special symbols
            r"[\x00-\x1F\x7F-\x9F]+",  # Control characters
            r"_{3,}",  # Multiple underscores
            r"-{4,}",  # Multiple dashes
            r"\.{4,}",  # Multiple dots
        ]

        for pattern in problematic_patterns:
            text = re.sub(pattern, " ", text)

        # Normalize excessive whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Medical text specific cleaning
        text = self.normalize_medical_text(text)

        if len(text) != original_length:
            logger.debug(f"Text cleaned: {original_length} → {len(text)} chars")

        return text, "valid"

    def truncate_text_precise(self, text: str) -> str:
        """Precise text truncation with multiple safety checks"""
        if not text:
            return ""

        # Conservative token estimation based on model type
        safety_factor = 0.85  # Use only 85% of max capacity
        max_safe_tokens = int(self.max_length * safety_factor)  # ~435 for BioBERT

        # Estimate tokens more accurately for medical text
        estimated_tokens = len(text.split()) * self.tokens_per_word

        # If already safe, return as-is
        if estimated_tokens <= max_safe_tokens and len(text) <= 3000:
            return text

        # Method 1: Word-based truncation
        words = text.split()
        max_words = int(
            max_safe_tokens / self.tokens_per_word
        )  # ~310 words for BioBERT

        if len(words) > max_words:
            # Try to truncate at sentence boundaries first
            sentences = text.split(". ")
            truncated_sentences = []
            word_count = 0

            for sentence in sentences:
                sentence_words = len(sentence.split())
                if word_count + sentence_words <= max_words:
                    truncated_sentences.append(sentence)
                    word_count += sentence_words
                else:
                    break

            if truncated_sentences:
                text = ". ".join(truncated_sentences)
                if not text.endswith("."):
                    text += "."
            else:
                # Fallback to word truncation
                text = " ".join(words[:max_words])

        # Method 2: Character-based safety check
        max_chars = max_safe_tokens * 4  # ~1740 chars for BioBERT
        if len(text) > max_chars:
            # Truncate at word boundary
            text = text[:max_chars]
            last_space = text.rfind(" ")
            if last_space > max_chars * 0.8:  # Only if we don't lose too much
                text = text[:last_space]

        # Method 3: Final safety validation
        final_estimated_tokens = len(text.split()) * self.tokens_per_word
        if final_estimated_tokens > max_safe_tokens:
            # Emergency word truncation
            emergency_max_words = int(max_safe_tokens / (self.tokens_per_word * 1.2))
            words = text.split()[:emergency_max_words]
            text = " ".join(words)
            logger.warning(f"Emergency truncation applied: {len(words)} words")

        logger.info(
            f"📏 Text truncated: {estimated_tokens:.0f} → {len(text.split()) * self.tokens_per_word:.0f} estimated tokens"
        )
        return text

    def normalize_medical_text(self, text: str) -> str:
        """Enhanced medical text normalization"""
        if not text:
            return ""

        # Medical abbreviation expansions
        medical_replacements = {
            # Common abbreviations that might cause tokenization issues
            " w/": " with",
            " w/o": " without",
            " vs ": " versus ",
            " vs.": " versus",
            "&": " and ",
            "@": " at ",
            "#": " number ",
            "%": " percent",
            # Medical specific
            " dx ": " diagnosis ",
            " tx ": " treatment ",
            " sx ": " symptoms ",
            " hx ": " history ",
            " pt ": " patient ",
            # Formatting
            "  ": " ",  # Double spaces
            "\t": " ",  # Tabs
            "\n": " ",  # Newlines
            "\r": " ",  # Carriage returns
        }

        normalized = text
        for old, new in medical_replacements.items():
            normalized = normalized.replace(old, new)

        # Clean up any remaining multiple spaces
        normalized = re.sub(r"\s+", " ", normalized).strip()

        return normalized

    def generate_embedding(self, text: str, retries: int = 3) -> List[float]:
        """Enhanced embedding generation with comprehensive error prevention"""

        # Step 1: Validate and clean text
        cleaned_text, status = self.validate_and_clean_text(text)

        if status != "valid":
            logger.warning(f"Invalid text: {status}")
            return [0.0] * self.embedding_dim

        # Step 2: Precise truncation
        truncated_text = self.truncate_text_precise(cleaned_text)

        # Step 3: Final safety checks
        if len(truncated_text) > 4000:  # Hard character limit
            truncated_text = truncated_text[:4000].rsplit(" ", 1)[0]
            logger.warning("Applied emergency character truncation")

        final_word_count = len(truncated_text.split())
        estimated_tokens = final_word_count * self.tokens_per_word

        logger.debug(
            f"Final text stats: {len(truncated_text)} chars, {final_word_count} words, ~{estimated_tokens:.0f} tokens"
        )

        # Step 4: Generate embedding with retries
        for attempt in range(retries):
            try:
                payload = {
                    "inputs": truncated_text,
                    "options": {
                        "wait_for_model": True,
                        "use_cache": False,
                        "truncation": True,  # Model-side truncation as backup
                        "max_length": self.max_length,
                        "padding": False,  # Avoid unnecessary padding
                        "return_tensors": False,  # Get raw arrays
                    },
                }

                logger.debug(
                    f"🔍 API call {attempt + 1} for text: {truncated_text[:100]}..."
                )

                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout,
                )

                logger.debug(f"Response status: {response.status_code}")

                if response.status_code == 200:
                    result = response.json()
                    embedding = self._extract_embedding(result)

                    if embedding and len(embedding) == self.embedding_dim:
                        # Validate embedding quality
                        non_zero_count = sum(1 for x in embedding if abs(x) > 0.001)

                        if (
                            non_zero_count > self.embedding_dim * 0.1
                        ):  # At least 10% non-zero
                            logger.info(
                                f"✅ Generated embedding: {non_zero_count}/{self.embedding_dim} non-zero values"
                            )
                            return embedding
                        else:
                            logger.warning(
                                f"Poor quality embedding: only {non_zero_count} non-zero values"
                            )

                elif response.status_code == 503:
                    wait_time = 15 + (attempt * 10)
                    logger.info(f"⏳ Model loading, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                elif response.status_code == 400:
                    error_text = response.text
                    logger.error(f"❌ API error 400: {error_text}")

                    # Check for tensor size errors specifically
                    if "tensor" in error_text.lower() and "size" in error_text.lower():
                        logger.error(
                            "🚨 Tensor size mismatch detected - text still too long"
                        )
                        # Further emergency truncation
                        if len(truncated_text) > 1000:
                            truncated_text = truncated_text[:1000].rsplit(" ", 1)[0]
                            logger.warning("Applied emergency re-truncation")
                            continue  # Retry with shorter text

                    # For other 400 errors, don't retry
                    break

                else:
                    logger.error(
                        f"❌ API error {response.status_code}: {response.text}"
                    )

                # Wait before retry
                if attempt < retries - 1:
                    time.sleep(3 + attempt)

            except requests.exceptions.Timeout:
                logger.error(f"❌ Timeout on attempt {attempt + 1}")
                if attempt < retries - 1:
                    time.sleep(5)
            except Exception as e:
                logger.error(f"❌ Request error (attempt {attempt + 1}): {str(e)}")
                if attempt < retries - 1:
                    time.sleep(3)

        logger.error("❌ All embedding attempts failed")
        return [0.0] * self.embedding_dim

    def _extract_embedding(self, result) -> Optional[List[float]]:
        """Enhanced embedding extraction with better error handling"""
        try:
            logger.debug(f"🔍 Extracting from result type: {type(result)}")

            # Handle different response formats
            if isinstance(result, list) and len(result) > 0:
                # Direct list of embeddings
                if isinstance(result[0], list) and len(result[0]) == self.embedding_dim:
                    logger.debug(
                        f"✅ Found {self.embedding_dim}-dim embedding in result[0]"
                    )
                    return [float(x) for x in result[0]]

                # Nested structure
                if isinstance(result[0], (list, tuple)):
                    for item in result[0]:
                        if hasattr(item, "__len__") and len(item) == self.embedding_dim:
                            logger.debug("✅ Found embedding in nested structure")
                            return [float(x) for x in item]

            # Handle numpy arrays
            if hasattr(result, "tolist"):
                as_list = result.tolist()
                if isinstance(as_list, list):
                    if len(as_list) == self.embedding_dim:
                        logger.debug("✅ Found embedding from numpy conversion")
                        return [float(x) for x in as_list]
                    elif (
                        len(as_list) > 0
                        and isinstance(as_list[0], list)
                        and len(as_list[0]) == self.embedding_dim
                    ):
                        logger.debug("✅ Found nested embedding from numpy")
                        return [float(x) for x in as_list[0]]

            # Handle dictionary responses
            if isinstance(result, dict):
                logger.debug(f"Dict keys: {list(result.keys())}")

                # Common keys for embeddings
                for key in [
                    "embeddings",
                    "features",
                    "hidden_states",
                    "last_hidden_state",
                    "pooler_output",
                    "sentence_embeddings",
                ]:
                    if key in result:
                        embedding_data = result[key]

                        # Handle nested structures
                        if isinstance(embedding_data, list):
                            if len(embedding_data) > 0:
                                candidate = (
                                    embedding_data[0]
                                    if isinstance(embedding_data[0], list)
                                    else embedding_data
                                )
                                if len(candidate) == self.embedding_dim:
                                    logger.debug(
                                        f"✅ Found {self.embedding_dim}-dim embedding in {key}"
                                    )
                                    return [float(x) for x in candidate]

            # Log structure for debugging
            logger.error(f"❌ Could not extract {self.embedding_dim}-dim embedding")
            if isinstance(result, list) and len(result) > 0:
                logger.error(f"   List length: {len(result)}")
                logger.error(f"   First item type: {type(result[0])}")
                if hasattr(result[0], "__len__"):
                    logger.error(f"   First item length: {len(result[0])}")
            elif isinstance(result, dict):
                logger.error(f"   Available keys: {list(result.keys())}")

            return None

        except Exception as e:
            logger.error(f"❌ Embedding extraction error: {str(e)}")
            return None

    def generate_batch_embeddings(
        self,
        texts: List[str],
        batch_size: int = 2,  # Reduced batch size
    ) -> List[List[float]]:
        """Generate embeddings with individual processing to avoid batch errors"""

        if not texts:
            return []

        all_embeddings = []
        total_texts = len(texts)
        successful_count = 0
        failed_count = 0

        logger.info(
            f"🚀 Generating {self.model_name} embeddings for {total_texts} texts"
        )

        for i, text in enumerate(texts):
            try:
                embedding = self.generate_embedding(text)
                all_embeddings.append(embedding)

                # Count successful embeddings
                non_zero_count = sum(1 for x in embedding if abs(x) > 0.001)
                if non_zero_count > self.embedding_dim * 0.1:
                    successful_count += 1
                    logger.debug(
                        f"   ✅ Text {i + 1}: SUCCESS ({non_zero_count} non-zero values)"
                    )
                else:
                    failed_count += 1
                    logger.debug(
                        f"   ❌ Text {i + 1}: Poor quality ({non_zero_count} non-zero values)"
                    )

            except Exception as e:
                logger.error(f"   ❌ Text {i + 1}: ERROR - {str(e)}")
                failed_count += 1
                all_embeddings.append([0.0] * self.embedding_dim)

            # Progress reporting
            if (i + 1) % 5 == 0 or i == total_texts - 1:
                logger.info(
                    f"📈 Progress: {i + 1}/{total_texts} (✅{successful_count} ❌{failed_count})"
                )

            # Rate limiting
            if i < total_texts - 1:
                time.sleep(self.rate_limit_delay)

        success_rate = (successful_count / total_texts) * 100 if total_texts > 0 else 0
        logger.info(
            f"✅ Batch completed: {successful_count}/{total_texts} successful ({success_rate:.1f}%)"
        )

        return all_embeddings

    def calculate_similarity(
        self, embedding1: List[float], embedding2: List[float]
    ) -> float:
        """Enhanced cosine similarity calculation with error handling"""
        try:
            if not embedding1 or not embedding2:
                return 0.0

            if len(embedding1) != len(embedding2):
                logger.error(
                    f"Dimension mismatch: {len(embedding1)} vs {len(embedding2)}"
                )
                return 0.0

            # Convert to numpy arrays with proper dtype
            emb1 = np.array(embedding1, dtype=np.float32)
            emb2 = np.array(embedding2, dtype=np.float32)

            # Check for invalid values
            if np.any(np.isnan(emb1)) or np.any(np.isnan(emb2)):
                logger.warning("NaN values detected in embeddings")
                return 0.0

            if np.any(np.isinf(emb1)) or np.any(np.isinf(emb2)):
                logger.warning("Infinite values detected in embeddings")
                return 0.0

            # Calculate cosine similarity
            dot_product = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            similarity = dot_product / (norm1 * norm2)

            # Ensure result is in valid range
            similarity = float(np.clip(similarity, -1.0, 1.0))

            return similarity

        except Exception as e:
            logger.error(f"Similarity calculation error: {str(e)}")
            return 0.0

    def find_best_matches(
        self,
        source_embedding: List[float],
        target_embeddings: List[List[float]],
        similarity_threshold: float = 0.75,
        top_k: int = 5,
    ) -> List[Dict]:
        """Enhanced matching with better error handling and performance"""
        matches = []

        if not source_embedding or not target_embeddings:
            return matches

        try:
            source_emb = np.array(source_embedding, dtype=np.float32)
            target_embs = np.array(target_embeddings, dtype=np.float32)

            # Validate dimensions
            if source_emb.shape[0] != self.embedding_dim:
                logger.error(
                    f"Source embedding dimension mismatch: {source_emb.shape[0]} != {self.embedding_dim}"
                )
                return matches

            if target_embs.shape[1] != self.embedding_dim:
                logger.error(
                    f"Target embeddings dimension mismatch: {target_embs.shape[1]} != {self.embedding_dim}"
                )
                return matches

            # Vectorized similarity calculation
            dot_products = np.dot(target_embs, source_emb)
            source_norm = np.linalg.norm(source_emb)
            target_norms = np.linalg.norm(target_embs, axis=1)

            # Avoid division by zero
            valid_mask = (source_norm > 0) & (target_norms > 0)
            similarities = np.zeros(len(target_embs))

            if source_norm > 0:
                valid_indices = np.where(valid_mask)[0]
                if len(valid_indices) > 0:
                    similarities[valid_indices] = dot_products[valid_indices] / (
                        source_norm * target_norms[valid_indices]
                    )

            # Filter by threshold
            above_threshold = similarities >= similarity_threshold

            if not np.any(above_threshold):
                logger.debug(f"No matches above threshold {similarity_threshold}")
                return matches

            # Get top matches
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

            logger.debug(f"Found {len(matches)} matches above threshold")
            return matches

        except Exception as e:
            logger.error(f"Error finding matches: {str(e)}")
            return matches

    def get_model_info(self) -> Dict:
        """Get comprehensive model information"""
        return {
            "model_name": self.model_name,
            "framework": f"{self.model_name} via HF API",
            "embedding_dimension": self.embedding_dim,
            "max_length": self.max_length,
            "tokens_per_word": self.tokens_per_word,
            "medical_optimized": "biobert" in self.model_name.lower(),
            "model_type": "BERT-based",
            "rate_limit_delay": self.rate_limit_delay,
        }

    def health_check(self) -> Dict:
        """Comprehensive health check"""
        try:
            # Test with a simple medical term
            test_embedding = self.generate_embedding("fever headache nausea")

            non_zero_count = sum(1 for x in test_embedding if abs(x) > 0.001)
            quality_score = non_zero_count / self.embedding_dim

            return {
                "status": "healthy" if quality_score > 0.1 else "degraded",
                "model": self.model_name,
                "embedding_dim": self.embedding_dim,
                "test_quality": quality_score,
                "non_zero_features": non_zero_count,
                "timestamp": time.time(),
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "model": self.model_name,
                "timestamp": time.time(),
            }


# Keep backward compatibility
HuggingFaceInferenceService = HuggingFaceAPIService
