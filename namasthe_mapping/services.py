# namasthe_mapping/services.py
# Updated to use InferenceClient instead of requests

import logging
import os
from typing import Dict, List

import numpy as np
from django.conf import settings
from huggingface_hub import InferenceClient

logger = logging.getLogger(__name__)


class HuggingFaceInferenceService:
    """Ultra-lightweight BioBERT using InferenceClient"""

    def __init__(self):
        self.model_name = "nlpie/tiny-biobert"

        # Get API token from environment or settings
        api_key = (
            os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGINGFACE_API_TOKEN")
            or getattr(settings, "HUGGINGFACE_API_TOKEN", None)
        )

        if not api_key:
            raise ValueError(
                "No Hugging Face API token found. Set HF_TOKEN or HUGGINGFACE_API_TOKEN "
                "in environment or Django settings."
            )

        # Initialize InferenceClient
        self.client = InferenceClient(model=self.model_name, token=api_key)

        logger.info(f"✅ InferenceClient initialized for {self.model_name}")

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

        return " ".join(normalized.split())

    def generate_embedding(self, text: str, retries: int = 3) -> List[float]:
        """Generate embedding for single text using InferenceClient"""
        if not text or text.strip() == "":
            return [0.0] * 768

        normalized_text = self.normalize_medical_text(text)

        for attempt in range(retries):
            try:
                # Use feature_extraction for embeddings
                embedding = self.client.feature_extraction(normalized_text)

                # Handle different response formats
                if isinstance(embedding, list):
                    if len(embedding) == 768:
                        return embedding
                    elif len(embedding) == 1 and len(embedding[0]) == 768:
                        return embedding[0]  # Sometimes returns nested list

                logger.error(f"Unexpected embedding format: {type(embedding)}")
                return [0.0] * 768

            except Exception as e:
                logger.error(f"API error (attempt {attempt + 1}): {str(e)}")
                if attempt == retries - 1:  # Last attempt
                    return [0.0] * 768

                # Wait before retry
                import time

                time.sleep((attempt + 1) * 2)

        return [0.0] * 768

    def generate_batch_embeddings(
        self, texts: List[str], batch_size: int = 5
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts with rate limiting"""
        if not texts:
            return []

        all_embeddings = []

        # Process texts individually for stability
        # (batch feature_extraction might not be supported)
        for i, text in enumerate(texts):
            logger.info(f"⚙️  Processing text {i + 1}/{len(texts)}")

            embedding = self.generate_embedding(text)
            all_embeddings.append(embedding)

            # Rate limiting - be gentle with API
            if (i + 1) % batch_size == 0 and i + 1 < len(texts):
                import time

                time.sleep(1)  # Wait 1 second every batch_size requests

        logger.info(f"✅ Generated {len(all_embeddings)} embeddings")
        return all_embeddings

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

    def get_model_info(self) -> Dict:
        """Get service information"""
        return {
            "model_name": self.model_name,
            "framework": "Hugging Face InferenceClient",
            "embedding_dimension": 768,
            "client_type": "InferenceClient",
        }
