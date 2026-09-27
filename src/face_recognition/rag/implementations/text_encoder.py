import logging
import numpy as np
from sentence_transformers import SentenceTransformer

from ..interfaces.base_text_encoder import BaseTextEncoder
from ..core.exceptions import EmbeddingModelLoadError

logger = logging.getLogger(__name__)


class SentenceTransformerEncoder(BaseTextEncoder):
    def __init__(self, model_name: str = "clip-ViT-B-32"):
        logger.info(f"Loading text encoder '{model_name}'...")
        try:
            self._model = SentenceTransformer(model_name)
            self._dim = self._model.get_sentence_embedding_dimension()
            logger.info(f"Text encoder loaded (output dim: {self._dim}).")
        except Exception as e:
            raise EmbeddingModelLoadError(f"Failed to load text encoder '{model_name}': {e}")

    def encode(self, text: str) -> np.ndarray:
        embedding = self._model.encode(text, normalize_embeddings=True)
        return embedding.astype(np.float32)

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        embeddings = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
        return embeddings.astype(np.float32)

    @property
    def embedding_dim(self) -> int:
        return self._dim
