import logging
import numpy as np

from ..interfaces.base_embedder import BaseEmbedder
from ..interfaces.base_text_encoder import BaseTextEncoder
from ..core.exceptions import ImageReadError

logger = logging.getLogger(__name__)


class FusionStrategy:
    CONCAT = "concat"
    ADD = "add"
    WEIGHTED_ADD = "weighted_add"


class MultimodalEmbedder:
    """
    Fuses an image embedding (from ArcFace) with a text embedding (from SentenceTransformer)
    into a single multimodal identity vector.

    Supports 3 fusion strategies:
        - concat:       [img_vec | txt_vec] → (img_dim + txt_dim)-D
        - add:          img_vec + txt_vec   → img_dim-D (requires same dimensions)
        - weighted_add: α·img_vec + (1-α)·txt_vec → img_dim-D
    """

    def __init__(
        self,
        image_embedder: BaseEmbedder,
        text_encoder: BaseTextEncoder,
        fusion: str = FusionStrategy.CONCAT,
        image_weight: float = 0.6,
    ):
        self._image_embedder = image_embedder
        self._text_encoder = text_encoder
        self._fusion = fusion
        self._image_weight = image_weight

        if fusion in (FusionStrategy.ADD, FusionStrategy.WEIGHTED_ADD):
            img_dim = 512  # ArcFace output
            txt_dim = text_encoder.embedding_dim
            if img_dim != txt_dim:
                raise ValueError(
                    f"Addition fusion requires same dimensions. "
                    f"Image: {img_dim}-D, Text: {txt_dim}-D. "
                    f"Use 'concat' fusion or a 512-D text model like clip-ViT-B-32."
                )

        logger.info(f"MultimodalEmbedder ready (fusion: {fusion})")

    @property
    def image_embedder(self) -> BaseEmbedder:
        return self._image_embedder

    @property
    def text_encoder(self) -> BaseTextEncoder:
        return self._text_encoder

    @property
    def output_dim(self) -> int:
        if self._fusion == FusionStrategy.CONCAT:
            return 512 + self._text_encoder.embedding_dim
        return 512

    def embed(self, image_path: str, text: str) -> np.ndarray | None:
        """
        Create a fused multimodal embedding from an image and its text description.
        Returns None if no face is detected in the image.
        """
        img_vec = self._image_embedder.embed(image_path)
        if img_vec is None:
            return None

        txt_vec = self._text_encoder.encode(text)

        return self._fuse(img_vec, txt_vec)

    def embed_image_only(self, image_path: str) -> np.ndarray | None:
        """Fallback: embed using only the image (zero-padded text for concat, or image-only for add)."""
        img_vec = self._image_embedder.embed(image_path)
        if img_vec is None:
            return None

        if self._fusion == FusionStrategy.CONCAT:
            zero_txt = np.zeros(self._text_encoder.embedding_dim, dtype=np.float32)
            return self._fuse(img_vec, zero_txt)
        else:
            # For addition-based fusion, just return the normalized image vector
            return img_vec / np.linalg.norm(img_vec)

    def embed_text_only(self, text: str) -> np.ndarray:
        """Fallback: embed using only the text (zero-padded image for concat, or text-only for add)."""
        txt_vec = self._text_encoder.encode(text)

        if self._fusion == FusionStrategy.CONCAT:
            zero_img = np.zeros(512, dtype=np.float32)
            return self._fuse(zero_img, txt_vec)
        else:
            return txt_vec / np.linalg.norm(txt_vec)

    def _fuse(self, img_vec: np.ndarray, txt_vec: np.ndarray) -> np.ndarray:
        # L2-normalize both vectors before fusion
        img_norm = img_vec / (np.linalg.norm(img_vec) + 1e-8)
        txt_norm = txt_vec / (np.linalg.norm(txt_vec) + 1e-8)

        if self._fusion == FusionStrategy.CONCAT:
            fused = np.concatenate([img_norm, txt_norm])

        elif self._fusion == FusionStrategy.ADD:
            fused = img_norm + txt_norm

        elif self._fusion == FusionStrategy.WEIGHTED_ADD:
            alpha = self._image_weight
            fused = alpha * img_norm + (1 - alpha) * txt_norm

        else:
            raise ValueError(f"Unknown fusion strategy: {self._fusion}")

        # Final L2-normalize so FAISS cosine similarity works correctly
        return (fused / (np.linalg.norm(fused) + 1e-8)).astype(np.float32)
