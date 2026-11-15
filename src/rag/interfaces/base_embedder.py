from abc import ABC, abstractmethod
import numpy as np

class BaseEmbedder(ABC):
    @abstractmethod
    def embed(self, image_path: str) -> np.ndarray | None:
        pass

    @abstractmethod
    def embed_batch(self, image_paths: list[str]) -> list[tuple[str, np.ndarray]]:
        pass
