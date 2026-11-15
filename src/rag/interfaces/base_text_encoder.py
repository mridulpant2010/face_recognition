from abc import ABC, abstractmethod
import numpy as np


class BaseTextEncoder(ABC):
    @abstractmethod
    def encode(self, text: str) -> np.ndarray:
        pass

    @abstractmethod
    def encode_batch(self, texts: list[str]) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        pass
