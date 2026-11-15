from abc import ABC, abstractmethod
import numpy as np

class BaseVectorStore(ABC):
    @abstractmethod
    def add(self, embeddings: np.ndarray, metadata_list: list[dict]) -> None:
        pass

    @abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: int) -> list[dict]:
        pass

    @abstractmethod
    def save(self, directory) -> None:
        pass

    @abstractmethod
    def load(self, directory) -> None:
        pass

    @property
    @abstractmethod
    def total_vectors(self) -> int:
        pass
