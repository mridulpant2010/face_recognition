class FaceRAGError(Exception):
    pass

class EmbeddingModelLoadError(FaceRAGError):
    pass

class ImageReadError(FaceRAGError):
    pass

class NoFaceDetectedError(FaceRAGError):
    pass

class VectorDimensionMismatchError(FaceRAGError):
    pass

class VectorCountMismatchError(FaceRAGError):
    pass

class IndexNotFoundError(FaceRAGError):
    pass

class EmptyIndexError(FaceRAGError):
    pass

class DatasetNotFoundError(FaceRAGError):
    pass

class InsufficientDataError(FaceRAGError):
    pass

class ProbeSetNotFoundError(FaceRAGError):
    pass
