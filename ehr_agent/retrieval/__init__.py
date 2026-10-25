"""Search over a patient's record text: BM25, dense embeddings, hybrid fusion, document context."""

from .hybrid import HybridIndex, document_context, document_of

__all__ = ["HybridIndex", "document_context", "document_of"]
