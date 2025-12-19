"""
Vision Embedding Cache Module

Caches vision encoder outputs to avoid recomputing for similar ROIs in video.
Exploits temporal similarity in video tracking.
"""

from .embedding_cache import VisionEmbeddingCache

__all__ = ['VisionEmbeddingCache']
