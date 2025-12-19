"""
Vision Embedding Cache

Caches vision encoder outputs and reuses them for similar ROIs.
Uses cosine similarity on vision embeddings for cache lookup.
"""

import torch
import numpy as np
from typing import Optional, Tuple, Dict
from dataclasses import dataclass
from collections import OrderedDict
import time


@dataclass
class CacheEntry:
    """A cached vision embedding."""
    track_id: Optional[int]  # Track ID if available
    embedding: torch.Tensor  # Vision encoder output
    timestamp: float  # When cached
    access_count: int  # How many times used
    image_hash: Optional[int]  # Perceptual hash of image for quick comparison


class VisionEmbeddingCache:
    """LRU cache for vision embeddings with similarity-based lookup."""

    def __init__(self, max_size: int = 200, similarity_threshold: float = 0.95):
        """
        Initialize vision embedding cache.

        Args:
            max_size: Maximum number of entries to cache
            similarity_threshold: Cosine similarity threshold for cache hit (0.9-0.99)
        """
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold

        # Cache storage: OrderedDict for LRU
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()

        # Track by track_id for fast exact lookup
        self.track_index: Dict[int, str] = {}

        # Statistics
        self.hits = 0
        self.misses = 0
        self.total_lookups = 0
        self.evictions = 0

    def _compute_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        """Compute cosine similarity between two embeddings."""
        # Flatten if needed
        if emb1.dim() > 2:
            emb1 = emb1.flatten(start_dim=1)
        if emb2.dim() > 2:
            emb2 = emb2.flatten(start_dim=1)

        # Mean pooling if sequence
        if emb1.dim() == 2 and emb1.size(0) > 1:
            emb1 = emb1.mean(dim=0, keepdim=True)
        if emb2.dim() == 2 and emb2.size(0) > 1:
            emb2 = emb2.mean(dim=0, keepdim=True)

        # Cosine similarity
        similarity = torch.nn.functional.cosine_similarity(emb1, emb2, dim=-1)
        return similarity.item()

    def get(self, query_embedding: torch.Tensor, track_id: Optional[int] = None) -> Optional[torch.Tensor]:
        """
        Look up cached embedding similar to query.

        Args:
            query_embedding: Vision embedding to match against (for similarity check)
            track_id: Optional track ID for exact match

        Returns:
            Cached embedding if found, None otherwise
        """
        self.total_lookups += 1

        # Fast path: exact track_id match
        if track_id is not None and track_id in self.track_index:
            cache_key = self.track_index[track_id]
            if cache_key in self.cache:
                entry = self.cache[cache_key]
                # Move to end (LRU)
                self.cache.move_to_end(cache_key)
                entry.access_count += 1
                self.hits += 1
                return entry.embedding

        # Slow path: similarity search
        best_match = None
        best_similarity = -1.0

        for key, entry in self.cache.items():
            sim = self._compute_similarity(query_embedding, entry.embedding)
            if sim > best_similarity:
                best_similarity = sim
                best_match = entry

        if best_match is not None and best_similarity >= self.similarity_threshold:
            # Cache hit!
            self.hits += 1
            best_match.access_count += 1
            # Move to end (LRU) - find the key
            for key, entry in self.cache.items():
                if entry is best_match:
                    self.cache.move_to_end(key)
                    break
            return best_match.embedding

        # Cache miss
        self.misses += 1
        return None

    def put(self, embedding: torch.Tensor, track_id: Optional[int] = None, image_hash: Optional[int] = None):
        """
        Store an embedding in the cache.

        Args:
            embedding: Vision embedding to cache
            track_id: Optional track ID for indexing
            image_hash: Optional perceptual hash for quick comparison
        """
        # Generate cache key
        if track_id is not None:
            cache_key = f"track_{track_id}"
        else:
            cache_key = f"emb_{len(self.cache)}_{time.time()}"

        # Evict if at capacity
        if len(self.cache) >= self.max_size:
            # Remove least recently used (first item)
            evicted_key, evicted_entry = self.cache.popitem(last=False)
            self.evictions += 1

            # Remove from track index
            if evicted_entry.track_id is not None:
                self.track_index.pop(evicted_entry.track_id, None)

        # Create and store entry
        entry = CacheEntry(
            track_id=track_id,
            embedding=embedding.detach().clone() if isinstance(embedding, torch.Tensor) else embedding,
            timestamp=time.time(),
            access_count=0,
            image_hash=image_hash
        )

        self.cache[cache_key] = entry

        # Update track index
        if track_id is not None:
            self.track_index[track_id] = cache_key

    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        self.track_index.clear()
        self.hits = 0
        self.misses = 0
        self.total_lookups = 0
        self.evictions = 0

    def get_stats(self) -> Dict[str, float]:
        """Get cache statistics."""
        hit_rate = self.hits / self.total_lookups if self.total_lookups > 0 else 0.0

        return {
            "hits": self.hits,
            "misses": self.misses,
            "total_lookups": self.total_lookups,
            "hit_rate": hit_rate,
            "cache_size": len(self.cache),
            "evictions": self.evictions,
            "max_size": self.max_size
        }

    def __len__(self):
        return len(self.cache)

    def __repr__(self):
        stats = self.get_stats()
        return f"VisionEmbeddingCache(size={stats['cache_size']}/{self.max_size}, hit_rate={stats['hit_rate']:.2%})"
