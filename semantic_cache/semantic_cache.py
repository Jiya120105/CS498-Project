"""Simple thread-safe semantic cache for video processing.

Provides CacheEntry dataclass and SemanticCache implementation.
Only uses standard library (threading, dataclasses, typing).
"""
from dataclasses import dataclass
from typing import List, Dict, Optional
import threading


@dataclass
class CacheEntry:
    """A cache entry representing a VLM semantic result for a tracked object.

    Attributes:
        track_id: unique identifier for the tracked object
        label: semantic label returned by the VLM
        bbox: bounding box as [x, y, w, h]
        confidence: confidence score between 0 and 1
        timestamp: frame number when the entry was created
    """

    track_id: int
    label: str
    bbox: List[int]
    confidence: float
    timestamp: int

    def is_stale(self, current_frame: int, ttl: int = 15) -> bool:
        """Return True if the entry is older than the given TTL (in frames).

        Args:
            current_frame: the current frame number
            ttl: time-to-live in frames (default 15)
        """
        return (current_frame - self.timestamp) > ttl

    @staticmethod
    def from_vlm_output(track_id: int, vlm_dict: Dict[str, object], bbox: List[int], frame_num: int) -> "CacheEntry":
        """Create a CacheEntry from a VLM output dictionary.

        Args:
            track_id: ID of the tracked object
            vlm_dict: dictionary produced by the VLM containing at least 'label' and 'confidence'
            bbox: bounding box as [x, y, w, h]
            frame_num: current frame number to use as timestamp

        Returns:
            CacheEntry constructed from the provided values.
        """
        label = vlm_dict.get("label", "")
        confidence = float(vlm_dict.get("confidence", 0.0))
        return CacheEntry(track_id=track_id, label=label, bbox=bbox, confidence=confidence, timestamp=frame_num)


class SemanticCache:
    """A simple thread-safe semantic cache keyed by track_id.

    This class stores CacheEntry objects in an internal dictionary and
    keeps simple hit/miss statistics.
    """

    def __init__(self, max_size: int = 1000) -> None:
        """Create an empty SemanticCache.

        Args:
            max_size: optional maximum number of entries to store. When the
                      cache would grow beyond this size a single oldest entry
                      is evicted. Set to None for unlimited size.
        """
        # use RLock so methods can call each other safely while holding the lock
        self._lock = threading.RLock()
        self._cache: Dict[int, CacheEntry] = {}
        self._hits: int = 0
        self._misses: int = 0
        self._max_size: Optional[int] = max_size
        # Track number of evictions for diagnostics
        self._evictions: int = 0

    def get(self, track_id: int, current_frame: int) -> Optional[CacheEntry]:
        """Return CacheEntry for track_id if present and not stale.

        Args:
            track_id: id of the tracked object to look up
            current_frame: current frame number used to check staleness

        Returns:
            The CacheEntry if found and fresh; otherwise None.

        Side effects:
            - increments the internal hit counter when a fresh entry is returned
            - increments the miss counter when the entry is missing or stale
            - removes stale entries from the cache
        """
        with self._lock:
            entry = self._cache.get(track_id)
            if entry is None:
                self._misses += 1
                return None
            if entry.is_stale(current_frame):
                # treat stale as miss and remove from cache
                self._misses += 1
                try:
                    del self._cache[track_id]
                except KeyError:
                    pass
                return None
            self._hits += 1
            return entry

    def get_batch(self, track_ids: List[int], current_frame: int) -> Dict[int, Optional[CacheEntry]]:
        """Return a mapping of track_id -> CacheEntry (or None) for multiple IDs.

        This performs the equivalent logic of get() for each id but does so
        while holding the lock once for efficiency, updating hit/miss counters
        and evicting stale entries as appropriate.

        Args:
            track_ids: iterable of track IDs to query
            current_frame: current frame number used to check staleness

        Returns:
            Dict mapping each requested track_id to its CacheEntry or None.
        """
        results: Dict[int, Optional[CacheEntry]] = {}
        with self._lock:
            for tid in track_ids:
                entry = self._cache.get(tid)
                if entry is None:
                    self._misses += 1
                    results[tid] = None
                    continue
                if entry.is_stale(current_frame):
                    self._misses += 1
                    try:
                        del self._cache[tid]
                    except KeyError:
                        pass
                    results[tid] = None
                    continue
                self._hits += 1
                results[tid] = entry
        return results

    def put(self, entry: CacheEntry) -> bool:
        """Store or update a CacheEntry in the cache.

        If the cache would grow beyond the configured max_size when adding a
        new entry, the oldest entry (by timestamp) is removed first.

        Args:
            entry: CacheEntry to store
        
        Returns:
            bool: True if the entry was stored (possibly with eviction).
                False if the cache failed to store the entry.
        """
        with self._lock:
            try:
                # Evict oldest if adding a new key would exceed capacity
                is_new = entry.track_id not in self._cache
                if is_new and self._max_size is not None and len(self._cache) >= self._max_size:
                    # remove the entry with the smallest timestamp
                    try:
                        oldest_tid = min(self._cache.items(), key=lambda kv: kv[1].timestamp)[0]
                        del self._cache[oldest_tid]
                        # record eviction
                        self._evictions += 1
                    except ValueError:
                        # empty cache, nothing to evict
                        pass
                self._cache[entry.track_id] = entry
                return True
            except Exception:
                # any unexpected failure
                return False


    def get_hit_rate(self) -> float:
        """Return the hit rate as hits / (hits + misses).

        Returns 0.0 if there have been no lookups.
        """
        with self._lock:
            total = self._hits + self._misses
            return (self._hits / total) if total > 0 else 0.0

    def clear(self) -> None:
        """Clear the cache and reset hit/miss counters."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            self._evictions = 0

    def get_stats(self) -> Dict[str, float]:
        """Return simple statistics about the cache.

        The returned dict contains: hits, misses, hit_rate, cache_size, evictions
        """
        with self._lock:
            # compute hit rate inline to avoid nested lock issues
            total = self._hits + self._misses
            hit_rate = (self._hits / total) if total > 0 else 0.0
            return {
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "cache_size": len(self._cache),
                "evictions": self._evictions,
            }
