"""Unit tests for semantic_cache.py using threading for concurrency tests.

Run this file directly with Python 3: python test_semantic_cache.py
"""
import threading
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from semantic_cache import SemanticCache, CacheEntry


def test_put_get_basic():
    cache = SemanticCache()
    entry = CacheEntry(track_id=1, label="person", bbox=[0, 0, 10, 20], confidence=0.9, timestamp=100)
    cache.put(entry)
    # get with same frame should be a hit
    got = cache.get(1, 100)
    assert got is not None and got.track_id == 1


def test_staleness():
    cache = SemanticCache()
    entry = CacheEntry(track_id=2, label="car", bbox=[1, 2, 3, 4], confidence=0.8, timestamp=0)
    cache.put(entry)
    # TTL default 15, so at frame 20 it should be stale
    assert cache.get(2, 20) is None


def test_thread_safety_and_hit_rate():
    cache = SemanticCache()

    # Seed cache with some entries
    for i in range(5):
        cache.put(CacheEntry(track_id=i, label="obj", bbox=[0, 0, 1, 1], confidence=0.5, timestamp=0))

    stop = threading.Event()

    def reader_thread():
        # continue reading until writers are done
        while not stop.is_set():
            for i in range(10):
                cache.get(i % 5, 0)  # majority will hit
            time.sleep(0.001)

    def writer_thread():
        # update timestamps so entries stay fresh
        for _ in range(50):
            for i in range(5):
                cache.put(CacheEntry(track_id=i, label="obj", bbox=[0, 0, 1, 1], confidence=0.6, timestamp=0))
            time.sleep(0.002)

    readers = [threading.Thread(target=reader_thread) for _ in range(10)]
    writers = [threading.Thread(target=writer_thread) for _ in range(2)]

    for t in readers + writers:
        t.start()
    for w in writers:
        w.join()

    # signal readers to stop and join
    stop.set()
    for r in readers:
        r.join()

    stats = cache.get_stats()
    assert stats["hits"] + stats["misses"] > 0
    # hit rate should be between 0 and 1
    assert 0.0 <= stats["hit_rate"] <= 1.0


if __name__ == "__main__":
    test_put_get_basic()
    test_staleness()
    test_thread_safety_and_hit_rate()
    print("All tests passed!")
