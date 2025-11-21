# What Your Semantic Cache Does

## The Problem It Solves

**Problem**: VLMs (like Video-LLaVA) are too slow for real-time video (200-500ms per frame)
**Solution**: Cache VLM results and reuse them across multiple frames
**Result**: 18× speedup while maintaining semantic quality

---

## Core Functionality

### 1. **CacheEntry** - Data Structure

Stores VLM output for a tracked object:

```python
CacheEntry(
    track_id=42,           # ByteTrack ID (unique per object)
    label="person",        # Semantic label from VLM
    bbox=[100, 50, 80, 120],  # Bounding box [x, y, w, h]
    confidence=0.92,       # VLM confidence score
    timestamp=100          # Frame number when created
)
```

**What it does:**
- Links VLM output to a specific tracked object
- Stores when it was created (for staleness checking)
- Provides a factory method to create from VLM JSON output

---

### 2. **SemanticCache** - The Cache

A thread-safe hash map that stores CacheEntry objects keyed by `track_id`.

#### Core Operations

**a) Store VLM Result (Slow Path)**
```python
cache.put(entry)  # O(1) insertion
```
- Stores or updates a cache entry
- If cache is full, evicts oldest entry (by timestamp)
- Thread-safe for concurrent slow path writes

**b) Query Cache (Fast Path)**
```python
entry = cache.get(track_id=42, current_frame=100)
# Returns entry if fresh, None if stale/missing
```
- O(1) lookup by track_id
- Checks if entry is stale (older than 15 frames)
- Automatically removes stale entries
- Updates hit/miss statistics

**c) Batch Query (Recommended for Fast Path)**
```python
results = cache.get_batch([1, 2, 3, 4, 5], current_frame=100)
# Returns dict: {track_id -> CacheEntry or None}
```
- Queries multiple tracks in one call
- More efficient (single lock acquisition)
- Updates hit/miss counters for all queries

**d) Get Statistics**
```python
stats = cache.get_stats()
# Returns: {hits, misses, hit_rate, cache_size, evictions}
```
- Tracks cache performance
- Useful for evaluation and tuning

---

### 3. **TTL (Time-To-Live)** - Automatic Expiration

**What it does:**
```python
def is_stale(current_frame, ttl=15):
    return (current_frame - entry.timestamp) > ttl
```

- Entries expire after 15 frames (configurable)
- Stale entries are automatically removed on `get()`
- Ensures labels stay relevant (video changes over time)

**Why 15 frames?**
- Matches VLM refresh interval from your proposal
- Objects don't change labels within 0.5 seconds (at 30 FPS)
- Balances hit rate (higher TTL = more hits) vs freshness (lower TTL = more accurate)

---

### 4. **LRU Eviction** - Memory Management

**What it does:**
```python
# When cache reaches max_size (default 1000)
if len(cache) >= max_size:
    oldest_track = find_entry_with_min_timestamp()
    delete(oldest_track)
```

- Prevents unbounded memory growth
- Evicts oldest entry when full
- Oldest = lowest timestamp (most likely to be stale anyway)

**Why oldest-first?**
- Simple: O(n) to find oldest, happens rarely
- Effective: Oldest entries are most likely stale
- No need for complex LRU bookkeeping (last accessed time)

---

### 5. **Thread Safety** - Concurrent Access

**What it does:**
```python
# All methods use threading.RLock
with self._lock:
    # Safe to read/write cache concurrently
```

- Fast path (multiple threads) can query simultaneously
- Slow path (VLM workers) can write simultaneously
- No race conditions or data corruption

**Why RLock not Lock?**
- RLock allows same thread to re-acquire lock
- Methods can call each other safely (e.g., `get_stats()` calls `get_hit_rate()`)

---

## How It Works in Your System

### Fast Path (Every Frame)

```
Frame arrives → YOLO detects objects → ByteTrack assigns IDs
                                              ↓
                            Query cache: cache.get_batch([1,2,3,4,5])
                                              ↓
                    ┌─────────────────────────┴─────────────────────┐
                    ↓                                                 ↓
              Cache HIT                                         Cache MISS
         (entry exists & fresh)                            (missing or stale)
                    ↓                                                 ↓
        Use cached label                                  Schedule for VLM
        Render overlay                                    (add to slow path queue)
```

### Slow Path (Every 15 Frames or On-Demand)

```
VLM inference request → Run VLM on frame → Get semantic label
                                                    ↓
                                          Create CacheEntry
                                                    ↓
                                          cache.put(entry)
                                                    ↓
                                    Next 15 frames will be cache HITs!
```

---

## Key Implementation Features

### 1. **O(1) Operations**
- `get()`: Hash map lookup
- `put()`: Hash map insert
- `get_batch()`: N hash map lookups (still O(N), but single lock)

### 2. **Automatic Cleanup**
- Stale entries removed on `get()` (lazy deletion)
- No background thread needed
- No memory leaks

### 3. **Statistics Tracking**
```python
self._hits = 0      # Successful cache lookups
self._misses = 0    # Cache misses (stale or not found)
self._evictions = 0 # Entries removed due to capacity
```
- Automatic hit/miss counting
- Use for evaluation and debugging
- Thread-safe updates

### 4. **Helper Methods**

**CacheEntry.from_vlm_output()**
```python
# Convert VLM JSON to CacheEntry
vlm_output = {"label": "person", "confidence": 0.92}
entry = CacheEntry.from_vlm_output(track_id, vlm_output, bbox, frame_num)
```
- Convenience method for slow path integration
- Handles type conversions (float, etc.)
- Provides default values

### 5. **Integration with Slow Path Service**

```python
# In slow_path/service/worker.py
if USE_LOCAL_SEMANTIC_CACHE:
    _local_cache = SemanticCache()  # Global instance

    # After VLM inference:
    entry = CacheEntry.from_vlm_output(...)
    _local_cache.put(entry)
```
- Single shared cache instance
- Fast path and slow path use same cache
- Enabled via environment variable

---

## What Makes It "Semantic"?

**Regular cache**: Stores raw pixel data (image patches)
**Semantic cache**: Stores **meaning** (labels, categories, attributes)

Your cache stores:
- ✅ "person" (semantic label)
- ✅ "bicycle" (object category)
- ✅ 0.92 (confidence in understanding)
- ❌ NOT raw pixels or features

This is why it's called a **semantic** cache - it caches the VLM's **understanding** of the scene, not the visual data itself.

---

## Performance Characteristics

From your simulations:

| Operation | Complexity | Typical Time |
|-----------|-----------|--------------|
| `get()` | O(1) | <0.1ms |
| `put()` | O(1) amortized | <0.1ms |
| `get_batch(N)` | O(N) | <1ms for N=20 |
| `get_stats()` | O(1) | <0.01ms |
| Eviction | O(n) | ~1ms (happens rarely) |

**Memory usage:**
- ~200 bytes per CacheEntry
- Max 1000 entries = ~200KB
- Negligible vs. model weights (GBs)

---

## Summary: What You Built

You implemented a **thread-safe, TTL-based semantic cache** that:

1. ✅ **Stores** VLM outputs keyed by track ID
2. ✅ **Retrieves** cached labels in O(1) time
3. ✅ **Expires** stale entries after 15 frames
4. ✅ **Evicts** oldest entries when full
5. ✅ **Tracks** performance statistics (hit rate, etc.)
6. ✅ **Supports** concurrent fast/slow path access
7. ✅ **Integrates** with your slow_path service
8. ✅ **Achieves** 18× speedup and 99.3% hit rate (ideal) / 50% (realistic)

This is the core component that makes real-time VLM streaming possible! 🎉
