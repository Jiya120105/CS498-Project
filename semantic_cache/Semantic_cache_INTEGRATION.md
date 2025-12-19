# Semantic Cache Integration Guide

Purpose
- The semantic cache stores VLM outputs per tracker ID so the fast path can reuse labels for up to 15 frames (TTL = 15) and avoid running the slow VLM every frame.

Key files
- `semantic_cache.py` — provides `CacheEntry` and `SemanticCache`.

Core API
- `CacheEntry.from_vlm_output(track_id, vlm_dict, bbox, frame_num)` — helper that builds a `CacheEntry` from VLM output.
- `SemanticCache.get(track_id, current_frame)` -> `Optional[CacheEntry]`
- `SemanticCache.get_batch(track_ids, current_frame)` -> `Dict[int, Optional[CacheEntry]]`
- `SemanticCache.put(entry)`
- `SemanticCache.get_stats()` -> `dict` with keys: `hits`, `misses`, `hit_rate`, `cache_size`
- `SemanticCache.get_hit_rate()` -> `float`
- `SemanticCache.clear()`

Behavior guarantees
- TTL: entries are stale when `(current_frame - entry.timestamp) > 15` frames by default.
- Thread-safety: all public methods use `threading.RLock` — safe for concurrent readers and writers.
- Capacity: `SemanticCache(max_size=1000)` by default. When inserting a new distinct `track_id` would exceed `max_size`, the single oldest entry (by `timestamp`) is evicted.
- No LRU or advanced eviction beyond the single-oldest rule.

For Person 1 — Fast Path (YOLO + ByteTrack)
- Use `get_batch` each frame with the tracker `track_ids`.
- Example (per-frame):

```python
track_ids = [t.id for t in tracks]  # from ByteTrack
results = cache.get_batch(track_ids, current_frame=frame_no)
for tid in track_ids:
    entry = results.get(tid)
    if entry is not None:
        label = entry.label
        confidence = entry.confidence
        # render overlay using label and entry.bbox
    else:
        # cache miss: render fallback and optionally mark for slow-path VLM
        pass
```

- `get_batch` reduces lock churn and updates hit/miss counters automatically. `None` means missing or stale entry.

For Person 2 — Slow Path (VLM)
- Convert VLM outputs to `CacheEntry` and call `cache.put()`.
- Example:

```python
# vlm_result example: {'label': 'person', 'confidence': 0.92}
entry = CacheEntry.from_vlm_output(track_id=tid, vlm_dict=vlm_result, bbox=bbox, frame_num=frame_no)
cache.put(entry)
```

- Use the same `frame_no` used by the fast path for consistency.

For Person 4 — Evaluation & Metrics
- Query `cache.get_stats()` periodically:

```python
stats = cache.get_stats()
# stats: {"hits": int, "misses": int, "hit_rate": float, "cache_size": int}
hit_rate = cache.get_hit_rate()
```

- Suggested monitoring: sample `get_stats()` every N frames (e.g. every 100 frames) and export to your metrics system.

Design notes and tips
- Fast path should treat `None` as a miss and schedule that track for the slow path when appropriate (e.g., every 15 frames or when necessary).
- Slow path should write VLM results immediately after inference to keep entries fresh.
- If you need unlimited cache size, construct with `SemanticCache(max_size=None)`.

Integration checklist
- Person 1 (Fast Path): integrate `get_batch` call into per-frame loop, collect misses for VLM scheduling.
- Person 2 (Slow Path): use `CacheEntry.from_vlm_output(...)` and `put()` after VLM inference.
- Person 4 (Evaluation): add periodic sampling of `cache.get_stats()` to dashboards.
- Person 3 (Semantic Cache owner): optionally provide a small `cache_adapter.py` with thin wrappers matching your team's data shapes.

