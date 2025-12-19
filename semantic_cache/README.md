# Semantic Cache

Thread-safe cache for vision-language model (VLM) outputs in real-time video processing. Reduces expensive VLM inference calls by 93% while maintaining semantic richness through temporal reuse.

## Overview

The semantic cache enables real-time streaming applications by amortizing VLM reasoning across frames:
- **Fast path** (every frame): O(1) cache lookup for semantic labels
- **Slow path** (every 15 frames): VLM inference to refresh cache
- **Result**: 15-20× speedup vs. per-frame VLM inference

## Core API

### CacheEntry
```python
@dataclass
class CacheEntry:
    track_id: int       # ByteTrack ID
    label: str          # Semantic label (e.g., "person", "car")
    bbox: List[int]     # Bounding box [x, y, w, h]
    confidence: float   # Confidence score [0, 1]
    timestamp: int      # Frame number when created
```

### SemanticCache
```python
cache = SemanticCache(max_size=1000)

# Query cache (fast path)
entry = cache.get(track_id=42, current_frame=100)

# Batch query (recommended)
results = cache.get_batch([1, 2, 3], current_frame=100)

# Store VLM result (slow path)
cache.put(entry)

# Get statistics
stats = cache.get_stats()  # hits, misses, hit_rate, cache_size, evictions
```

## Features

- **Thread-safe**: `threading.RLock` for concurrent access
- **TTL-based expiration**: Entries stale after 15 frames (configurable)
- **LRU eviction**: Oldest entries removed when capacity reached
- **O(1) lookup**: Hash map keyed by `track_id`
- **Batch operations**: Efficient multi-track queries

## Performance

Based on simulations with 200 frames:

| Metric | Value |
|--------|-------|
| **Mean speedup** | 18.45× |
| **p50 speedup** | 61× |
| **p50 latency** | 10ms |
| **Hit rate** | 99.3% (ideal), 50% (realistic with tracking noise) |
| **VLM reduction** | 93.3% fewer calls |

## Design Decisions

**Why TTL-based expiration?**
Video frames are temporally correlated - an object labeled "person" at frame 100 is likely still a person at frame 105. TTL balances freshness with hit rate.

**Why track_id as key?**
ByteTrack maintains stable IDs across frames, making `track_id` the natural key for associating semantic labels with tracked objects.

**Why evict oldest by timestamp?**
Oldest entries are most likely stale. This simple policy performs well without LRU bookkeeping overhead.

**Why batch operations?**
Fast path queries 5-20 tracks per frame. Batching reduces lock contention and provides cleaner API than repeated `get()` calls.

## Integration

### With Slow Path Service
```python
# Enable local cache
export USE_LOCAL_SEMANTIC_CACHE=1

# Access shared cache instance
from slow_path.service import get_local_cache
cache = get_local_cache()
```

### Fast Path Example
```python
# Per-frame loop
tracks = tracker.update(frame)
results = cache.get_batch([t.track_id for t in tracks], frame_num)

for track in tracks:
    entry = results[track.track_id]
    if entry:
        render_overlay(entry.label, entry.bbox)
    else:
        schedule_vlm(track)  # Cache miss
```

### Slow Path Example
```python
# VLM inference
vlm_output = model.infer(frame, prompt)

# Store in cache
entry = CacheEntry.from_vlm_output(
    track_id=tid,
    vlm_dict=vlm_output,
    bbox=bbox,
    frame_num=frame_num
)
cache.put(entry)
```

## Folder Structure

```
semantic_cache/
├── README.md                  # This file
├── semantic_cache.py          # Core implementation
├── examples/
│   ├── integration_example.py       # Standalone demo (99.3% hit rate)
│   ├── cache_simulation.py          # Realistic simulation with noise
│   ├── visualize_cache_performance.py  # Generate plots
│   └── generate_report_metrics.py   # Comprehensive report generation
├── tests/
│   └── test_semantic_cache.py       # Unit tests
└── results/
    ├── report_figure_latency.png    # Latency comparison graphs
    ├── report_figure_scalability.png  # Scalability analysis
    ├── report_figure_cache_behavior.png  # Cache behavior over time
    ├── report_metrics.json          # Raw data
    └── report_tables.tex            # LaTeX tables
```

## Quick Start

```bash
# Run unit tests
python3 semantic_cache/tests/test_semantic_cache.py

# Run integration demo (ideal case)
python3 semantic_cache/examples/integration_example.py

# Generate all report figures and tables
python3 semantic_cache/examples/generate_report_metrics.py

# Quick performance test
python3 semantic_cache/examples/visualize_cache_performance.py --frames 100 --mode measure
```

## Related Work

This implementation is inspired by:
- SlowFast Networks (Feichtenhofer et al., 2019) - Multi-rate processing
- Token Merging (Bolya et al., 2023) - Spatial redundancy reduction
- CacheGen (Liu et al., 2023) - KV cache reuse for LLMs
