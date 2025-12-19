# Semantic Cache

Thread-safe cache for vision-language model (VLM) outputs in real-time video processing. Reduces expensive VLM inference calls by 93% while maintaining semantic richness through temporal reuse.

## Overview

The semantic cache enables real-time streaming applications by amortizing VLM reasoning across frames:
- **Fast path** (every frame): O(1) cache lookup for semantic labels
- **Slow path** (every 15 frames): VLM inference to refresh cache
- **Result**: 17.6× speedup vs. per-frame VLM inference

## Quick Start

```bash
# Run unit tests
python3 -m semantic_cache.tests.test_semantic_cache

# Run integration demo (ideal case, 99.3% hit rate)
python3 -m semantic_cache.examples.integration_example

# Generate all report figures and metrics
python3 -m semantic_cache.examples.generate_report_metrics

# Quick performance visualization
python3 -m semantic_cache.examples.visualize_cache_performance --frames 100 --mode measure
```

## Core API

### CacheEntry
```python
from semantic_cache import CacheEntry

entry = CacheEntry(
    track_id=42,            # ByteTrack ID
    label="person",         # Semantic label
    bbox=[100, 50, 80, 120], # Bounding box [x, y, w, h]
    confidence=0.92,        # Confidence score [0, 1]
    timestamp=100           # Frame number when created
)
```

### SemanticCache
```python
from semantic_cache import SemanticCache

cache = SemanticCache(max_size=1000, ttl_frames=15)

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
- **Zero external dependencies**: Pure Python stdlib

## Performance

Based on simulations with 200 frames:

| Metric | Value |
|--------|-------|
| **Mean speedup** | 17.6× |
| **p50 speedup** | 61× |
| **p50 latency** | 10ms |
| **Hit rate** | 99.3% (ideal), 51.7% (realistic with tracking noise) |
| **VLM reduction** | 93.3% fewer calls |

## Project Structure

```
semantic_cache/
├── README.md                           # This file
├── __init__.py                         # Package initialization
├── semantic_cache.py                   # Core implementation (200 lines)
│
├── examples/                           # Runnable examples
│   ├── integration_example.py          # Demo with ideal tracking (99.3% hit rate)
│   ├── cache_simulation.py             # Realistic simulation with noise (51.7% hit rate)
│   ├── visualize_cache_performance.py  # Performance visualization
│   └── generate_report_metrics.py      # Generate all report figures
│
├── tests/                              # Unit tests
│   └── test_semantic_cache.py          # 8 unit tests (all pass)
│
├── results/                            # Generated outputs
│   ├── latency_comparison.png          # Per-frame latency graph
│   ├── scalability_analysis.png        # Speedup vs frame count
│   ├── cache_behavior.png              # Cache dynamics over time
│   ├── cache_performance.png           # 3-plot performance visualization
│   ├── metrics.json                    # Raw performance data
│   └── tables.tex                      # LaTeX tables for report
│
└── docs/                               # Documentation
    └── DESIGN_AND_IMPLEMENTATION.md    # Detailed design documentation
```

## Integration Examples

### Fast Path (Every Frame)
```python
# Per-frame processing loop
tracks = yolo_detector.track(frame)  # ByteTrack
track_ids = [t.track_id for t in tracks]

# Query cache
results = cache.get_batch(track_ids, frame_num)

for track in tracks:
    entry = results[track.track_id]
    if entry:
        render_overlay(entry.label, entry.bbox)  # Cache hit
    else:
        schedule_vlm(track)  # Cache miss, needs VLM
```

### Slow Path (Every 15 Frames or On-Demand)
```python
# VLM inference
vlm_output = model.infer(image, prompt)  # ~200ms

# Store in cache
entry = CacheEntry.from_vlm_output(
    track_id=track_id,
    vlm_dict=vlm_output,  # {"label": "person", "confidence": 0.92}
    bbox=bbox,
    frame_num=frame_num
)
cache.put(entry)

# Next 15 frames will be cache hits!
```

### With Slow Path Service
```python
# Enable local cache
export USE_LOCAL_SEMANTIC_CACHE=1

# Access shared cache instance
from slow_path.service.worker import get_local_cache
cache = get_local_cache()
```

## Design Decisions

**Why TTL-based expiration?**
Video frames are temporally correlated - an object labeled "person" at frame 100 is likely still a person at frame 105. TTL balances freshness with hit rate.

**Why track_id as key?**
ByteTrack maintains stable IDs across frames, making `track_id` the natural key for associating semantic labels with tracked objects.

**Why evict oldest by timestamp?**
Oldest entries are most likely stale. This simple policy performs well without LRU bookkeeping overhead.

**Why batch operations?**
Fast path queries 5-20 tracks per frame. Batching reduces lock contention and provides cleaner API than repeated `get()` calls.

## Testing

```bash
# Run all unit tests
python3 -m semantic_cache.tests.test_semantic_cache

# Expected output:
# Test 1: Basic put and get... PASS
# Test 2: TTL expiration... PASS
# Test 3: Batch operations... PASS
# ...
# All tests passed!
```

## Dependencies

**None** - Uses only Python standard library:
- `dataclasses` - for CacheEntry
- `threading` - for thread safety
- `typing` - for type hints

## Documentation

- **API Reference**: This README
- **Examples**: See [examples/](examples/) directory
- **Tests**: See [tests/](tests/) directory

