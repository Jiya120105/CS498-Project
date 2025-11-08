import random
import time
from semantic_cache import SemanticCache, CacheEntry

def simulate_cache(frames=100, use_cache=True):
    """
    Simulate video processing pipeline with/without semantic caching
    Returns statistics for performance comparison
    """
    cache = SemanticCache() if use_cache else None
    # Real tracks (ground truth)
    real_track_ids = [1, 2, 3]  # Only 3 real objects
    
    # Simulation parameters
    VLM_INTERVAL = 15  # VLM runs every N frames
    FAST_PATH_TIME = 0.01  # 10ms for object detection
    SLOW_PATH_TIME = 0.2   # 200ms for VLM inference per object
    
    # Fast Path noise parameters
    FALSE_POSITIVE_RATE = 0.2  # 20% chance of false detection per frame
    TRACK_LOSS_RATE = 0.1    # 10% chance to temporarily lose a real track
    
    stats = {
        "frame": [],
        "hits": [],
        "misses": [],
        "hit_rate": [],
        "hit_rate_percent": [],
        "cache_size": [],
        "fast_path_time": [],
        "slow_path_time": [],
        "total_processing_time": [],
        "cache_benefit": [],
        "pattern": [],  # Track hit/miss patterns
        "cumulative_real_hits": [],  # Track cumulative hits for real tracks
        "cumulative_real_misses": []  # Track cumulative misses for real tracks
    }
    
    # Initialize cumulative counters
    cumulative_real_hits = 0
    cumulative_real_misses = 0

    labels = ["person", "car", "bicycle", "dog", "truck"]

    for frame in range(frames):
        if frame % 20 == 0:  # Show progress every 20 frames
            mode = "with cache" if use_cache else "without cache"
            print(f"\rProcessing frame {frame}/{frames} ({mode}) - {frame/frames*100:.1f}%", end="", flush=True)
            
        frame_start_time = time.perf_counter()
        frame_hits = 0
        frame_misses = 0

        # Fast path: object detection always runs
        time.sleep(FAST_PATH_TIME)  # Simulate YOLO processing
        fast_time = FAST_PATH_TIME

        # Calculate active tracks for this frame (real + false positives)
        active_track_ids = []
        # Add real tracks (with potential track loss)
        for track_id in real_track_ids:
            if random.random() > TRACK_LOSS_RATE:  # 90% chance to keep track
                active_track_ids.append(track_id)
        
        # Add false positive tracks
        num_false_positives = int(len(real_track_ids) * FALSE_POSITIVE_RATE + 0.5)  # Round to nearest int
        for _ in range(num_false_positives):
            false_track_id = random.randint(len(real_track_ids) + 1, len(real_track_ids) + 10)  # Higher IDs for false positives
            active_track_ids.append(false_track_id)

        # Query cache for semantic labels FIRST
        if use_cache:
            # Use batch get for efficiency and accurate stats
            results = cache.get_batch(active_track_ids, frame)
            frame_hits = sum(1 for entry in results.values() if entry is not None)
            frame_misses = len(active_track_ids) - frame_hits
            
            # Calculate hits/misses for real tracks only (more meaningful metric)
            real_track_hits = sum(1 for tid in active_track_ids 
                                 if tid in real_track_ids and results.get(tid) is not None)
            real_track_misses = sum(1 for tid in active_track_ids 
                                   if tid in real_track_ids and results.get(tid) is None)
        else:
            frame_hits = 0
            frame_misses = len(active_track_ids)  # Without cache, always miss
            real_track_hits = 0
            real_track_misses = sum(1 for tid in active_track_ids if tid in real_track_ids)
            results = {}

        # Slow path: VLM inference when needed (run AFTER querying to see what's missing)
        slow_time = 0
        if (not use_cache) or (frame % VLM_INTERVAL == 0):
            # Simulate VLM processing for required tracks
            if use_cache:
                # On VLM frames, process tracks that are missing or stale
                # This includes: tracks that weren't in cache, or tracks that were stale
                tracks_to_process = []
                for tid in active_track_ids:
                    if tid in real_track_ids:
                        # Check if this track needs VLM processing (miss or stale)
                        entry = results.get(tid)
                        if entry is None:  # Miss or stale
                            tracks_to_process.append(tid)
            else:
                # Without cache, process all detected real tracks every frame
                tracks_to_process = [tid for tid in active_track_ids if tid in real_track_ids]

            # Simulate VLM inference time
            vlm_time = len(tracks_to_process) * SLOW_PATH_TIME
            time.sleep(vlm_time)
            
            if use_cache:
                for track_id in tracks_to_process:
                    entry = CacheEntry(
                        track_id=track_id,
                        label=labels[(track_id - 1) % len(labels)],
                        bbox=[100 * track_id, 50, 80, 120],
                        confidence=0.85 + (track_id * 0.02),
                        timestamp=frame
                    )
                    cache.put(entry)
            
            slow_time = vlm_time  # Record actual VLM processing time

        # Record pattern
        if use_cache:
            if frame_hits == len(active_track_ids):
                stats["pattern"].append("H")  # All hits
            elif frame_misses == len(active_track_ids):
                stats["pattern"].append("M")  # All misses
            else:
                stats["pattern"].append("P")  # Partial hits
        else:
            stats["pattern"].append("M")

        total_time = fast_time + slow_time
        
        # Calculate time that would have been spent without cache
        # Only count real tracks (not false positives) for fair comparison
        real_tracks_detected = [tid for tid in active_track_ids if tid in real_track_ids]
        no_cache_time = SLOW_PATH_TIME * len(real_tracks_detected)
        
        # Collect stats
        stats["frame"].append(frame)
        stats["fast_path_time"].append(fast_time)
        stats["slow_path_time"].append(slow_time)
        stats["total_processing_time"].append(total_time)
        
        if use_cache:
            # Track per-frame stats
            stats["hits"].append(frame_hits)
            stats["misses"].append(frame_misses)
            
            # Update cumulative counters for real tracks only
            cumulative_real_hits += real_track_hits
            cumulative_real_misses += real_track_misses
            stats["cumulative_real_hits"].append(cumulative_real_hits)
            stats["cumulative_real_misses"].append(cumulative_real_misses)
            
            # Calculate cumulative hit rate (more meaningful than per-frame)
            total_real_queries_cumulative = cumulative_real_hits + cumulative_real_misses
            if total_real_queries_cumulative > 0:
                hit_rate = cumulative_real_hits / total_real_queries_cumulative
                # Ensure hit_rate is between 0 and 1
                hit_rate = max(0.0, min(1.0, hit_rate))
            else:
                # No real tracks queried yet - can't calculate meaningful hit rate
                hit_rate = 0.0  # Start at 0% when no queries yet
            hit_rate_percent = hit_rate * 100
            stats["hit_rate"].append(hit_rate)
            stats["hit_rate_percent"].append(hit_rate_percent)
            
            # Get actual number of entries in cache
            stats["cache_size"].append(len(cache._cache) if cache else 0)
            # Calculate actual time saved compared to no-cache scenario
            stats["cache_benefit"].append(no_cache_time - slow_time)
        else:
            stats["hits"].append(0)
            stats["misses"].append(len(active_track_ids))  # All queries are misses
            stats["hit_rate"].append(0.0)
            stats["hit_rate_percent"].append(0.0)
            stats["cache_size"].append(0)
            stats["cache_benefit"].append(0)
            stats["cumulative_real_hits"].append(0)
            stats["cumulative_real_misses"].append(0)

        # Simulate real-time processing
        time.sleep(0.001)

    mode = "with cache" if use_cache else "without cache"
    print(f"\rProcessed {frames} frames ({mode}) - 100%")  # Final progress update
    
    # Print summary statistics
    if use_cache:
        total_hits = sum(stats["hits"])
        total_misses = sum(stats["misses"])
        # Calculate average hit rate (excluding frames with no real tracks detected)
        valid_hit_rates = [hr for hr in stats["hit_rate"] if hr is not None]
        avg_hit_rate = sum(valid_hit_rates) / len(valid_hit_rates) if valid_hit_rates else 0.0
        print(f"  Cache Summary: {total_hits} hits, {total_misses} misses, {avg_hit_rate:.1%} avg hit rate")
    
    return stats
