"""Integration example demonstrating fast/slow path with SemanticCache.

Shows how semantic cache enables real-time video processing by caching
VLM outputs across frames, reducing expensive inference calls by >90%.
"""
import random
import time
from semantic_cache import SemanticCache, CacheEntry


def simulate_video_pipeline(total_frames: int = 150):
    """Simulate a video processing pipeline with semantic caching."""
    cache = SemanticCache()
    
    # Fixed set of tracks that persist through video (more realistic)
    active_track_ids = [1, 2, 3, 4, 5]
    
    # VLM runs every N frames
    VLM_INTERVAL = 15
    
    # Statistics
    vlm_runs = 0
    frame_patterns = []  # Track hit/miss patterns
    
    print("=" * 60)
    print("SEMANTIC CACHE DEMO - Video Processing Pipeline")
    print("=" * 60)
    print(f"Settings: {total_frames} frames, VLM every {VLM_INTERVAL} frames")
    print(f"Tracking {len(active_track_ids)} objects\n")
    
    for frame in range(total_frames):
        frame_hits = 0
        frame_misses = 0
        
        # ========== FAST PATH (EVERY FRAME) ==========
        # Query cache for all tracked objects
        for track_id in active_track_ids:
            entry = cache.get(track_id, frame)
            
            if entry is not None:
                frame_hits += 1
                # In real system: use entry.label, entry.bbox for overlay
                if frame % 30 == 0 and track_id == active_track_ids[0]:  # Show example
                    print(f"  Cache HIT: Track {track_id} = '{entry.label}' (conf: {entry.confidence:.2f})")
            else:
                frame_misses += 1
                if frame % 30 == 0 and track_id == active_track_ids[0]:  # Show example
                    print(f"  Cache MISS: Track {track_id} (no semantic info)")
        
        # Record pattern for visualization
        if frame_hits == len(active_track_ids):
            frame_patterns.append("H")
        elif frame_misses == len(active_track_ids):
            frame_patterns.append("M")
        else:
            frame_patterns.append("P")  # Partial hit
        
        # ========== SLOW PATH (EVERY 15 FRAMES) ==========
        if frame % VLM_INTERVAL == 0:
            vlm_runs += 1
            print(f"\nFrame {frame}: 🤖 Running VLM inference...")
            
            # VLM generates labels for ALL visible tracks
            labels = ["person", "car", "bicycle", "dog", "truck"]
            for track_id in active_track_ids:
                # Simulate VLM output
                entry = CacheEntry(
                    track_id=track_id,
                    label=labels[(track_id - 1) % len(labels)],  # Consistent labels
                    bbox=[100 * track_id, 50, 80, 120],  # Fake bbox
                    confidence=0.85 + (track_id * 0.02),  # Varying confidence
                    timestamp=frame
                )
                cache.put(entry)
            
            print(f"  ✓ Updated {len(active_track_ids)} tracks in cache")
        
        # ========== PERIODIC STATS (EVERY SECOND @ 30FPS) ==========
        if frame > 0 and frame % 30 == 0:
            stats = cache.get_stats()
            print(f"\n--- Frame {frame} Stats ---")
            print(f"  Hit Rate: {stats['hit_rate']:.1%}")
            print(f"  Pattern (last 30): {''.join(frame_patterns[-30:])}")
            print(f"  (H=all hits, M=all misses, P=partial)")
            
        # Simulate real-time processing
        time.sleep(0.001)
    
    # ========== FINAL SUMMARY ==========
    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE - Summary")
    print("=" * 60)
    
    final_stats = cache.get_stats()
    total_queries = final_stats['hits'] + final_stats['misses']
    
    print(f"Cache Performance:")
    print(f"  - Total Queries: {total_queries}")
    print(f"  - Cache Hits: {final_stats['hits']}")
    print(f"  - Cache Misses: {final_stats['misses']}")
    print(f"  - Hit Rate: {final_stats['hit_rate']:.1%}")
    
    print(f"\nEfficiency Gains:")
    print(f"  - VLM Runs: {vlm_runs} (every {VLM_INTERVAL} frames)")
    print(f"  - Without Cache: {total_frames} (every frame)")
    print(f"  - Reduction: {(1 - vlm_runs/total_frames):.1%} fewer VLM calls")
    
    print(f"\nPattern Visualization (full sequence):")
    for i in range(0, len(frame_patterns), 60):
        print(f"  Frames {i:3d}-{min(i+59, len(frame_patterns)-1):3d}: {''.join(frame_patterns[i:i+60])}")


if __name__ == "__main__":