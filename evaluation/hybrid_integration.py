"""
Hybrid integration combining fast path, slow path, and semantic cache
"""
import cv2
import time
from typing import Dict, List, Optional
import sys
import os

# Add parent directory to path for semantic_cache
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semantic_cache import SemanticCache, CacheEntry
from .fast_path_integration import FastPathProcessor
from .slow_path_integration import SlowPathProcessor
from .dataset_loader import BoundingBox, MOTDataset


def get_shared_cache() -> Optional[SemanticCache]:
    """
    Get the shared cache instance if USE_LOCAL_SEMANTIC_CACHE is enabled.
    
    Returns:
        The shared SemanticCache instance if available, None otherwise.
    """
    use_local_cache = os.getenv("USE_LOCAL_SEMANTIC_CACHE", "0") == "1"
    if use_local_cache:
        try:
            from slow_path.service.worker import get_local_cache
            shared_cache = get_local_cache()
            if shared_cache is not None:
                return shared_cache
        except Exception as e:
            print(f"Warning: Could not get shared cache: {e}")
    return None


class HybridProcessor:
    """Hybrid processor combining fast path with selective slow path"""
    
    def __init__(self, model_name: str = 'yolov8n.pt', tracker_config: str = "bytetrack.yaml",
                 use_slow_path: bool = True, wait_for_slow_path: bool = False,
                 cache_max_size: int = 1000):
        """
        Args:
            model_name: YOLO model name
            tracker_config: Tracker config file name
            use_slow_path: Whether to use slow path for VLM processing
            wait_for_slow_path: Whether to wait for slow path results (blocking)
            cache_max_size: Maximum cache size (only used if creating new cache)
        """
        self.fast_path = FastPathProcessor(model_name, tracker_config)
        self.slow_path = SlowPathProcessor() if use_slow_path else None
        self.wait_for_slow_path = wait_for_slow_path
        self.use_slow_path = use_slow_path
        
        # Try to use shared cache if available, otherwise create new one
        shared_cache = get_shared_cache()
        if shared_cache is not None:
            self.cache = shared_cache
            print("Using shared semantic cache from slow path worker")
        else:
            self.cache = SemanticCache(max_size=cache_max_size)
            print("Using independent semantic cache instance")
    
    def process_sequence(self, dataset: MOTDataset) -> Dict:
        """
        Process entire sequence with hybrid approach
        
        Returns:
            Dictionary with 'predictions' and 'processing_times'
        """
        all_frames = sorted(dataset.det_data.keys()) if dataset.det_data else []
        if not all_frames:
            all_frames = sorted(dataset.gt_data.keys()) if dataset.gt_data else []
        
        predictions = {}
        processing_times = []
        
        # Track pending slow path jobs
        pending_jobs = {}  # frame_id -> {track_id: job_id}
        total_frames = len(all_frames)
        
        for idx, frame in enumerate(all_frames):
            if (idx + 1) % 100 == 0 or idx == 0:
                # Show cache stats periodically
                cache_stats = self.cache.get_stats()
                print(f"  Processing frame {idx + 1}/{total_frames} (frame {frame})... "
                      f"[Cache: {cache_stats['hits']} hits, {cache_stats['misses']} misses, "
                      f"{cache_stats['hit_rate']:.1%} hit rate]")
            frame_start_time = time.perf_counter()
            
            img_path = dataset.get_frame_image_path(frame)
            if not img_path:
                processing_times.append(0.0)
                predictions[frame] = []
                continue
            
            # Load image
            frame_image = cv2.imread(img_path)
            if frame_image is None:
                processing_times.append(0.0)
                predictions[frame] = []
                continue
            
            # Fast path: detect and track
            detections, fast_time = self.fast_path.process_frame(frame_image)
            
            # Update frame numbers
            for det in detections:
                det.frame = frame
            
            # Check cache for semantic labels
            track_ids = [det.track_id for det in detections]
            cache_results = self.cache.get_batch(track_ids, frame)
            
            # Count cache hits/misses for this frame
            frame_cache_hits = sum(1 for tid in track_ids if cache_results.get(tid) is not None)
            frame_cache_misses = len(track_ids) - frame_cache_hits
            
            # Prepare ROIs for slow path (if needed)
            rois_for_slow_path = []
            if self.use_slow_path and self.slow_path:
                for det in detections:
                    track_id = det.track_id
                    cache_entry = cache_results.get(track_id)
                    
                    # If cache miss, add to slow path queue
                    if cache_entry is None:
                        # Convert bbox from (x, y, w, h) to [x, y, w, h] list
                        rois_for_slow_path.append({
                            'track_id': track_id,
                            'bbox': [int(det.x), int(det.y), int(det.w), int(det.h)]
                        })
            
            # Trigger slow path if needed
            if rois_for_slow_path and self.slow_path:
                frame_b64 = self.slow_path.cv2_to_b64(frame_image)
                trigger_result = self.slow_path.trigger_tick(
                    frame_id=frame,
                    image_b64=frame_b64,
                    rois=rois_for_slow_path
                )
                
                # Store job IDs for tracking (needed for both blocking and non-blocking modes)
                for enqueued in trigger_result.get('enqueued_track_ids', []):
                    track_id = enqueued['track_id']
                    job_id = enqueued['job_id']
                    if frame not in pending_jobs:
                        pending_jobs[frame] = {}
                    pending_jobs[frame][track_id] = job_id
            
            # Wait for slow path results if requested
            if self.wait_for_slow_path and frame in pending_jobs:
                for track_id, job_id in pending_jobs[frame].items():
                    result = self.slow_path.wait_for_result(job_id, timeout=30.0)
                    if result and result.get('status') == 'done':
                        record = result.get('record', {})
                        # Update cache with result (if not already updated by worker)
                        # Note: When using shared cache, the worker may have already updated it
                        cache_entry = self.cache.get(track_id, frame)
                        if cache_entry is None:
                            cache_entry = CacheEntry(
                                track_id=track_id,
                                label=record.get('label', ''),
                                bbox=record.get('bbox', []),
                                confidence=record.get('confidence', 0.0),
                                timestamp=frame
                            )
                            self.cache.put(cache_entry)
                        # Update cache results for this frame
                        cache_results[track_id] = cache_entry
            
            # Check for results from previous frames (non-blocking)
            # Only check recent frames to avoid O(n²) complexity
            # With shared cache, worker already stores results, so we just clean up old pending jobs
            if not self.wait_for_slow_path and self.slow_path:
                # Only check recent frames (last 20 frames) to limit overhead
                # With shared cache, results are already in cache, so we mainly clean up tracking
                frames_to_check = sorted([f for f in pending_jobs.keys() if f < frame])[-20:]
                for prev_frame in frames_to_check:
                    if prev_frame not in pending_jobs:
                        continue
                    # Quick check: if using shared cache, worker already updated it
                    # Just verify and clean up tracking
                    jobs_to_remove = []
                    for track_id, job_id in list(pending_jobs[prev_frame].items()):
                        # Check cache first (fast) - if using shared cache, worker already put it there
                        cache_entry = self.cache.get(track_id, frame)
                        if cache_entry is not None:
                            # Result is in cache (either from worker or previous check), clean up
                            jobs_to_remove.append(track_id)
                        else:
                            # Only poll if cache miss and frame is recent (within last 5 frames)
                            if frame - prev_frame <= 5:
                                result = self.slow_path.get_result(job_id)
                                if result and result.get('status') == 'done':
                                    # Update cache from result (for non-shared cache mode)
                                    record = result.get('record', {})
                                    cache_entry = CacheEntry(
                                        track_id=track_id,
                                        label=record.get('label', ''),
                                        bbox=record.get('bbox', []),
                                        confidence=record.get('confidence', 0.0),
                                        timestamp=prev_frame
                                    )
                                    self.cache.put(cache_entry)
                                    jobs_to_remove.append(track_id)
                    
                    # Remove completed jobs
                    for track_id in jobs_to_remove:
                        del pending_jobs[prev_frame][track_id]
                    
                    # Clean up empty frames
                    if not pending_jobs[prev_frame]:
                        del pending_jobs[prev_frame]
            
            # Total processing time
            processing_time = time.perf_counter() - frame_start_time
            processing_times.append(processing_time)
            
            # Store predictions (detections with potential semantic labels from cache)
            predictions[frame] = detections
        
        # Final cleanup: wait for remaining pending results (for cache population)
        if self.slow_path and pending_jobs:
            total_pending = sum(len(jobs) for jobs in pending_jobs.values())
            print(f"  Waiting for {total_pending} pending slow path results...")
            max_wait_time = 30.0  # Wait up to 30 seconds for remaining results
            start_wait = time.perf_counter()
            results_stored = 0
            
            while pending_jobs and (time.perf_counter() - start_wait) < max_wait_time:
                for prev_frame in list(pending_jobs.keys()):
                    for track_id, job_id in list(pending_jobs[prev_frame].items()):
                        result = self.slow_path.get_result(job_id)
                        if result and result.get('status') == 'done':
                            # Check if already in cache (shared cache mode - worker may have stored it)
                            cache_entry = self.cache.get(track_id, prev_frame)
                            if cache_entry is None:
                                # Update cache from result (for non-shared cache mode)
                                record = result.get('record', {})
                                cache_entry = CacheEntry(
                                    track_id=track_id,
                                    label=record.get('label', ''),
                                    bbox=record.get('bbox', []),
                                    confidence=record.get('confidence', 0.0),
                                    timestamp=prev_frame
                                )
                                self.cache.put(cache_entry)
                                results_stored += 1
                            else:
                                # Already in cache (shared cache mode)
                                results_stored += 1
                            del pending_jobs[prev_frame][track_id]
                    # Clean up empty frames
                    if not pending_jobs[prev_frame]:
                        del pending_jobs[prev_frame]
                time.sleep(0.1)  # Small delay between checks
            
            if results_stored > 0:
                print(f"  Stored {results_stored} results in cache")
            cache_stats_after = self.cache.get_stats()
            print(f"  Cache after cleanup: {cache_stats_after['cache_size']} entries, "
                  f"{cache_stats_after['hits']} hits, {cache_stats_after['misses']} misses")
        
        # Clean up
        if self.slow_path:
            self.slow_path.close()
        
        print(f"  Completed processing {total_frames} frames")
        cache_stats = self.cache.get_stats()
        print(f"  Cache stats: {cache_stats['hits']} hits, {cache_stats['misses']} misses, "
              f"hit rate: {cache_stats['hit_rate']:.2%}")
        return {
            'predictions': predictions,
            'processing_times': processing_times,
            'cache_stats': cache_stats
        }

