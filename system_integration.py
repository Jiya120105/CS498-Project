"""
System integration that properly connects fast path, slow path, and semantic cache.

This integration ensures that when USE_LOCAL_SEMANTIC_CACHE=1 is set, the slow path
worker and the hybrid processor share the same cache instance for proper coordination.
"""
import cv2
import time
import os
from typing import Dict, List, Optional
import sys

# Add current directory to path for imports
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from semantic_cache import SemanticCache, CacheEntry
from evaluation.fast_path_integration import FastPathProcessor
from evaluation.slow_path_integration import SlowPathProcessor
from evaluation.dataset_loader import BoundingBox, MOTDataset


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


class SystemIntegratedProcessor:
    """
    System-integrated processor that properly coordinates fast path, slow path,
    and semantic cache. Uses shared cache when USE_LOCAL_SEMANTIC_CACHE=1.
    """
    
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
        Process entire sequence with integrated system approach
        
        Returns:
            Dictionary with 'predictions', 'processing_times', and 'cache_stats'
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
                print(f"  Processing frame {idx + 1}/{total_frames} (frame {frame})...")
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
                
                # Store job IDs if we need to wait for results
                if self.wait_for_slow_path:
                    for enqueued in trigger_result.get('enqueued_track_ids', []):
                        track_id = enqueued['track_id']
                        job_id = enqueued['job_id']
                        if frame not in pending_jobs:
                            pending_jobs[frame] = {}
                        pending_jobs[frame][track_id] = job_id
            
            # Wait for slow path results if requested
            if self.wait_for_slow_path and frame in pending_jobs:
                for track_id, job_id in pending_jobs[frame].items():
                    result = self.slow_path.wait_for_result(job_id, timeout=5.0)
                    if result and result.get('status') == 'done':
                        record = result.get('record', {})
                        # Update cache with result (if not already updated by worker)
                        # Note: When using shared cache, the worker may have already updated it
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
            if not self.wait_for_slow_path and self.slow_path:
                # Check previous frames' pending jobs
                for prev_frame in list(pending_jobs.keys()):
                    if prev_frame < frame:
                        for track_id, job_id in list(pending_jobs[prev_frame].items()):
                            result = self.slow_path.get_result(job_id)
                            if result and result.get('status') == 'done':
                                # When using shared cache, the worker has already updated it
                                # But we can still check the cache to see if it's there
                                cache_entry = self.cache.get(track_id, frame)
                                if cache_entry is None:
                                    # Fallback: update cache from result
                                    record = result.get('record', {})
                                    cache_entry = CacheEntry(
                                        track_id=track_id,
                                        label=record.get('label', ''),
                                        bbox=record.get('bbox', []),
                                        confidence=record.get('confidence', 0.0),
                                        timestamp=prev_frame
                                    )
                                    self.cache.put(cache_entry)
                                # Remove from pending
                                del pending_jobs[prev_frame][track_id]
                        # Clean up empty frames
                        if not pending_jobs[prev_frame]:
                            del pending_jobs[prev_frame]
            
            # Total processing time
            processing_time = time.perf_counter() - frame_start_time
            processing_times.append(processing_time)
            
            # Store predictions (detections with potential semantic labels from cache)
            predictions[frame] = detections
        
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

