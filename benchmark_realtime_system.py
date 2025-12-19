"""
Real-Time System Benchmark

Simulates processing MOT16 videos at 30 FPS with actual system constraints:
- Queue limits
- Stability requirements
- Track drops on overflow
- Real-time throughput measurement

Tests vanilla and optimized VLM approaches to compare real-time performance.

Usage:
    python benchmark_realtime_system.py --approach vanilla
    python benchmark_realtime_system.py --approach cached --threshold 0.93
    python benchmark_realtime_system.py --approach int8
    python benchmark_realtime_system.py --approach combined
"""

import argparse
import os
import cv2
import time
import json
import numpy as np
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from collections import defaultdict

os.environ["USE_LOCAL_SEMANTIC_CACHE"] = "0"

# Configuration
VIDEOS = [
    "football_video",
]
QUERY = "Is this a player with the possession in the football match?"
STABLE_MIN_FRAMES = 5
QUEUE_SIZE = 50
TARGET_FPS = 30
FRAME_TIME_MS = 1000.0 / TARGET_FPS  # 33.33ms per frame
LOG_FILE = "football_system_events.jsonl"

print("="*70)
print("REAL-TIME SYSTEM BENCHMARK")
print("="*70)


def load_ground_truth():
    """Load ground truth labels for accuracy measurement."""
    gt_file = Path("football_dataset/ground_truth.json")
    if not gt_file.exists():
        # Fallback to empty if not found (we might be running just for visual demo)
        print(f"⚠️  Ground truth not found: {gt_file}")
        return {}

    with open(gt_file, 'r') as f:
        data = json.load(f)

    return data['tracks']


    with open(gt_file, 'r') as f:
        data = json.load(f)

    return data['tracks']


def load_vlm_model(approach: str, device: str, threshold: float = 0.93):
    """
    Load VLM model based on approach.

    Args:
        approach: 'vanilla', 'cached', 'int8', 'pruned', 'combined'
        device: 'cuda' or 'cpu'
        threshold: Similarity threshold for cached approach
    """
    print(f"\n[Loading VLM] Approach: {approach}")

    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-500M-Instruct")
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"

    if approach == "vanilla":
        # Standard FP16 model
        if device == "cuda":
            model = AutoModelForVision2Seq.from_pretrained(
                "HuggingFaceTB/SmolVLM-500M-Instruct",
                torch_dtype=torch.float16,
                _attn_implementation="eager"
            ).to(device)
        else:
            model = AutoModelForVision2Seq.from_pretrained(
                "HuggingFaceTB/SmolVLM-500M-Instruct",
                _attn_implementation="eager"
            ).to(device)
        model.eval()
        vlm_wrapper = VanillaVLMWrapper(model, processor, device)

    elif approach == "cached":
        # Cached VLM with real vision embeddings
        from improved_cached_vlm import ImprovedCachedVLM

        if device == "cuda":
            base_model = AutoModelForVision2Seq.from_pretrained(
                "HuggingFaceTB/SmolVLM-500M-Instruct",
                torch_dtype=torch.float16,
                _attn_implementation="eager"
            ).to(device)
        else:
            base_model = AutoModelForVision2Seq.from_pretrained(
                "HuggingFaceTB/SmolVLM-500M-Instruct",
                _attn_implementation="eager"
            ).to(device)
        base_model.eval()

        vlm_wrapper = ImprovedCachedVLM(
            base_model, processor,
            cache_size=200,
            similarity_threshold=threshold,
            device=device,
            embedding_layers=[6, 7]
        )

    elif approach == "int8":
        # True INT8 quantization
        from true_int8_vlm import load_quantized_vlm
        vlm_wrapper = load_quantized_vlm(processor, device, quantization_ratio=0.5)

    elif approach == "adaptive":
        # Online Adaptive INT8 Quantization
        from adaptive_int8_vlm import AdaptiveINT8VLM
        vlm_wrapper = AdaptiveINT8VLM(
            processor,
            device=device,
            profiling_samples=10,
            inference_samples=90,
            quantization_ratio=0.5,
            correlation_threshold=0.9
        )

    elif approach == "combined":
        # Cached + INT8 (can add pruning later)
        print("⚠️  Combined approach not yet implemented")
        print("   Using cached approach for now")
        from improved_cached_vlm import ImprovedCachedVLM

        if device == "cuda":
            base_model = AutoModelForVision2Seq.from_pretrained(
                "HuggingFaceTB/SmolVLM-500M-Instruct",
                torch_dtype=torch.float16,
                _attn_implementation="eager"
            ).to(device)
        else:
            base_model = AutoModelForVision2Seq.from_pretrained(
                "HuggingFaceTB/SmolVLM-500M-Instruct",
                _attn_implementation="eager"
            ).to(device)
        base_model.eval()

        vlm_wrapper = ImprovedCachedVLM(
            base_model, processor,
            cache_size=200,
            similarity_threshold=threshold,
            device=device
        )

    else:
        raise ValueError(f"Unknown approach: {approach}")

    print(f"✓ VLM loaded")
    return vlm_wrapper


class VanillaVLMWrapper:
    """Wrapper for vanilla VLM to match interface."""

    def __init__(self, model, processor, device):
        self.model = model
        self.processor = processor
        self.device = device

    def infer(self, image: Image.Image, prompt: str, track_id=None):
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=text_prompt, images=[image], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=50)

        answer = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        if "Assistant:" in answer:
            answer = answer.split("Assistant:")[-1].strip()

        # Parse
        lower_ans = answer.lower()
        if "yes" in lower_ans:
            label = "Yes"
        elif "no" in lower_ans:
            label = "No"
        else:
            label = answer[:50]

        return {
            "label": label,
            "confidence": 0.95,
            "metadata": {"cache_hit": False}
        }

    def get_cache_stats(self):
        return {}


def simulate_realtime_processing(videos, vlm_wrapper, device, ground_truth):
    """
    Simulate real-time video processing at 30 FPS.

    Returns performance metrics.
    """
    print(f"\n{'='*70}")
    print("SIMULATING REAL-TIME PROCESSING")
    print(f"{'='*70}")
    print(f"Target FPS: {TARGET_FPS}")
    print(f"Frame time budget: {FRAME_TIME_MS:.1f}ms")
    print(f"Queue size limit: {QUEUE_SIZE}")
    print(f"Stability requirement: {STABLE_MIN_FRAMES} consecutive frames\n")

    # Load YOLO
    yolo = YOLO('yolov8n.pt')

    # Metrics
    total_frames = 0
    total_tracks_detected = set()
    tracks_evaluated = {}  # track_id -> label
    tracks_dropped = set()
    queue_depths = []
    vlm_latencies = []
    frame_latencies = []
    cache_hits = 0
    cache_misses = 0

    # Simulated queue (list of pending tracks)
    pending_queue = []

    # Track stability state
    consecutive_seen = {}
    last_seen_frame = {}
    evaluated_tracks = set()

    # VIRTUAL CLOCK STATE
    # We decouple "video time" from "processing time".
    # - current_sim_time: The timestamp of the current frame in the video stream (advances by 33.3ms fixed)
    # - vlm_free_time: The timestamp when the VLM worker will become free (advances by actual VLM latency)
    current_sim_time = 0.0
    vlm_free_time = 0.0
    
    # Open Log File
    with open(LOG_FILE, 'w') as log_f:
        pass # Clear file

    # Process each video
    for video_path in videos:

            video_name = Path(video_path).name

            print(f"\nProcessing: {video_name}")

    

                    img_dir = Path(video_path) / "img1"

    

                    if not img_dir.exists():

    

                        print(f"   ✗ Not found: {img_dir}")

    

                        continue

    

            

    

                    frame_files = sorted(list(img_dir.glob("*.jpg")))

    

                    print(f"DEBUG: Found {len(frame_files)} frames in {img_dir}")

    

            

    

                    for frame_idx, frame_file in enumerate(frame_files):

                # 1. Advance Virtual Clock (The Camera never stops)

                current_sim_time += FRAME_TIME_MS

                

                frame_start = time.perf_counter()

    

                frame = cv2.imread(str(frame_file))

                if frame is None:

                    continue

    

                total_frames += 1

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    

                # FAST PATH: YOLO tracking (Runs on every frame, consumes frame budget)

                # We assume YOLO takes ~10-15ms, fitting within the 33ms budget.

                # In a real threaded app, this runs in main thread.

                results = yolo.track(frame, persist=True, verbose=False, tracker="bytetrack.yaml")

    

                current_frame_logs = []

    

                if results[0].boxes.id is None or len(results[0].boxes) == 0:

                    pass # No tracks, but time still passes

                else:

                    boxes = results[0].boxes.xyxy.cpu().numpy()

                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)

                    classes = results[0].boxes.cls.cpu().numpy()

    

                    for i, cls_id in enumerate(classes):

                        if cls_id != 0:  # Not a person

                            continue

    

                        local_track_id = int(track_ids[i])

                        if "football" in video_name:

                            global_track_id = f"football_track_{local_track_id}"

                        else:

                            global_track_id = f"{video_name}_track_{local_track_id}"

                        

                        total_tracks_detected.add(global_track_id)

    

                        # Update stability

                        prev = last_seen_frame.get(global_track_id)

                        if prev is not None and prev == frame_idx:

                            consecutive_seen[global_track_id] = consecutive_seen.get(global_track_id, 0) + 1

                        else:

                            consecutive_seen[global_track_id] = 1

                        last_seen_frame[global_track_id] = frame_idx + 1

    

                        # Check if should evaluate (stable and not yet evaluated)

                        if (global_track_id not in evaluated_tracks and

                            global_track_id not in [t[0] for t in pending_queue] and

                            consecutive_seen.get(global_track_id, 0) >= STABLE_MIN_FRAMES):

    

                            x1, y1, x2, y2 = map(int, boxes[i])

                            roi = frame_rgb[y1:y2, x1:x2]

    

                            if roi.size > 0:

                                # Try to add to queue

                                if len(pending_queue) < QUEUE_SIZE:

                                    pending_queue.append((global_track_id, Image.fromarray(roi)))

                                else:

                                    # Queue full - drop track

                                    tracks_dropped.add(global_track_id)

                        

                        # LOGGING PREP

                        x1, y1, x2, y2 = map(int, boxes[i])

                        status = "Resolved" if global_track_id in evaluated_tracks else "Pending"

                        label = tracks_evaluated.get(global_track_id, "Unknown")

                        

                        current_frame_logs.append({

                            "frame": frame_idx + 1, # 1-based for render_video.py

                            "track_id": local_track_id,

                            "bbox": [x1, y1, x2, y2],

                            "status": status,

                            "semantic_label": label,

                            "confidence": 1.0 if status == "Resolved" else 0.0

                        })

    

                # SLOW PATH: VLM Worker (Runs in parallel background thread)

                # Logic: If the worker is free at current_sim_time, it picks a job.

                if pending_queue and current_sim_time >= vlm_free_time:

                    track_id, roi_image = pending_queue.pop(0)

    

                    # Run VLM (Synchronous for us, but we treat duration as "busy time")

                    if device == "cuda":

                        torch.cuda.synchronize()

                    vlm_start = time.perf_counter()

    

                    result = vlm_wrapper.infer(roi_image, QUERY, track_id=track_id)

    

                    if device == "cuda":

                        torch.cuda.synchronize()

                    

                    # Actual wall-clock time taken by VLM

                    vlm_latency = (time.perf_counter() - vlm_start) * 1000

                    vlm_latencies.append(vlm_latency)

    

                    # Mark worker as busy until future time

                    # Ideally, worker became busy at max(current_sim_time, vlm_free_time)

                    # But since we only check when current > free, we are effectively starting NOW.

                    vlm_free_time = current_sim_time + vlm_latency

    

                    tracks_evaluated[track_id] = result['label']

                    evaluated_tracks.add(track_id)

    

                    if result['metadata'].get('cache_hit', False):

                        cache_hits += 1

                    else:

                        cache_misses += 1

                

            # WRITE LOGS
            if current_frame_logs:
                with open(LOG_FILE, 'a') as log_f:
                    for entry in current_frame_logs:
                        log_f.write(json.dumps(entry) + "\n")

            # Record queue depth
            queue_depths.append(len(pending_queue))

            # Frame processing time (just for logging stats, not for flow control)
            frame_time = (time.perf_counter() - frame_start) * 1000
            frame_latencies.append(frame_time)

            # Progress
            if (frame_idx + 1) % 100 == 0:
                avg_queue = np.mean(queue_depths[-100:])
                print(f"   Frame {frame_idx+1}/{len(frame_files)}: "
                      f"Queue={avg_queue:.1f}, "
                      f"SimTime={current_sim_time/1000:.1f}s, "
                      f"Evaluated={len(tracks_evaluated)}, Dropped={len(tracks_dropped)}")

    # In a real real-time system, when the video ends, the system stops.
    # We DO NOT process the remaining queue. If it wasn't processed by the end of the video,
    # it effectively "timed out" or the operator stopped the feed.
    # (Optional: You could allow a grace period, but strict real-time usually stops).
    print(f"\nVideo ended. Dropping {len(pending_queue)} items remaining in queue.")
    for t_id, _ in pending_queue:
        tracks_dropped.add(t_id)

    # Calculate accuracy against ground truth
    correct = 0
    total_with_gt = 0
    precision_tp = 0
    precision_fp = 0
    recall_tp = 0
    recall_fn = 0

    for track_id, predicted_label in tracks_evaluated.items():
        if track_id in ground_truth:
            gt_label = ground_truth[track_id]['label']
            total_with_gt += 1

            if predicted_label == gt_label:
                correct += 1

            # Precision/Recall for "Yes" class
            if predicted_label == "Yes":
                if gt_label == "Yes":
                    precision_tp += 1
                else:
                    precision_fp += 1

            if gt_label == "Yes":
                if predicted_label == "Yes":
                    recall_tp += 1
                else:
                    recall_fn += 1

    accuracy = (correct / total_with_gt * 100) if total_with_gt > 0 else 0
    precision = (precision_tp / (precision_tp + precision_fp)) if (precision_tp + precision_fp) > 0 else 0
    recall = (recall_tp / (recall_tp + recall_fn)) if (recall_tp + recall_fn) > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

    # Return metrics (Cast to native Python types for JSON serialization)
    return {
        "total_frames": int(total_frames),
        "total_tracks_detected": int(len(total_tracks_detected)),
        "tracks_evaluated": int(len(tracks_evaluated)),
        "tracks_dropped": int(len(tracks_dropped)),
        "evaluation_coverage_pct": float(len(tracks_evaluated) / len(total_tracks_detected) * 100 if total_tracks_detected else 0),
        "drop_rate_pct": float(len(tracks_dropped) / len(total_tracks_detected) * 100 if total_tracks_detected else 0),
        "accuracy_pct": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "vlm_latency_mean_ms": float(np.mean(vlm_latencies)) if vlm_latencies else 0.0,
        "vlm_latency_median_ms": float(np.median(vlm_latencies)) if vlm_latencies else 0.0,
        "vlm_latency_p95_ms": float(np.percentile(vlm_latencies, 95)) if vlm_latencies else 0.0,
        "vlm_latency_std_ms": float(np.std(vlm_latencies)) if vlm_latencies else 0.0,
        "frame_latency_mean_ms": float(np.mean(frame_latencies)) if frame_latencies else 0.0,
        "queue_depth_mean": float(np.mean(queue_depths)) if queue_depths else 0.0,
        "queue_depth_max": int(np.max(queue_depths)) if queue_depths else 0,
        "cache_hits": int(cache_hits),
        "cache_misses": int(cache_misses),
        "cache_hit_rate_pct": float(cache_hits / (cache_hits + cache_misses) * 100 if (cache_hits + cache_misses) > 0 else 0),
        "effective_fps": float(total_frames / (np.sum(frame_latencies) / 1000) if frame_latencies else 0),
        "throughput_tracks_per_sec": float(len(tracks_evaluated) / (np.sum(vlm_latencies) / 1000) if vlm_latencies else 0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--approach", default="vanilla",
                        choices=["vanilla", "cached", "int8", "adaptive", "pruned", "combined"],
                        help="VLM approach to benchmark")
    parser.add_argument("--threshold", type=float, default=0.93,
                        help="Similarity threshold for cached approach")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run on")
    args = parser.parse_args()

    print(f"\nApproach: {args.approach}")
    print(f"Device: {args.device}")
    if args.approach == "cached":
        print(f"Similarity threshold: {args.threshold}")

    # Load ground truth
    print(f"\nLoading ground truth...")
    ground_truth = load_ground_truth()
    print(f"✓ Loaded {len(ground_truth)} ground truth labels")

    # Load VLM
    vlm_wrapper = load_vlm_model(args.approach, args.device, args.threshold)

    # Run simulation
    metrics = simulate_realtime_processing(VIDEOS, vlm_wrapper, args.device, ground_truth)

    # Get cache stats if available
    cache_stats = vlm_wrapper.get_cache_stats() if hasattr(vlm_wrapper, 'get_cache_stats') else {}

    # Print results
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")

    print(f"\n📊 Coverage:")
    print(f"   Total tracks detected: {metrics['total_tracks_detected']}")
    print(f"   Tracks evaluated:      {metrics['tracks_evaluated']} ({metrics['evaluation_coverage_pct']:.1f}%)")
    print(f"   Tracks dropped:        {metrics['tracks_dropped']} ({metrics['drop_rate_pct']:.1f}%)")

    print(f"\n⏱️  Latency:")
    print(f"   VLM mean latency:  {metrics['vlm_latency_mean_ms']:.0f} ms")
    print(f"   VLM P95 latency:   {metrics['vlm_latency_p95_ms']:.0f} ms")
    print(f"   Frame processing:  {metrics['frame_latency_mean_ms']:.1f} ms/frame")

    print(f"\n📈 Throughput:")
    print(f"   Effective FPS:         {metrics['effective_fps']:.1f}")
    print(f"   Tracks/second:         {metrics['throughput_tracks_per_sec']:.1f}")
    print(f"   Avg queue depth:       {metrics['queue_depth_mean']:.1f}")
    print(f"   Max queue depth:       {metrics['queue_depth_max']}")

    print(f"\n🎯 Accuracy:")
    print(f"   Accuracy:   {metrics['accuracy_pct']:.1f}%")
    print(f"   Precision:  {metrics['precision']:.3f}")
    print(f"   Recall:     {metrics['recall']:.3f}")
    print(f"   F1 Score:   {metrics['f1_score']:.3f}")

    if metrics['cache_hit_rate_pct'] > 0:
        print(f"\n💾 Cache:")
        print(f"   Hit rate: {metrics['cache_hit_rate_pct']:.1f}%")
        print(f"   Hits:     {metrics['cache_hits']}")
        print(f"   Misses:   {metrics['cache_misses']}")

    # Save results
    output_file = f"realtime_benchmark_{args.approach}.json"
    results = {
        "approach": args.approach,
        "device": args.device,
        "threshold": args.threshold if args.approach == "cached" else None,
        "metrics": metrics,
        "cache_stats": cache_stats
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to: {output_file}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
