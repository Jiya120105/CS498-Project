"""
Vision Cache Accuracy Benchmark

Tests both speedup AND accuracy preservation by:
1. Running baseline inference on 300 frames to establish ground truth
2. Running cached inference on the same frames
3. Comparing: speedup, accuracy, cache hit rate, background corrections
"""

import os
import time
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
from ultralytics import YOLO
from transformers import AutoProcessor, AutoModelForVision2Seq
from vision_cache.cached_vlm import CachedSmolVLM
import json
from tqdm import tqdm

os.environ["USE_LOCAL_SEMANTIC_CACHE"] = "0"

print("="*70)
print("VISION CACHE ACCURACY & SPEEDUP BENCHMARK")
print("="*70)

# Configuration
QUERY = "Is this person with a backpack? Answer Yes or No."
NUM_FRAMES = 100  # Use 100 frames (will extract ~2000-3000 ROIs)
SIMILARITY_THRESHOLD = 0.98
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\nDevice: {device}")
print(f"Frames to process: {NUM_FRAMES}")
print(f"Similarity threshold: {SIMILARITY_THRESHOLD}")

# Load models
print("\n[Step 1/5] Loading models...")
yolo = YOLO('yolov8n.pt')
processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-500M-Instruct")
if hasattr(processor, "tokenizer"):
    processor.tokenizer.padding_side = "left"

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
print("   ✓ Models loaded")

# Extract ROIs with tracking - ONLY FIRST STABLE APPEARANCE
print(f"\n[Step 2/5] Extracting first stable appearance of unique tracks from {NUM_FRAMES} frames...")
print(f"   (Matching run_system.py behavior: stable_min_frames=5)")
img_dir = Path("MOT16/train/MOT16-04/img1")
frame_files = sorted(list(img_dir.glob("*.jpg")))[:NUM_FRAMES]

# Track stability across frames (matching run_system.py logic)
consecutive_seen = {}
last_seen_frame = {}
evaluated_tracks = set()
rois_data = []  # (image, track_id, frame_id)

for frame_idx, frame_file in enumerate(tqdm(frame_files, desc="Extracting ROIs")):
    frame = cv2.imread(str(frame_file))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = yolo.track(frame, persist=True, verbose=False, tracker="bytetrack.yaml")

    if results[0].boxes.id is None or len(results[0].boxes) == 0:
        continue

    boxes = results[0].boxes.xyxy.cpu().numpy()
    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
    classes = results[0].boxes.cls.cpu().numpy()

    for i, cls_id in enumerate(classes):
        if cls_id != 0:  # Not a person
            continue

        track_id = int(track_ids[i])

        # Update stability counters (matching run_system.py:154-160)
        prev = last_seen_frame.get(track_id)
        if prev is not None and prev == frame_idx:
            consecutive_seen[track_id] = consecutive_seen.get(track_id, 0) + 1
        else:
            consecutive_seen[track_id] = 1
        last_seen_frame[track_id] = frame_idx + 1

        # Only extract ROI at FIRST STABLE APPEARANCE (matching run_system.py:163)
        if (track_id not in evaluated_tracks and
            consecutive_seen.get(track_id, 0) >= 5):  # stable_min_frames = 5

            x1, y1, x2, y2 = map(int, boxes[i])
            roi = frame_rgb[y1:y2, x1:x2]
            if roi.size > 0:
                rois_data.append((Image.fromarray(roi), track_id, frame_idx + 1))
                evaluated_tracks.add(track_id)

print(f"   ✓ Extracted {len(rois_data)} unique stable tracks from {NUM_FRAMES} frames")
print(f"   (Only first stable appearance per track, matching actual system behavior)")

# Baseline: Establish ground truth
print(f"\n[Step 3/5] Establishing ground truth (baseline inference on all ROIs)...")
print("   This will take a while but establishes correct answers...")

ground_truth = {}  # roi_idx -> (label, confidence, latency_ms)
baseline_latencies = []

for idx, (roi, track_id, frame_id) in enumerate(tqdm(rois_data, desc="Ground truth")):
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": QUERY}]}]
    text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=text_prompt, images=[roi], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.no_grad():
        outputs = base_model.generate(**inputs, max_new_tokens=50)

    if device == "cuda":
        torch.cuda.synchronize()
    latency = (time.perf_counter() - start) * 1000

    answer = processor.batch_decode(outputs, skip_special_tokens=True)[0]
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

    ground_truth[idx] = (label, latency)
    baseline_latencies.append(latency)

baseline_mean = np.mean(baseline_latencies)
baseline_median = np.median(baseline_latencies)

print(f"\n   Ground truth established:")
print(f"   Mean latency: {baseline_mean:.0f} ms")
print(f"   Median latency: {baseline_median:.0f} ms")

# Save ground truth
with open("ground_truth_answers.json", "w") as f:
    gt_serializable = {str(k): {"label": v[0], "latency_ms": v[1]} for k, v in ground_truth.items()}
    json.dump(gt_serializable, f, indent=2)
print(f"   ✓ Ground truth saved to ground_truth_answers.json")

# Cached inference - Test embedding similarity matching
print(f"\n[Step 4/5] Running cached inference with similarity matching...")
print(f"   Testing if visually similar tracks can skip VLM via embedding cache...")
cached_vlm = CachedSmolVLM(
    base_model, processor,
    cache_size=100,
    similarity_threshold=SIMILARITY_THRESHOLD,
    device=device,
    enable_background_validation=True
)

cached_results = {}  # roi_idx -> (label, confidence, latency_ms, cache_hit, from_similarity)
cached_latencies = []
similarity_hits = 0  # Tracks matched via similarity (not answer cache)
vlm_calls = 0  # Actual VLM calls needed

for idx, (roi, track_id, frame_id) in enumerate(tqdm(rois_data, desc="Cached inference")):
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    result = cached_vlm.infer(roi, QUERY, track_id=track_id)

    if device == "cuda":
        torch.cuda.synchronize()
    latency = (time.perf_counter() - start) * 1000

    # Track whether this was similarity-based hit or VLM call
    cache_hit = result['metadata']['cache_hit']
    from_answer_cache = result['metadata'].get('from_answer_cache', False)

    if cache_hit and not from_answer_cache:
        similarity_hits += 1  # Matched via embedding similarity
    if not cache_hit:
        vlm_calls += 1  # Actual VLM call

    cached_results[idx] = (result['label'], result['confidence'], latency, cache_hit)
    cached_latencies.append(latency)

# Ensure background validation is processed
if cached_vlm.validation_queue:
    cached_vlm._process_validation_queue()

cached_mean = np.mean(cached_latencies)
cached_median = np.median(cached_latencies)
total_unique_tracks = len(rois_data)
similarity_reduction_pct = (similarity_hits / total_unique_tracks * 100) if total_unique_tracks > 0 else 0

print(f"\n   Cached inference complete:")
print(f"   Total unique tracks: {total_unique_tracks}")
print(f"   VLM calls needed: {vlm_calls}")
print(f"   Similarity matches: {similarity_hits} ({similarity_reduction_pct:.1f}% avoided VLM)")
print(f"   Mean latency: {cached_mean:.0f} ms")
print(f"   Median latency: {cached_median:.0f} ms")

# Accuracy comparison
print(f"\n[Step 5/5] Comparing accuracy against ground truth...")

matches = 0
mismatches = []

for idx in ground_truth.keys():
    gt_label = ground_truth[idx][0]
    cached_label = cached_results[idx][0]

    if gt_label == cached_label:
        matches += 1
    else:
        mismatches.append((idx, gt_label, cached_label, rois_data[idx][1]))  # idx, gt, cached, track_id

accuracy = matches / len(ground_truth) * 100

print(f"\n   Accuracy: {accuracy:.2f}% ({matches}/{len(ground_truth)})")
print(f"   Mismatches: {len(mismatches)}")

if mismatches[:10]:
    print(f"\n   First 10 mismatches:")
    for idx, gt, cached, track_id in mismatches[:10]:
        print(f"     ROI {idx} (Track {track_id}): GT='{gt}' vs Cached='{cached}'")

# Final results
print(f"\n{'='*70}")
print("FINAL RESULTS")
print(f"{'='*70}")

# Calculate effective speedup from avoiding VLM calls
baseline_total_time = baseline_mean * total_unique_tracks
cached_total_time = cached_mean * total_unique_tracks
effective_vlm_reduction = (total_unique_tracks - vlm_calls) / total_unique_tracks if total_unique_tracks > 0 else 0

print(f"\n📊 Baseline System Performance:")
print(f"   Unique tracks:        {total_unique_tracks}")
print(f"   VLM calls (baseline): {total_unique_tracks} (1 per track)")
print(f"   Latency per call:     {baseline_mean:.0f} ms")
print(f"   Total time:           {baseline_total_time/1000:.1f}s")

print(f"\n📊 Embedding Cache Performance:")
print(f"   VLM calls needed:     {vlm_calls}")
print(f"   Similarity matches:   {similarity_hits}")
print(f"   VLM call reduction:   {effective_vlm_reduction*100:.1f}%")
print(f"   Latency per call:     {cached_mean:.0f} ms")
print(f"   Total time:           {cached_total_time/1000:.1f}s")
print(f"   Effective speedup:    {baseline_total_time/cached_total_time:.2f}×")

print(f"\n🎯 Accuracy:")
print(f"   Accuracy vs GT:    {accuracy:.2f}%")
print(f"   Matches:           {matches}/{len(ground_truth)}")
print(f"   Mismatches:        {len(mismatches)}")

cache_stats = cached_vlm.get_cache_stats()
print(f"\n📈 Cache Statistics:")
print(f"   Similarity threshold: {SIMILARITY_THRESHOLD}")
print(f"   Cache size:           {cache_stats['cache_size']}/{cache_stats['max_size']}")
print(f"   Bg validations:       {cache_stats['background_validations']}")
print(f"   Bg corrections:       {cache_stats['cache_corrections']}")
if cache_stats['background_validations'] > 0:
    print(f"   Cache accuracy:       {cache_stats['cache_accuracy']:.1f}%")

print(f"\n💡 Analysis:")
if accuracy >= 98:
    print(f"   ✅ EXCELLENT accuracy preservation ({accuracy:.1f}%)")
elif accuracy >= 95:
    print(f"   ✓ GOOD accuracy preservation ({accuracy:.1f}%)")
elif accuracy >= 90:
    print(f"   ~ ACCEPTABLE accuracy ({accuracy:.1f}%)")
else:
    print(f"   ✗ LOW accuracy ({accuracy:.1f}%) - needs improvement")

if similarity_hits > 0:
    print(f"   ✅ Embedding cache working: {similarity_hits} tracks matched via similarity")
    print(f"   → Reduced VLM calls from {total_unique_tracks} to {vlm_calls} ({effective_vlm_reduction*100:.0f}% reduction)")
else:
    print(f"   ⚠️  No similarity matches found")
    print(f"   → Try lowering similarity threshold (current: {SIMILARITY_THRESHOLD})")
    print(f"   → Or tracks may be too diverse for similarity matching")

if cache_stats['background_validations'] > 0 and cache_stats['cache_accuracy'] >= 95:
    print(f"   ✅ Background validation working well ({cache_stats['cache_accuracy']:.1f}% accurate)")
elif cache_stats['cache_corrections'] > 0:
    print(f"   ⚠️  Background validation found issues ({cache_stats['cache_corrections']} corrections)")

print(f"\n🎓 Research Value:")
print(f"   - Embedding cache reduces VLM calls by {effective_vlm_reduction*100:.0f}% via similarity matching")
print(f"   - Demonstrates feasibility of cross-track similarity for VLM optimization")
print(f"   - {accuracy:.1f}% accuracy preservation shows robustness of approach")
if cache_stats['cache_corrections'] > 0:
    print(f"   - Background validation self-corrects {cache_stats['cache_corrections']} errors")

# Save results
results_summary = {
    "num_unique_tracks": len(rois_data),
    "num_frames": NUM_FRAMES,
    "baseline_mean_ms": float(baseline_mean),
    "cached_mean_ms": float(cached_mean),
    "vlm_calls_needed": vlm_calls,
    "similarity_matches": similarity_hits,
    "vlm_call_reduction_pct": float(effective_vlm_reduction * 100),
    "effective_speedup": float(baseline_total_time / cached_total_time),
    "accuracy_pct": float(accuracy),
    "mismatches": len(mismatches),
    "similarity_threshold": SIMILARITY_THRESHOLD,
    "cache_stats": cache_stats
}

with open("vision_cache_results.json", "w") as f:
    json.dump(results_summary, f, indent=2)

print(f"\n{'='*70}")
print(f"Results saved to vision_cache_results.json")
print(f"Ground truth saved to ground_truth_answers.json")
print(f"{'='*70}\n")
