"""
Aggressive Compression Prototype

Tests more aggressive compression strategies:
1. Structured pruning (remove attention heads from unimportant layers)
2. Profile to find bottlenecks
3. More aggressive quantization (75% of layers)

Usage:
    python aggressive_compression_prototype.py
"""

import os
import time
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO
from transformers import AutoProcessor, AutoModelForVision2Seq
import json

os.environ["USE_LOCAL_SEMANTIC_CACHE"] = "0"

print("="*70)
print("AGGRESSIVE COMPRESSION PROTOTYPE")
print("="*70)

# Configuration
QUERY = "Is this person with a backpack? Answer Yes or No."
NUM_TEST_ROIS = 10

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nDevice: {device}")

# Load models
print("\n[Step 1] Loading models...")
yolo = YOLO('yolov8n.pt')
processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-500M-Instruct")
if hasattr(processor, "tokenizer"):
    processor.tokenizer.padding_side = "left"

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
print("   ✓ Models loaded")

# Extract ROIs - ONLY FIRST STABLE APPEARANCE OF UNIQUE TRACKS
print(f"\n[Step 2] Extracting up to {NUM_TEST_ROIS} unique stable tracks from MOT16...")
print(f"   (Using tracking with stable_min_frames=5 to match actual system)")
img_dir = Path("MOT16/train/MOT16-04/img1")
frame_files = sorted(list(img_dir.glob("*.jpg")))

# Track stability across frames (matching run_system.py logic)
consecutive_seen = {}
last_seen_frame = {}
evaluated_tracks = set()
test_rois = []

for frame_idx, frame_file in enumerate(frame_files):
    if len(test_rois) >= NUM_TEST_ROIS:
        break

    frame = cv2.imread(str(frame_file))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Use tracking (not just detection) to match run_system.py
    results = yolo.track(frame, persist=True, verbose=False, tracker="bytetrack.yaml")

    if results[0].boxes.id is None or len(results[0].boxes) == 0:
        continue

    boxes = results[0].boxes.xyxy.cpu().numpy()
    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
    classes = results[0].boxes.cls.cpu().numpy()

    for i, cls_id in enumerate(classes):
        if cls_id != 0 or len(test_rois) >= NUM_TEST_ROIS:
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
                test_rois.append(Image.fromarray(roi))
                evaluated_tracks.add(track_id)

print(f"   ✓ Extracted {len(test_rois)} unique stable tracks (no duplicates)")

# Inference function
def run_inference(model, processor, roi, query, device):
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": query}]}]
    text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=text_prompt, images=[roi], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50)

    if device == "cuda":
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - start) * 1000

    answer = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    if "Assistant:" in answer:
        answer = answer.split("Assistant:")[-1].strip()

    return answer, latency_ms

# Baseline
print(f"\n[Step 3] Measuring BASELINE...")
baseline_latencies = []
baseline_answers = []

for i, roi in enumerate(test_rois[:5]):  # Use 5 for faster testing
    answer, latency = run_inference(model, processor, roi, QUERY, device)
    baseline_latencies.append(latency)
    baseline_answers.append(answer)
    print(f"   ROI {i+1}/5: {latency:.0f}ms - {answer}")

baseline_mean = np.mean(baseline_latencies)
print(f"\n   Baseline: {baseline_mean:.0f} ms")

# Load importance
print(f"\n[Step 4] Loading layer importance...")
try:
    with open("validation_results.json") as f:
        importance_data = json.load(f)
    layer_importance = importance_data["layer_importance"]
    sorted_layers = sorted(layer_importance.items(), key=lambda x: x[1])
    print(f"   ✓ Loaded importance data")
except Exception as e:
    print(f"   ✗ Could not load: {e}")
    sorted_layers = []

# Strategy 1: Prune attention heads in unimportant layers
print(f"\n[Step 5] STRATEGY 1: Structured Pruning (Attention Heads)")
print("   Removing 50% of attention heads from bottom 25% of layers...")

# Identify bottom 25% layers
threshold_idx = int(len(sorted_layers) * 0.25)
prune_layers = [layer for layer, _ in sorted_layers[:threshold_idx]]

vision_layers = model.model.vision_model.encoder.layers
text_layers = model.model.text_model.layers

pruned_count = 0
for layer_name in prune_layers:
    if layer_name.startswith("vision_"):
        idx = int(layer_name.split("_")[1])
        if idx < len(vision_layers):
            layer = vision_layers[idx]
            # Prune attention by zeroing out half of attention heads
            if hasattr(layer, 'self_attn'):
                attn = layer.self_attn
                if hasattr(attn, 'q_proj'):
                    # Zero out half the weights (simulated head pruning)
                    with torch.no_grad():
                        q_weight = attn.q_proj.weight
                        k_weight = attn.k_proj.weight
                        v_weight = attn.v_proj.weight

                        # Prune second half of heads
                        half = q_weight.shape[0] // 2
                        q_weight[half:, :] = 0
                        k_weight[half:, :] = 0
                        v_weight[half:, :] = 0
                    pruned_count += 1

    elif layer_name.startswith("text_"):
        idx = int(layer_name.split("_")[1])
        if idx < len(text_layers):
            layer = text_layers[idx]
            if hasattr(layer, 'self_attn'):
                attn = layer.self_attn
                if hasattr(attn, 'q_proj'):
                    with torch.no_grad():
                        q_weight = attn.q_proj.weight
                        k_weight = attn.k_proj.weight
                        v_weight = attn.v_proj.weight

                        half = q_weight.shape[0] // 2
                        q_weight[half:, :] = 0
                        k_weight[half:, :] = 0
                        v_weight[half:, :] = 0
                    pruned_count += 1

print(f"   ✓ Pruned attention in {pruned_count} layers")

# Measure pruned performance
print(f"\n[Step 6] Measuring PRUNED performance...")
pruned_latencies = []
pruned_answers = []

for i, roi in enumerate(test_rois[:5]):
    answer, latency = run_inference(model, processor, roi, QUERY, device)
    pruned_latencies.append(latency)
    pruned_answers.append(answer)
    print(f"   ROI {i+1}/5: {latency:.0f}ms - {answer}")

pruned_mean = np.mean(pruned_latencies)
pruned_speedup = baseline_mean / pruned_mean
print(f"\n   Pruned: {pruned_mean:.0f} ms")
print(f"   Speedup: {pruned_speedup:.2f}×")

# Check accuracy
pruned_accuracy = sum([1 for b, p in zip(baseline_answers, pruned_answers) if b == p]) / len(baseline_answers) * 100
print(f"   Accuracy: {pruned_accuracy:.0f}%")

# Summary
print(f"\n{'='*70}")
print("RESULTS SUMMARY")
print(f"{'='*70}")

print(f"\n📊 Performance:")
print(f"   Baseline:            {baseline_mean:.0f} ms")
print(f"   Structured Pruning:  {pruned_mean:.0f} ms ({pruned_speedup:.2f}× speedup)")
print(f"   Latency reduction:   {baseline_mean - pruned_mean:.0f} ms")

print(f"\n🎯 Accuracy:")
print(f"   Pruning accuracy:    {pruned_accuracy:.0f}%")

print(f"\n💡 Analysis:")
if pruned_speedup > 1.15:
    print(f"   ✓ Structured pruning shows promise!")
    print(f"   → {pruned_speedup:.2f}× speedup with attention head pruning")
elif pruned_speedup > 1.05:
    print(f"   ~ Modest improvement from pruning")
    print(f"   → May need more aggressive pruning or different approach")
else:
    print(f"   ✗ Pruning didn't provide expected speedup")
    print(f"   → Bottleneck likely elsewhere (vision encoder, memory, etc.)")

# Profiling insight
print(f"\n🔍 Bottleneck Analysis:")
print(f"   Baseline variance: {np.std(baseline_latencies):.0f}ms (high variance suggests caching effects)")
print(f"   First inference: {baseline_latencies[0]:.0f}ms (often slower due to warmup)")
print(f"   Avg of rest: {np.mean(baseline_latencies[1:]):.0f}ms")

if baseline_latencies[0] > baseline_mean * 1.5:
    print(f"   → First inference is much slower (warmup overhead)")
if np.std(baseline_latencies) > 50:
    print(f"   → High variance suggests GPU scheduling or memory effects")

print(f"\n💡 Recommendations:")
print(f"   1. True INT8 quantization (using torch.quantization or bitsandbytes) may give better results")
print(f"   2. Profile to find actual bottleneck (vision encoder vs text decoder vs memory)")
print(f"   3. Consider caching vision embeddings instead (avoid recomputing vision encoder)")
print(f"   4. For real speedup, may need model distillation or smaller base model")

print(f"\n{'='*70}\n")

# Save results
results = {
    "baseline_mean_ms": float(baseline_mean),
    "pruned_mean_ms": float(pruned_mean),
    "pruned_speedup": float(pruned_speedup),
    "pruned_accuracy": float(pruned_accuracy),
    "layers_pruned": pruned_count,
}

with open("aggressive_compression_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Results saved to: aggressive_compression_results.json")
print("\nDone!")
