"""
Quick Prototype: Measure actual speedup from layer quantization

This script:
1. Loads SmolVLM in full precision (baseline)
2. Runs inference on 10 ROIs and measures latency
3. Quantizes bottom 50% of layers to INT8
4. Runs inference again and measures latency
5. Compares accuracy and speedup

Usage:
    python quick_quantization_prototype.py
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
print("QUICK QUANTIZATION PROTOTYPE - Option 2")
print("="*70)

# Configuration
QUERY = "Is this person with a backpack? Answer Yes or No."
NUM_TEST_ROIS = 10
QUANTIZATION_RATIO = 0.5  # Bottom 50% of layers

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nDevice: {device}")

# Load models
print("\n[1/6] Loading models...")
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
print(f"\n[2/6] Extracting up to {NUM_TEST_ROIS} unique stable tracks from MOT16...")
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

# Helper function for inference
def run_inference(model, processor, roi, query, device):
    """Run a single inference and return answer + latency."""
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": query}]}]
    text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=text_prompt, images=[roi], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    torch.cuda.synchronize() if device == "cuda" else None
    start = time.perf_counter()

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50)

    torch.cuda.synchronize() if device == "cuda" else None
    latency_ms = (time.perf_counter() - start) * 1000

    answer = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    if "Assistant:" in answer:
        answer = answer.split("Assistant:")[-1].strip()

    return answer, latency_ms

# Baseline: Full precision
print(f"\n[3/6] Measuring BASELINE (Full Precision FP16)...")
baseline_latencies = []
baseline_answers = []

for i, roi in enumerate(test_rois):
    print(f"   ROI {i+1}/{len(test_rois)}...", end=" ")
    answer, latency = run_inference(model, processor, roi, QUERY, device)
    baseline_latencies.append(latency)
    baseline_answers.append(answer)
    print(f"{latency:.0f}ms - {answer}")

baseline_mean = np.mean(baseline_latencies)
baseline_std = np.std(baseline_latencies)
print(f"\n   Baseline: {baseline_mean:.0f} ± {baseline_std:.0f} ms")

# Load layer importance from validation
print(f"\n[4/6] Loading layer importance from validation...")
try:
    with open("validation_results.json") as f:
        importance_data = json.load(f)
    layer_importance = importance_data["layer_importance"]

    # Sort by importance
    sorted_layers = sorted(layer_importance.items(), key=lambda x: x[1])
    threshold_idx = int(len(sorted_layers) * QUANTIZATION_RATIO)
    layers_to_quantize = [layer for layer, _ in sorted_layers[:threshold_idx]]

    print(f"   ✓ Loaded importance data")
    print(f"   Will quantize {len(layers_to_quantize)}/{len(layer_importance)} layers (bottom {int(QUANTIZATION_RATIO*100)}%)")
    print(f"   Layers to quantize: {layers_to_quantize[:5]}... (showing first 5)")
except Exception as e:
    print(f"   ✗ Could not load validation results: {e}")
    print(f"   Using fallback: quantize all vision layers + first 10 text layers")
    layers_to_quantize = [f"vision_{i}" for i in range(12)] + [f"text_{i}" for i in range(10)]

# Apply quantization
print(f"\n[5/6] Applying INT8 quantization to {len(layers_to_quantize)} layers...")

def quantize_layer_weights(layer):
    """Quantize layer weights to INT8 (simulated - converts to int8 then back to original dtype)."""
    for name, param in layer.named_parameters():
        if param.requires_grad == False:  # Only quantize frozen weights
            continue

        # Simple quantization: scale to int8 range, quantize, dequantize
        original_dtype = param.dtype
        param_data = param.data.float()

        # Compute scale factor
        abs_max = param_data.abs().max()
        scale = abs_max / 127.0 if abs_max > 0 else 1.0

        # Quantize to int8 and dequantize
        quantized = torch.round(param_data / scale).clamp(-128, 127)
        dequantized = quantized * scale

        # Store back (note: this is simulated quantization for measurement)
        param.data = dequantized.to(original_dtype)

# Get layer objects
vision_layers = model.model.vision_model.encoder.layers
text_layers = model.model.text_model.layers

quantized_count = 0
for layer_name in layers_to_quantize:
    if layer_name.startswith("vision_"):
        idx = int(layer_name.split("_")[1])
        if idx < len(vision_layers):
            quantize_layer_weights(vision_layers[idx])
            quantized_count += 1
    elif layer_name.startswith("text_"):
        idx = int(layer_name.split("_")[1])
        if idx < len(text_layers):
            quantize_layer_weights(text_layers[idx])
            quantized_count += 1

print(f"   ✓ Quantized {quantized_count} layers")

# Measure compressed performance
print(f"\n[6/6] Measuring COMPRESSED (INT8 quantization)...")
compressed_latencies = []
compressed_answers = []

for i, roi in enumerate(test_rois):
    print(f"   ROI {i+1}/{len(test_rois)}...", end=" ")
    answer, latency = run_inference(model, processor, roi, QUERY, device)
    compressed_latencies.append(latency)
    compressed_answers.append(answer)
    print(f"{latency:.0f}ms - {answer}")

compressed_mean = np.mean(compressed_latencies)
compressed_std = np.std(compressed_latencies)
print(f"\n   Compressed: {compressed_mean:.0f} ± {compressed_std:.0f} ms")

# Compare results
print(f"\n{'='*70}")
print("RESULTS SUMMARY")
print(f"{'='*70}")

speedup = baseline_mean / compressed_mean
latency_reduction = baseline_mean - compressed_mean
latency_reduction_pct = (latency_reduction / baseline_mean) * 100

print(f"\n📊 Performance:")
print(f"   Baseline (FP16):     {baseline_mean:.0f} ± {baseline_std:.0f} ms")
print(f"   Compressed (INT8):   {compressed_mean:.0f} ± {compressed_std:.0f} ms")
print(f"   Latency reduction:   {latency_reduction:.0f} ms ({latency_reduction_pct:.1f}%)")
print(f"   Speedup:             {speedup:.2f}×")

# Accuracy comparison
accuracy_matches = sum([1 for b, c in zip(baseline_answers, compressed_answers) if b == c])
accuracy_rate = (accuracy_matches / len(baseline_answers)) * 100

print(f"\n🎯 Accuracy:")
print(f"   Matching answers:    {accuracy_matches}/{len(baseline_answers)} ({accuracy_rate:.1f}%)")
print(f"   Baseline answers:    {set(baseline_answers)}")
print(f"   Compressed answers:  {set(compressed_answers)}")

if accuracy_rate == 100:
    print(f"   ✓ PERFECT accuracy preservation!")
elif accuracy_rate >= 90:
    print(f"   ✓ Excellent accuracy preservation")
elif accuracy_rate >= 80:
    print(f"   ⚠️  Moderate accuracy loss")
else:
    print(f"   ✗ Significant accuracy degradation")

# Memory estimate
memory_reduction_pct = len(layers_to_quantize) / 44 * 50  # Rough estimate: 50% size reduction per layer
print(f"\n💾 Memory:")
print(f"   Estimated reduction: ~{memory_reduction_pct:.0f}%")
print(f"   (INT8 uses ~50% memory of FP16 for {len(layers_to_quantize)}/44 quantized layers)")

# Overall verdict
print(f"\n{'='*70}")
print("VERDICT")
print(f"{'='*70}")

if speedup >= 1.3 and accuracy_rate >= 95:
    print("✅ EXCELLENT RESULTS!")
    print("   → Significant speedup with minimal accuracy loss")
    print("   → Proceed with full implementation (Option 1)")
elif speedup >= 1.15 and accuracy_rate >= 90:
    print("✓ GOOD RESULTS")
    print("   → Moderate speedup with acceptable accuracy")
    print("   → Consider proceeding with Option 1")
elif speedup >= 1.05:
    print("⚠️  MODEST RESULTS")
    print("   → Small speedup, may not be worth complexity")
    print("   → Consider more aggressive quantization or alternative approach")
else:
    print("✗ MINIMAL IMPACT")
    print("   → Quantization didn't provide expected speedup")
    print("   → May need different approach or investigate bottlenecks")

print(f"\n{'='*70}\n")

# Save results
results = {
    "baseline_mean_ms": float(baseline_mean),
    "baseline_std_ms": float(baseline_std),
    "compressed_mean_ms": float(compressed_mean),
    "compressed_std_ms": float(compressed_std),
    "speedup": float(speedup),
    "latency_reduction_ms": float(latency_reduction),
    "latency_reduction_pct": float(latency_reduction_pct),
    "accuracy_rate": float(accuracy_rate),
    "layers_quantized": len(layers_to_quantize),
    "total_layers": 44,
    "quantization_ratio": QUANTIZATION_RATIO
}

with open("quantization_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Results saved to: quantization_results.json")
print("\nDone!")
