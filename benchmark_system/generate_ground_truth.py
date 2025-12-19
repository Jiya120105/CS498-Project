"""
Generate Ground Truth Labels

Runs VLM offline on all extracted tracks to establish ground truth labels.
This is NOT a system evaluation - just labeling for accuracy measurement.

Usage:
    python generate_ground_truth.py
"""

import os
import json
import time
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from tqdm import tqdm

os.environ["USE_LOCAL_SEMANTIC_CACHE"] = "0"

# Configuration
DATASET_DIR = Path("football_dataset")
QUERY = "Is this player wearing a yellow jersey? Answer Yes or No."
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("="*70)
print("GROUND TRUTH GENERATION")
print("="*70)
print(f"\nDevice: {DEVICE}")
print(f"Query: {QUERY}")

# Load dataset metadata
print(f"\n[Step 1/4] Loading dataset...")
metadata_file = DATASET_DIR / "dataset_metadata.json"
if not metadata_file.exists():
    print(f"   ✗ Dataset not found: {metadata_file}")
    print(f"   Please run create_mot16_dataset.py first")
    exit(1)

with open(metadata_file, 'r') as f:
    dataset_meta = json.load(f)

tracks = dataset_meta['tracks']
print(f"   ✓ Loaded {len(tracks)} tracks")
print(f"   Videos: {', '.join(dataset_meta['videos'])}")

# Load VLM model
print(f"\n[Step 2/4] Loading VLM model (SmolVLM-500M)...")
processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-500M-Instruct")
if hasattr(processor, "tokenizer"):
    processor.tokenizer.padding_side = "left"

if DEVICE == "cuda":
    model = AutoModelForVision2Seq.from_pretrained(
        "HuggingFaceTB/SmolVLM-500M-Instruct",
        torch_dtype=torch.float16,
        _attn_implementation="eager"
    ).to(DEVICE)
else:
    model = AutoModelForVision2Seq.from_pretrained(
        "HuggingFaceTB/SmolVLM-500M-Instruct",
        _attn_implementation="eager"
    ).to(DEVICE)

model.eval()
print(f"   ✓ Model loaded")

# Generate ground truth labels
print(f"\n[Step 3/4] Generating ground truth labels (offline, no time pressure)...")
print(f"   This will take ~{len(tracks) * 0.37 / 60:.1f} minutes...")

ground_truth = {}
latencies = []
label_distribution = {"Yes": 0, "No": 0, "Other": 0}

for track_info in tqdm(tracks, desc="   Labeling"):
    track_id = track_info['track_id']
    roi_path = DATASET_DIR / track_info['roi_path']

    if not roi_path.exists():
        print(f"   ⚠️  ROI not found: {roi_path}")
        continue

    # Load ROI
    roi = Image.open(roi_path).convert("RGB")

    # Prepare input
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": QUERY}]}]
    text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=text_prompt, images=[roi], return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # Run inference
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50)

    if DEVICE == "cuda":
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - start) * 1000
    latencies.append(latency_ms)

    # Parse answer
    answer = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    if "Assistant:" in answer:
        answer = answer.split("Assistant:")[-1].strip()

    # Classify label
    lower_ans = answer.lower()
    if "yes" in lower_ans:
        label = "Yes"
        label_distribution["Yes"] += 1
    elif "no" in lower_ans:
        label = "No"
        label_distribution["No"] += 1
    else:
        label = answer[:50]  # Keep first 50 chars
        label_distribution["Other"] += 1

    # Store ground truth
    ground_truth[track_id] = {
        "label": label,
        "raw_answer": answer,
        "latency_ms": latency_ms,
        "video": track_info['video'],
        "frame_id": track_info['frame_id']
    }

# Calculate statistics
mean_latency = np.mean(latencies)
median_latency = np.median(latencies)
std_latency = np.std(latencies)
p95_latency = np.percentile(latencies, 95)

print(f"\n{'='*70}")
print("GROUND TRUTH STATISTICS")
print(f"{'='*70}")

print(f"\n📊 Label Distribution:")
for label, count in label_distribution.items():
    pct = count / len(tracks) * 100
    print(f"   {label}: {count} ({pct:.1f}%)")

print(f"\n⏱️  Latency Statistics:")
print(f"   Mean:   {mean_latency:.0f} ms")
print(f"   Median: {median_latency:.0f} ms")
print(f"   Std:    {std_latency:.0f} ms")
print(f"   P95:    {p95_latency:.0f} ms")
print(f"   Min:    {np.min(latencies):.0f} ms")
print(f"   Max:    {np.max(latencies):.0f} ms")

# Save ground truth
print(f"\n[Step 4/4] Saving ground truth labels...")
output_file = DATASET_DIR / "ground_truth.json"
with open(output_file, 'w') as f:
    json.dump({
        "query": QUERY,
        "num_tracks": len(ground_truth),
        "label_distribution": label_distribution,
        "latency_stats": {
            "mean_ms": float(mean_latency),
            "median_ms": float(median_latency),
            "std_ms": float(std_latency),
            "p95_ms": float(p95_latency),
            "min_ms": float(np.min(latencies)),
            "max_ms": float(np.max(latencies))
        },
        "tracks": ground_truth
    }, f, indent=2)

print(f"   ✓ Saved to: {output_file}")

print(f"\n{'='*70}")
print("GROUND TRUTH ESTABLISHED")
print(f"{'='*70}")
print(f"\n📁 Output: {output_file}")
print(f"\n✅ Ground truth labels for {len(ground_truth)} tracks")
print(f"   This will be used as the 'answer key' for accuracy measurements")
print(f"\n✅ Next step: Create optimized VLM implementations and run benchmarks")
print(f"{'='*70}\n")
