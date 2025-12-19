"""
Re-run validation to check layer importance stability across ROIs.
This time we'll track per-sample importance to compute correlations.
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
import torch
from ultralytics import YOLO
from transformers import AutoProcessor, AutoModelForVision2Seq
from scipy.stats import pearsonr

os.environ["USE_LOCAL_SEMANTIC_CACHE"] = "0"

print("Loading models...")
yolo = YOLO('yolov8n.pt')
processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-500M-Instruct")
if hasattr(processor, "tokenizer"):
    processor.tokenizer.padding_side = "left"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

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

# Load ROIs
print("Loading MOT16 frames...")
img_dir = Path("MOT16/train/MOT16-04/img1")
frame_files = sorted(list(img_dir.glob("*.jpg")))[:10]

rois = []
for frame_file in frame_files:
    frame = cv2.imread(str(frame_file))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = yolo.predict(frame, verbose=False)

    if len(results[0].boxes) == 0:
        continue

    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()

    for i, cls_id in enumerate(classes):
        if cls_id == 0 and len(rois) < 8:  # Get 8 person ROIs
            x1, y1, x2, y2 = map(int, boxes[i])
            roi = frame_rgb[y1:y2, x1:x2]
            if roi.size > 0:
                rois.append(Image.fromarray(roi))

print(f"Extracted {len(rois)} ROIs")

# Hook setup
activation_samples = []  # List of dict, one per ROI

def make_hook(storage, name):
    def hook(module, input, output):
        if isinstance(output, tuple):
            act = output[0]
        else:
            act = output
        if isinstance(act, torch.Tensor):
            storage[name] = act.detach().abs().mean().item()
    return hook

# Register hooks
hooks = []
text_layers = model.model.text_model.layers
vision_layers = model.model.vision_model.encoder.layers

for i, layer in enumerate(vision_layers):
    hooks.append(layer.register_forward_hook(make_hook({}, f"vision_{i}")))
for i, layer in enumerate(text_layers):
    hooks.append(layer.register_forward_hook(make_hook({}, f"text_{i}")))

print(f"Registered {len(hooks)} hooks")

# Run inference on each ROI
query = "Is this person with a backpack? Answer Yes or No."
print(f"\\nRunning inference on {len(rois)} ROIs...")

for idx, roi in enumerate(rois):
    print(f"  ROI {idx+1}/{len(rois)}...", end=" ")

    storage = {}

    # Re-register hooks with this sample's storage
    for hook in hooks:
        hook.remove()
    hooks = []
    for i, layer in enumerate(vision_layers):
        hooks.append(layer.register_forward_hook(make_hook(storage, f"vision_{i}")))
    for i, layer in enumerate(text_layers):
        hooks.append(layer.register_forward_hook(make_hook(storage, f"text_{i}")))

    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": query}]}]
    text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=text_prompt, images=[roi], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50)

    answer = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    if "Assistant:" in answer:
        answer = answer.split("Assistant:")[-1].strip()

    activation_samples.append(dict(storage))
    print(f"✓ ({len(storage)} layers captured)")

# Remove hooks
for hook in hooks:
    hook.remove()

print(f"\\nCollected {len(activation_samples)} samples")

# Compute pairwise correlations
if len(activation_samples) >= 2:
    layer_names = sorted(activation_samples[0].keys())
    importance_matrix = np.array([
        [sample.get(layer, 0) for layer in layer_names]
        for sample in activation_samples
    ])

    print(f"\\n{'='*60}")
    print("STABILITY ANALYSIS")
    print(f"{'='*60}")

    correlations = []
    for i in range(len(importance_matrix)):
        for j in range(i+1, len(importance_matrix)):
            corr, pval = pearsonr(importance_matrix[i], importance_matrix[j])
            if not np.isnan(corr):
                correlations.append(corr)

    mean_corr = np.mean(correlations)
    std_corr = np.std(correlations)

    print(f"\\nPairwise Correlations (n={len(correlations)}):")
    print(f"  Mean: {mean_corr:.3f}")
    print(f"  Std:  {std_corr:.3f}")
    print(f"  Min:  {np.min(correlations):.3f}")
    print(f"  Max:  {np.max(correlations):.3f}")

    print(f"\\n{'='*60}")
    print("VERDICT")
    print(f"{'='*60}")

    if mean_corr > 0.7:
        print("✅ HIGH STABILITY (correlation > 0.7)")
        print("   → Layer importance is CONSISTENT across different ROIs")
        print("   → Safe to compress layers identified in profiling phase")
        print("   → PROCEED WITH IMPLEMENTATION!")
    elif mean_corr > 0.4:
        print("⚠️  MODERATE STABILITY (correlation 0.4-0.7)")
        print("   → Some consistency but variable")
        print("   → May work with larger profiling budget (K=15-20)")
        print("   → Test with more diverse queries before implementing")
    else:
        print("❌ LOW STABILITY (correlation < 0.4)")
        print("   → Layer importance varies significantly across ROIs")
        print("   → This approach may not work reliably")
        print("   → Consider alternative: static compression or different metric")

    print(f"\\n{'='*60}\\n")
else:
    print("[Error] Not enough samples for correlation analysis")

print("Done!")
