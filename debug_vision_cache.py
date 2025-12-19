"""
Debug Vision Cache - Visualize cache hits/misses

Saves images of cache hits and their matched pairs to verify correctness.
"""

import os
import time
import torch
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import cv2
from ultralytics import YOLO
from transformers import AutoProcessor, AutoModelForVision2Seq
from vision_cache.cached_vlm import CachedSmolVLM

os.environ["USE_LOCAL_SEMANTIC_CACHE"] = "0"

# Create debug output directory
debug_dir = Path("debug_cache_output")
debug_dir.mkdir(exist_ok=True)

print("="*70)
print("VISION CACHE DEBUG - Visualize Matches")
print("="*70)

QUERY = "Is this person with a backpack? Answer Yes or No."
NUM_FRAMES = 20
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\nDevice: {device}")
print(f"Debug output: {debug_dir}/")

# Load models
print("\n[1/3] Loading models...")
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

# Test different similarity thresholds
for sim_threshold in [0.99, 0.95, 0.90]:
    print(f"\n{'='*70}")
    print(f"Testing with similarity threshold: {sim_threshold}")
    print(f"{'='*70}")

    cached_vlm = CachedSmolVLM(base_model, processor, cache_size=50, similarity_threshold=sim_threshold, device=device)

    # Extract ROIs
    print(f"\n[2/3] Extracting ROIs from {NUM_FRAMES} frames...")
    img_dir = Path("MOT16/train/MOT16-04/img1")
    frame_files = sorted(list(img_dir.glob("*.jpg")))[:NUM_FRAMES]

    rois_data = []  # (image, track_id, frame_id)

    for frame_idx, frame_file in enumerate(frame_files):
        frame = cv2.imread(str(frame_file))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = yolo.track(frame, persist=True, verbose=False, tracker="bytetrack.yaml")

        if results[0].boxes.id is None or len(results[0].boxes) == 0:
            continue

        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        classes = results[0].boxes.cls.cpu().numpy()

        for i, cls_id in enumerate(classes):
            if cls_id == 0:  # Person
                track_id = int(track_ids[i])
                x1, y1, x2, y2 = map(int, boxes[i])
                roi = frame_rgb[y1:y2, x1:x2]
                if roi.size > 0:
                    rois_data.append((Image.fromarray(roi), track_id, frame_idx + 1))

    print(f"   ✓ Extracted {len(rois_data)} ROIs")

    # Process and save debug images
    print(f"\n[3/3] Processing and saving debug images...")

    cache_matches = []  # (query_idx, matched_idx, similarity, track_ids)
    hits = 0
    misses = 0

    for idx, (roi, track_id, frame_id) in enumerate(rois_data[:30]):  # First 30 for debugging
        # Compute embedding
        query_emb = cached_vlm._compute_image_embedding(roi)

        # Check cache manually to get similarity scores
        best_match_idx = None
        best_similarity = -1.0

        # Search through cache entries
        for cache_idx, (cache_key, cache_entry) in enumerate(cached_vlm.cache.cache.items()):
            sim = cached_vlm.cache._compute_similarity(query_emb, cache_entry.embedding)
            if sim > best_similarity:
                best_similarity = sim
                best_match_idx = cache_idx

        is_hit = best_similarity >= sim_threshold if best_match_idx is not None else False

        if is_hit:
            hits += 1
            # Save the match pair
            cache_matches.append((idx, best_match_idx, best_similarity, track_id))

            # Create side-by-side comparison
            if len(cache_matches) <= 10:  # Save first 10 matches
                fig_width = roi.width * 2 + 40
                fig_height = roi.height + 60

                canvas = Image.new('RGB', (fig_width, fig_height), 'white')
                draw = ImageDraw.Draw(canvas)

                # Draw query image
                canvas.paste(roi, (10, 40))
                draw.text((10, 10), f"Query: Track {track_id}, Frame {frame_id}", fill='black')

                # Draw matched image (find it in rois_data)
                if best_match_idx < len(rois_data):
                    matched_roi, matched_track, matched_frame = rois_data[best_match_idx]
                    canvas.paste(matched_roi.resize(roi.size), (roi.width + 30, 40))
                    draw.text((roi.width + 30, 10),
                             f"Match: Track {matched_track}, Frame {matched_frame}\nSim: {best_similarity:.3f}",
                             fill='blue')

                # Save
                canvas.save(debug_dir / f"threshold_{sim_threshold}_match_{len(cache_matches)}.png")
        else:
            misses += 1

        # Run inference and add to cache
        result = cached_vlm.infer(roi, QUERY, track_id=track_id)

        if idx % 10 == 0:
            print(f"   Processed {idx+1}/{min(30, len(rois_data))} ROIs - Hits: {hits}, Misses: {misses}")

    hit_rate = hits / (hits + misses) * 100 if (hits + misses) > 0 else 0
    print(f"\n   Results for threshold {sim_threshold}:")
    print(f"   Hit rate: {hit_rate:.1f}%")
    print(f"   Hits: {hits}, Misses: {misses}")
    print(f"   Saved {min(10, len(cache_matches))} match visualizations to {debug_dir}/")

    # Clear cache for next threshold test
    cached_vlm.clear_cache()

print(f"\n{'='*70}")
print("DEBUG COMPLETE")
print(f"{'='*70}")
print(f"\nCheck {debug_dir}/ for visualizations")
print("Look for:")
print("  - Do visually similar images have high similarity scores?")
print("  - Are matches between same track IDs?")
print("  - Is 0.95 threshold too loose (matching different people)?")
print("\nRecommendation:")
print("  - If matches look wrong: Increase threshold to 0.98-0.99")
print("  - If matches look right: Current setup is working correctly!")
