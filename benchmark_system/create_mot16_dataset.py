"""
Create MOT16 Dataset for Comprehensive Evaluation

Extracts 500-800 unique stable tracks from multiple MOT16 videos.
Saves ROIs as images + metadata for evaluation.

Usage:
    python create_mot16_dataset.py
"""

import os
import cv2
import json
import numpy as np
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
from tqdm import tqdm

# Configuration
VIDEOS = [
    "MOT16/train/MOT16-04",  # ~83 unique tracks over 1050 frames
    "MOT16/train/MOT16-02",  # ~62 unique tracks over 600 frames
]
OUTPUT_DIR = Path("mot16_dataset")
STABLE_MIN_FRAMES = 5
TARGET_TRACKS = 500  # Stop early if we reach this

print("="*70)
print("MOT16 DATASET CREATION FOR COMPREHENSIVE EVALUATION")
print("="*70)

# Create output directories
OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "rois").mkdir(exist_ok=True)

# Load YOLO
print("\n[Step 1/3] Loading YOLO model...")
yolo = YOLO('yolov8n.pt')
print("   ✓ YOLO loaded")

# Dataset storage
dataset = []  # List of track metadata
track_counter = 0

print(f"\n[Step 2/3] Extracting unique stable tracks from {len(VIDEOS)} videos...")
print(f"   Target: {TARGET_TRACKS}+ unique tracks")
print(f"   Stability requirement: {STABLE_MIN_FRAMES} consecutive frames\n")

for video_path in VIDEOS:
    video_name = Path(video_path).name
    print(f"\n{'='*70}")
    print(f"Processing: {video_name}")
    print(f"{'='*70}")

    # Check if video directory exists
    img_dir = Path(video_path) / "img1"
    if not img_dir.exists():
        print(f"   ✗ Directory not found: {img_dir}")
        continue

    frame_files = sorted(list(img_dir.glob("*.jpg")))
    print(f"   Total frames: {len(frame_files)}")

    # Track stability state
    consecutive_seen = {}
    last_seen_frame = {}
    evaluated_tracks = set()
    tracks_from_video = 0

    for frame_idx, frame_file in enumerate(tqdm(frame_files, desc=f"   {video_name}")):
        if track_counter >= TARGET_TRACKS:
            print(f"   ✓ Reached target of {TARGET_TRACKS} tracks, stopping early")
            break

        frame = cv2.imread(str(frame_file))
        if frame is None:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Use tracking to get consistent track IDs
        results = yolo.track(frame, persist=True, verbose=False, tracker="bytetrack.yaml")

        if results[0].boxes.id is None or len(results[0].boxes) == 0:
            continue

        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        classes = results[0].boxes.cls.cpu().numpy()

        for i, cls_id in enumerate(classes):
            if cls_id != 0:  # Not a person
                continue

            # Create unique track ID across videos
            local_track_id = int(track_ids[i])
            global_track_id = f"{video_name}_track_{local_track_id}"

            # Update stability counters
            prev = last_seen_frame.get(global_track_id)
            if prev is not None and prev == frame_idx:
                consecutive_seen[global_track_id] = consecutive_seen.get(global_track_id, 0) + 1
            else:
                consecutive_seen[global_track_id] = 1
            last_seen_frame[global_track_id] = frame_idx + 1

            # Extract ROI at FIRST STABLE APPEARANCE
            if (global_track_id not in evaluated_tracks and
                consecutive_seen.get(global_track_id, 0) >= STABLE_MIN_FRAMES):

                x1, y1, x2, y2 = map(int, boxes[i])
                roi = frame_rgb[y1:y2, x1:x2]

                if roi.size == 0:
                    continue

                # Save ROI image
                roi_filename = f"{global_track_id}.jpg"
                roi_path = OUTPUT_DIR / "rois" / roi_filename
                Image.fromarray(roi).save(roi_path)

                # Store metadata
                dataset.append({
                    "track_id": global_track_id,
                    "video": video_name,
                    "frame_id": frame_idx + 1,
                    "frame_file": frame_file.name,
                    "bbox": [x1, y1, x2, y2],
                    "bbox_width": x2 - x1,
                    "bbox_height": y2 - y1,
                    "roi_path": str(roi_path.relative_to(OUTPUT_DIR)),
                    "stable_frames": consecutive_seen[global_track_id]
                })

                evaluated_tracks.add(global_track_id)
                tracks_from_video += 1
                track_counter += 1

                if track_counter >= TARGET_TRACKS:
                    break

    print(f"   ✓ Extracted {tracks_from_video} unique stable tracks from {video_name}")

    if track_counter >= TARGET_TRACKS:
        break

print(f"\n{'='*70}")
print(f"EXTRACTION COMPLETE")
print(f"{'='*70}")
print(f"   Total unique tracks: {track_counter}")
print(f"   Videos processed: {len([v for v in VIDEOS if any(d['video'] == Path(v).name for d in dataset)])}")

# Calculate statistics
video_distribution = {}
for track in dataset:
    video = track['video']
    video_distribution[video] = video_distribution.get(video, 0) + 1

print(f"\n   Distribution by video:")
for video, count in video_distribution.items():
    print(f"     {video}: {count} tracks ({count/track_counter*100:.1f}%)")

bbox_sizes = [(t['bbox_width'], t['bbox_height']) for t in dataset]
avg_width = np.mean([w for w, h in bbox_sizes])
avg_height = np.mean([h for w, h in bbox_sizes])
print(f"\n   Average ROI size: {avg_width:.0f} × {avg_height:.0f} pixels")

# Save metadata
print(f"\n[Step 3/3] Saving dataset metadata...")
metadata_file = OUTPUT_DIR / "dataset_metadata.json"
with open(metadata_file, 'w') as f:
    json.dump({
        "num_tracks": len(dataset),
        "videos": list(video_distribution.keys()),
        "video_distribution": video_distribution,
        "stable_min_frames": STABLE_MIN_FRAMES,
        "avg_bbox_width": float(avg_width),
        "avg_bbox_height": float(avg_height),
        "tracks": dataset
    }, f, indent=2)

print(f"   ✓ Saved to: {metadata_file}")

print(f"\n{'='*70}")
print("DATASET READY FOR EVALUATION")
print(f"{'='*70}")
print(f"\n📁 Output:")
print(f"   ROIs: {OUTPUT_DIR / 'rois'}/ ({len(dataset)} images)")
print(f"   Metadata: {metadata_file}")
print(f"\n✅ Next step: Run generate_ground_truth.py to establish ground truth labels")
print(f"{'='*70}\n")
