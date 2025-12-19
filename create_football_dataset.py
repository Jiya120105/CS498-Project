"""
Create Football Dataset

Extracts unique stable tracks from the football video for evaluation.
Similar to create_mot16_dataset.py but for a single video file.

Process:
1. Run YOLO tracking on football_video.mp4
2. Identify stable tracks (seen >= 5 frames)
3. Extract the first stable ROI for each track
4. Save metadata and images to football_dataset/
"""

import cv2
import json
import shutil
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
from PIL import Image

# Configuration
VIDEO_PATH = "football_video.mp4"
OUTPUT_DIR = Path("football_dataset")
ROIS_DIR = OUTPUT_DIR / "rois"
METADATA_FILE = OUTPUT_DIR / "dataset_metadata.json"
STABLE_MIN_FRAMES = 5

def create_dataset():
    # Setup directories
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    ROIS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Processing {VIDEO_PATH}...")
    
    # Load YOLO
    model = YOLO('yolov8n.pt')
    
    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open {VIDEO_PATH}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Tracking state
    track_history = {}  # track_id -> consecutive_frames
    saved_tracks = set()
    dataset_tracks = []
    
    frame_idx = 0
    with tqdm(total=total_frames) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Run tracking
            results = model.track(frame, persist=True, verbose=False, tracker="bytetrack.yaml")
            
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                classes = results[0].boxes.cls.cpu().numpy()
                
                current_frame_tracks = set(track_ids)
                
                # Check for lost tracks to reset counters (optional, but good for robustness)
                # For simplicity, we just increment counter if seen
                
                for i, track_id in enumerate(track_ids):
                    cls_id = int(classes[i])
                    if cls_id != 0: # Only people (players)
                        continue
                        
                    # Skip if already saved
                    if track_id in saved_tracks:
                        continue
                        
                    # Increment stability counter
                    track_history[track_id] = track_history.get(track_id, 0) + 1
                    
                    # Check stability
                    if track_history[track_id] >= STABLE_MIN_FRAMES:
                        # Save this track!
                        x1, y1, x2, y2 = map(int, boxes[i])
                        
                        # Validate bbox
                        h, w, _ = frame.shape
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        
                        if x2 <= x1 or y2 <= y1:
                            continue
                            
                        # Crop ROI
                        roi = frame[y1:y2, x1:x2]
                        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                        roi_filename = f"track_{track_id}.jpg"
                        roi_path = ROIS_DIR / roi_filename
                        
                        Image.fromarray(roi_rgb).save(roi_path)
                        
                        # Add to metadata
                        dataset_tracks.append({
                            "track_id": f"football_track_{track_id}",
                            "video": "football_video",
                            "frame_id": frame_idx,
                            "bbox": [x1, y1, x2, y2],
                            "roi_path": str(Path("rois") / roi_filename),
                            "stable_frames": track_history[track_id]
                        })
                        
                        saved_tracks.add(track_id)
            
            pbar.update(1)
            frame_idx += 1
            
    cap.release()
    
    # Save metadata
    metadata = {
        "num_tracks": len(dataset_tracks),
        "videos": ["football_video"],
        "stable_min_frames": STABLE_MIN_FRAMES,
        "tracks": dataset_tracks
    }
    
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)
        
    print(f"\nDataset created at {OUTPUT_DIR}")
    print(f"Total unique stable tracks: {len(dataset_tracks)}")

if __name__ == "__main__":
    create_dataset()
