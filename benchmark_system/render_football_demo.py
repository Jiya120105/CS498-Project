"""
Render Football Demo (Fast)

Uses pre-computed VLM labels from ground_truth.json to render the video quickly.
No VLM inference is performed during rendering.

Output: football_demo_fast.mp4
"""

import cv2
import json
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm

# Configuration
VIDEO_PATH = "football_video.mp4"
OUTPUT_PATH = "football_jersey_demo.mp4"
GT_PATH = "football_dataset/ground_truth.json"

def load_labels():
    if not Path(GT_PATH).exists():
        print(f"Error: {GT_PATH} not found.")
        return {}
        
    with open(GT_PATH, 'r') as f:
        data = json.load(f)
    
    # Map track_id (e.g. "football_track_1") to label
    # We need to map local integer ID (1) to label
    labels = {}
    for tid_str, info in data['tracks'].items():
        # format: "football_track_123" -> 123
        try:
            local_id = int(tid_str.split('_')[-1])
            label = info['label']
            if label == "Yes":
                labels[local_id] = ("Yellow", (0, 255, 0)) # Green
            else:
                labels[local_id] = ("Not Yellow", (0, 0, 255)) # Red
        except:
            pass
            
    return labels

def process_video():
    print("Loading labels...")
    track_labels = load_labels()
    print(f"Loaded labels for {len(track_labels)} tracks.")
    
    print("Loading YOLO...")
    yolo = YOLO('yolov8n.pt')
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    print(f"Rendering to {OUTPUT_PATH}...")
    
    with tqdm(total=total_frames) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # Fast Path (YOLO)
            results = yolo.track(frame, persist=True, verbose=False, tracker="bytetrack.yaml")
            
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                classes = results[0].boxes.cls.cpu().numpy()
                
                for i, track_id in enumerate(track_ids):
                    cls_id = int(classes[i])
                    if cls_id != 0: continue # Only people
                    
                    x1, y1, x2, y2 = map(int, boxes[i])
                    
                    # Look up label
                    label, color = track_labels.get(track_id, ("Pending...", (128, 128, 128)))
                    
                    # Draw
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Label bg
                    label_text = f"#{track_id} {label}"
                    (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    cv2.rectangle(frame, (x1, y1 - 20), (x1 + tw, y1), color, -1)
                    
                    # Label text
                    text_color = (0,0,0)
                    cv2.putText(frame, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

            out.write(frame)
            pbar.update(1)
            
    cap.release()
    out.release()
    print("Done.")

if __name__ == "__main__":
    process_video()