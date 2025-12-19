import cv2
import os
from pathlib import Path
from tqdm import tqdm

video_path = "football_video.mp4"
output_dir = Path("football_video/img1")
output_dir.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Extracting {total_frames} frames to {output_dir}...")

frame_idx = 0
with tqdm(total=total_frames) as pbar:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        filename = output_dir / f"{frame_idx:06d}.jpg"
        cv2.imwrite(str(filename), frame)
        
        frame_idx += 1
        pbar.update(1)

cap.release()
print("Done.")
