import cv2
import os
from ultralytics import YOLO
import argparse
from tqdm import tqdm

def process_video(video_path, output_path):
    print(f"Processing {video_path}...")
    
    # Load YOLO
    model = YOLO('yolov8n.pt')
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Output video
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    frame_idx = 0
    with tqdm(total=total_frames) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Run YOLO
            results = model.track(frame, persist=True, verbose=False, tracker="bytetrack.yaml")
            
            # Draw bounding boxes
            annotated_frame = results[0].plot()
            
            out.write(annotated_frame)
            pbar.update(1)
            frame_idx += 1
            
            # Optional: Stop early for testing
            # if frame_idx > 300: break
            
    cap.release()
    out.release()
    print(f"Saved annotated video to {output_path}")

if __name__ == "__main__":
    process_video("football_video.mp4", "football_yolo_preview.mp4")
