import cv2
import time
import numpy as np
from ultralytics import YOLO
import json


VIDEO_PATH = "MOT16_01.mp4"
MODEL_NAME = 'yolov8n.pt'
TRACKER_CONFIG = "bytetrack.yaml"
ROI_OUTPUT_FILE = 'fast_path_rois.json'

def benchmark_fast_path(video_path):
    print(f"Starting Fast-Only (YOLOv8-nano + ByteTrack) benchmark...")
    model = YOLO(MODEL_NAME)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    frame_count = 0
    latencies = []
    all_frame_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        
        start_time = time.perf_counter()
        
        
        results = model.track(frame, persist=True, tracker=TRACKER_CONFIG, verbose=False)
        
        
        end_time = time.perf_counter()
        
        
        latency_ms = (end_time - start_time) * 1000
        latencies.append(latency_ms)
        
        
        current_frame_detections = []
        result = results[0]
        
        if result.boxes.id is not None:
            class_names = result.names
            
            
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.cpu().numpy().astype(int)
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            
           
            for i in range(len(track_ids)):
                detection_data = {
                    "track_id": int(track_ids[i]),
                    "bbox_xyxy": boxes_xyxy[i].tolist(), # [x1, y1, x2, y2]
                    "confidence": float(confidences[i]),
                    "class_id": int(class_ids[i]),
                    "class_name": class_names[class_ids[i]]
                }
                current_frame_detections.append(detection_data)
        
        # Add this frame's data to the main list
        all_frame_data.append({
            "frame_id": frame_count,
            "latency_ms": latency_ms,
            "detections": current_frame_detections
        })
        
        
        if frame_count % 100 == 0:
            print(f"Processed frame {frame_count}... Current latency: {latency_ms:.2f} ms")

    
    cap.release()
    cv2.destroyAllWindows()
    
    # --- Print and Save Results ---
    latencies_np = np.array(latencies)
    print("\n--- Fast-Only Benchmark Complete ---")
    print(f"Total frames: {frame_count}")
    print(f"Mean latency: {np.mean(latencies_np):.2f} ms")
    print(f"p50 (Median): {np.percentile(latencies_np, 50):.2f} ms")
    print(f"p95 latency:  {np.percentile(latencies_np, 95):.2f} ms")
    print(f"p99 latency:  {np.percentile(latencies_np, 99):.2f} ms")
    
    # Save latencies for plotting
    np.save('fast_latencies.npy', latencies_np)
    print("Saved latencies to fast_latencies.npy")
    
    
    print(f"Saving ROI data to {ROI_OUTPUT_FILE}...")
    with open(ROI_OUTPUT_FILE, 'w') as f:
        json.dump(all_frame_data, f, indent=2)
    print("ROI data saved.")

if __name__ == "__main__":
    benchmark_fast_path(VIDEO_PATH)