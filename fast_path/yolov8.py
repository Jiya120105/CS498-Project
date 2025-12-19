import cv2
import time
import numpy as np
from ultralytics import YOLO


VIDEO_PATH = "MOT16_01.mp4"
MODEL_NAME = 'yolov8n.pt'     # YOLOv8-Nano (fastest)
TRACKER_CONFIG = "bytetrack.yaml"
# ---------------------

def benchmark_fast_path(video_path):
    print(f"Starting Fast-Only (YOLOv8-nano + ByteTrack) benchmark...")
    
    
    model = YOLO(MODEL_NAME)
    
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    frame_count = 0
    latencies = []

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
        
        if frame_count % 100 == 0:
            print(f"Processed frame {frame_count}... Current latency: {latency_ms:.2f} ms")

        # Display the results
        # annotated_frame = results[0].plot()
        # cv2.imshow("Fast Path", annotated_frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    
    
    latencies_np = np.array(latencies)
    print("\n--- Fast-Only Benchmark Complete ---")
    print(f"Total frames: {frame_count}")
    print(f"Mean latency: {np.mean(latencies_np):.2f} ms")
    print(f"p50 (Median): {np.percentile(latencies_np, 50):.2f} ms")
    print(f"p95 latency:  {np.percentile(latencies_np, 95):.2f} ms")
    print(f"p99 latency:  {np.percentile(latencies_np, 99):.2f} ms")
    
    
    np.save('fast_latencies.npy', latencies_np)
    print("Saved latencies to fast_latencies.npy")

if __name__ == "__main__":
    benchmark_fast_path(VIDEO_PATH)