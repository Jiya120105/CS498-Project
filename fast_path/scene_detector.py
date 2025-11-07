import cv2
import time
import numpy as np

# --- Configuration ---
VIDEO_PATH = "MOT16_01.mp4" 
OUTPUT_FILE = 'flow_latencies.npy'
DOWNSAMPLE_SIZE = (320, 240) 
# ---------------------

def benchmark_optical_flow(video_path):
    print(f"Starting Optical Flow benchmark (Downsampled to {DOWNSAMPLE_SIZE})...")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    frame_count = 0
    latencies = []
    
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read first frame.")
        return
    
    prev_frame_small = cv2.resize(prev_frame, DOWNSAMPLE_SIZE, interpolation=cv2.INTER_AREA)
    prev_gray = cv2.cvtColor(prev_frame_small, cv2.COLOR_BGR2GRAY)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        frame_small = cv2.resize(frame, DOWNSAMPLE_SIZE, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
        
        # Start timer
        start_time = time.perf_counter()
        
        # --- 1. Calculate Optical Flow ---
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # --- 2. Calculate Mean Magnitude (The "Change Score") ---
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2) # <<< --- FIX
        mean_magnitude = np.mean(magnitude) 
        
        end_time = time.perf_counter()
        
        latency_ms = (end_time - start_time) * 1000
        latencies.append(latency_ms)
        
        # Update previous frame
        prev_gray = gray
        
        if frame_count % 30 == 0: # Print every second or so
            print(f"Frame {frame_count} | Latency: {latency_ms:.2f} ms | Change Score: {mean_magnitude:.4f}")

    # Clean up
    cap.release()
    
    # --- Print and Save Results ---
    latencies_np = np.array(latencies)
    print("\n--- Optical Flow Benchmark Complete ---")
    print(f"Total frames: {frame_count}")
    print(f"Mean latency: {np.mean(latencies_np):.2f} ms")
    print(f"p50 (Median): {np.percentile(latencies_np, 50):.2f} ms")
    print(f"p95 latency:  {np.percentile(latencies_np, 95):.2f} ms")
    print(f"p99 latency:  {np.percentile(latencies_np, 99):.2f} ms")
    
    np.save(OUTPUT_FILE, latencies_np)
    print(f"Saved latencies to {OUTPUT_FILE}")

if __name__ == "__main__":
    benchmark_optical_flow(VIDEO_PATH)