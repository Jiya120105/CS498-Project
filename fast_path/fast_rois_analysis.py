import json
import numpy as np
import matplotlib.pyplot as plt


ROI_FILE = 'fast_path_rois.json'
OUTPUT_HISTOGRAM_FILE = 'track_duration_histogram.png'


def analyze_track_stability(roi_file):
    print(f"Loading ROI data from {roi_file}...")
    try:
        with open(roi_file, 'r') as f:
            all_frame_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {roi_file} not found. Please run the yolov8.py script first.")
        return

    tracks = {}
    for frame in all_frame_data:
        frame_id = frame['frame_id']
        for det in frame['detections']:
            track_id = det['track_id']
            if track_id not in tracks:
                tracks[track_id] = []
            tracks[track_id].append(frame_id)

    if not tracks:
        print("No tracks found in the data.")
        return

    print(f"Found {len(tracks)} unique track IDs.")
    all_consecutive_durations = []
    
    for track_id, frame_ids in tracks.items():
        if not frame_ids:
            continue
        
        frame_ids.sort() # Ensure they are in order
        
        current_run_start = frame_ids[0]
        current_run_length = 1
        
        for i in range(1, len(frame_ids)):
            if frame_ids[i] == frame_ids[i-1] + 1:
                
                current_run_length += 1
            else:
                # Gap detected, end the previous run
                all_consecutive_durations.append(current_run_length)
                # Start a new run
                current_run_start = frame_ids[i]
                current_run_length = 1
        
        # Add the last run
        all_consecutive_durations.append(current_run_length)

    if not all_consecutive_durations:
        print("No consecutive runs found.")
        return

    # 3. Print statistics
    durations_np = np.array(all_consecutive_durations)
    
    print("\n--- Track Stability Analysis ---")
    print(f"Total consecutive track segments analyzed: {len(durations_np)}")
    print(f"Mean duration: {np.mean(durations_np):.2f} frames")
    print(f"p25 duration: {np.percentile(durations_np, 25):.2f} frames")
    print(f"p50 (Median) duration: {np.percentile(durations_np, 50):.2f} frames")
    print(f"p75 duration: {np.percentile(durations_np, 75):.2f} frames")
    print(f"p95 duration: {np.percentile(durations_np, 95):.2f} frames")
    print(f"Max duration: {np.max(durations_np)} frames")

    # 4. Plot histogram
    plt.figure(figsize=(10, 6))
    # Use a log scale for the x-axis if durations are very long
    max_val = np.percentile(durations_np, 98) # Clip for better viz
    bins = np.linspace(0, max_val, 50)
    
    plt.hist(durations_np, bins=bins, edgecolor='black')
    
    plt.title('Histogram of Consecutive Track Durations', fontsize=16)
    plt.xlabel('Track Duration (Consecutive Frames)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add a line for the median
    median_val = np.median(durations_np)
    plt.axvline(median_val, color='red', linestyle='dashed', linewidth=2,
                label=f'Median Duration: {median_val:.1f} frames')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_HISTOGRAM_FILE)
    print(f"\nHistogram saved to {OUTPUT_HISTOGRAM_FILE}")
    plt.show()

if __name__ == "__main__":
    analyze_track_stability(ROI_FILE)