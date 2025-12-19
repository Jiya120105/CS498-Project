import cv2
import json
import argparse
import os
import glob
from collections import defaultdict

def render_video(image_dir, log_file, output_video, fps=30):
    print(f"[Render] Reading log file: {log_file}")
    
    # Load logs: FrameID -> List[TrackData]
    frame_data = defaultdict(list)
    with open(log_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                frame_data[data['frame']].append(data)
            except json.JSONDecodeError:
                continue
    
    # Get image files
    image_files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    if not image_files:
        print(f"[Render] No images found in {image_dir}")
        return

    # Video Writer Setup
    first_img = cv2.imread(image_files[0])
    h, w, _ = first_img.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (w, h))
    
    print(f"[Render] Processing {len(image_files)} frames...")
    
    for i, img_path in enumerate(image_files):
        frame_id = i + 1
        frame = cv2.imread(img_path)
        
        # Draw Overlays
        if frame_id in frame_data:
            tracks = frame_data[frame_id]
            
            # Status HUD
            resolved_count = sum(1 for t in tracks if t['status'] == 'Resolved')
            pending_count = sum(1 for t in tracks if t['status'] != 'Resolved')
            
            cv2.putText(frame, f"Frame: {frame_id}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, f"Resolved: {resolved_count} | Pending: {pending_count}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            for track in tracks:
                x1, y1, x2, y2 = map(int, track['bbox'])
                status = track['status']
                label = track.get('semantic_label', 'Unknown')
                conf = track.get('confidence', 0.0)
                
                # Color Logic
                color = (128, 128, 128) # Gray (Pending)
                if status == 'Resolved':
                    if label.lower() in ['yes', 'player', 'pedestrian']: # "player" is the mock response
                         color = (0, 255, 0) # Green (Yes)
                    elif label.lower() in ['no', 'background']:
                        color = (0, 0, 255) # Red (No)
                    else:
                         color = (0, 255, 255) # Yellow (Uncertain)

                # Draw Box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw Label
                caption = f"ID:{track['track_id']} {label}"
                cv2.putText(frame, caption, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        out.write(frame)
        
        if frame_id % 100 == 0:
            print(f"[Render] Processed {frame_id} frames...")

    out.release()
    print(f"[Render] Saved video to {output_video}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir", help="Directory containing .jpg frames")
    parser.add_argument("log_file", help="Path to system_events.jsonl")
    parser.add_argument("--output", default="output_demo.mp4", help="Output video filename")
    parser.add_argument("--fps", type=int, default=30, help="Video FPS")
    args = parser.parse_args()
    
    render_video(args.image_dir, args.log_file, args.output, args.fps)
