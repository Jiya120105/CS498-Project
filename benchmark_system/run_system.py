import argparse
import os
import cv2
import time
import json
import asyncio
import uuid
import numpy as np
from ultralytics import YOLO

# --- Configuration ---
# Force the worker to use the local semantic cache instance we will create
os.environ["USE_LOCAL_SEMANTIC_CACHE"] = "1"

# Import system components
# Note: These imports must happen AFTER setting the env var if the module reads it at top-level
# (which worker.py does for _use_local_cache)
from slow_path.service.worker import Worker, Job, get_local_cache
from slow_path.service.triggers.policy import TriggerPolicy, TriggerConfig

# --- Constants ---
YOLO_MODEL = 'yolov8n.pt'
TRACKER_CONFIG = "bytetrack.yaml"
OUTPUT_LOG = 'system_events.jsonl'
DEBUG = True

import glob

async def run_system(video_path, query, model_name, max_queue,
                     stable_min_frames: int = 5,
                     vlm_device: str = None):
    # 1. Setup Environment
    os.environ["SLOWPATH_MODEL"] = model_name
    if vlm_device:
        os.environ["SLOWPATH_DEVICE"] = vlm_device
    print(f"[System] Initializing... Model: {model_name}, Query: '{query}'")

    # 2. Initialize Components
    # Fast Path
    print(f"[System] Loading YOLO: {YOLO_MODEL}")
    yolo = YOLO(YOLO_MODEL)

    # Optional: place YOLO on a specific device to avoid contention with VLM
    # Ultralytics accepts .to('cpu') or .to('cuda:0')
    # If you need finer control, we can add a CLI flag for this (--yolo_device)
    try:
        # Honor an optional env override if present
        ydev = os.getenv("YOLO_DEVICE")
        if ydev:
            yolo.to(ydev)
    except Exception:
        pass

    # Slow Path Worker
    print(f"[System] Starting VLM Worker (Queue={max_queue})")
    worker = Worker(max_queue_size=max_queue)
    worker_task = asyncio.create_task(worker.run())

    # Offload mode is stability-based; no policy strategies used

    # Shared State
    # The worker automatically uses the local cache singleton if USE_LOCAL_SEMANTIC_CACHE is set
    # We get a reference to it to read from it.
    cache = get_local_cache()
    if cache is None:
        print("[System] Error: Local cache not initialized correctly in worker.")
        return

    # Track stability state
    consecutive_seen: dict[int, int] = {}
    last_seen_frame: dict[int, int] = {}

    # 3. Input Source (Video or Image Directory)
    if os.path.isdir(video_path):
        # Handle MOT dataset structure: if 'img1' exists, use it
        if os.path.isdir(os.path.join(video_path, "img1")):
            video_path = os.path.join(video_path, "img1")

        print(f"[System] Reading images from directory: {video_path}")
        image_files = sorted(glob.glob(os.path.join(video_path, "*.jpg")))
        if not image_files:
             print(f"[System] Error: No .jpg files found in {video_path}")
             return
        # Create a generator that yields frames
        def frame_generator():
             for img_file in image_files:
                 frame = cv2.imread(img_file)
                 yield frame
        cap = None
        frames = frame_generator()
        total_frames_est = len(image_files)
    else:
        # Video File
        print(f"[System] Processing video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[System] Error: Cannot open video {video_path}")
            return
        def frame_generator():
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                yield frame
        frames = frame_generator()
        total_frames_est = "Unknown"

    print(f"[System] Logging to: {OUTPUT_LOG}")
    
    # Clear log file
    with open(OUTPUT_LOG, 'w') as f:
        pass

    frame_id = 0
    start_time = time.time()

    try:
        for frame in frames:
            if frame is None: break
            
            frame_id += 1
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # For VLM/Policy

            # --- FAST PATH: Tracking ---
            # persist=True is crucial for tracking
            results = yolo.track(frame, persist=True, verbose=False, tracker=TRACKER_CONFIG)
            
            current_tracks = []
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                
                for i, track_id_np in enumerate(track_ids):
                    track_id = int(track_id_np) # Ensure native int for dict keys/cache
                    box = boxes[i] # x1, y1, x2, y2
                    x1, y1, x2, y2 = map(int, box)
                    w, h = x2-x1, y2-y1
                    bbox = [x1, y1, w, h]

                    # --- DECISION LOGIC ---
                    # 1. Check Cache (no expiry)
                    entry = cache.get(track_id, frame_id, ttl=0)

                    status = "PENDING"
                    label = "Unknown"
                    confidence = 0.0
                    is_resolved = False

                    if entry:
                        status = "Resolved"
                        label = entry.label
                        confidence = entry.confidence
                        is_resolved = True

                    # 2. Update stability counters
                    prev = last_seen_frame.get(track_id)
                    if prev is not None and prev == frame_id - 1:
                        consecutive_seen[track_id] = consecutive_seen.get(track_id, 0) + 1
                    else:
                        consecutive_seen[track_id] = 1
                    last_seen_frame[track_id] = frame_id

                    # 3. Stability-based offload: enqueue once when stable and not pending/resolved
                    if (not is_resolved) and (track_id not in worker.pending_tracks) and consecutive_seen.get(track_id, 0) >= stable_min_frames:
                        prompt = f"Is this {query}? Answer Yes or No."
                        job = Job(
                            job_id=str(uuid.uuid4()),
                            frame_id=frame_id,
                            track_id=track_id,
                            bbox=bbox,
                            frame_rgb=frame_rgb,
                            prompt=prompt
                        )
                        worker.submit_job(job)
                        status = "TriggeredStable"

                    # --- LOGGING ---
                    log_entry = {
                        "frame": frame_id,
                        "track_id": int(track_id),
                        "bbox": [x1, y1, x2, y2],
                        "status": status,
                        "semantic_label": label,
                        "confidence": confidence,
                        "query": query
                    }
                    current_tracks.append(log_entry)

            # No periodic sweep; stability-based offload runs continuously

            # Write batch to file
            if current_tracks:
                with open(OUTPUT_LOG, 'a') as f:
                    for t in current_tracks:
                        f.write(json.dumps(t) + "\n")

            if frame_id % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_id / elapsed
                q_depth = worker.queue.qsize()
                print(f"Frame {frame_id} | FPS: {fps:.1f} | Queue: {q_depth}/{max_queue} | Dropped: {worker.jobs_dropped_total}")
            
            # Yield control to the worker task
            await asyncio.sleep(0)

    except KeyboardInterrupt:
        print("[System] Stopping...")
    finally:
        if cap: cap.release()
        # Cancel worker
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass
        
        print("[System] Finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", help="Path to input video")
    parser.add_argument("--query", default="person wearing a red shirt", help="Natural language query")
    parser.add_argument("--model", default="mock", help="VLM model to use (mock, blip)")
    parser.add_argument("--queue_size", type=int, default=25, help="Max slow path queue depth")
    # deprecated: strategies arg removed; offload is stability-based
    # no batch size; sequential VLM for minimal GPU contention
    parser.add_argument("--stable_min_frames", type=int, default=5, help="Only offload tracks seen in at least this many consecutive frames")
    parser.add_argument("--vlm_device", default=None, help="Force VLM device (e.g., 'cpu', 'cuda', 'cuda:1')")
    
    args = parser.parse_args()
    
    asyncio.run(run_system(args.video_path, args.query, args.model, args.queue_size,
                           stable_min_frames=args.stable_min_frames,
                           vlm_device=args.vlm_device))
