import argparse
import json
import time
from typing import Dict, List
import os, sys

# Ensure repo root is on sys.path when running as: python slow_path/bench_roi_vlm.py
THIS_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import cv2
import numpy as np
from PIL import Image

try:
    import torch
except Exception:
    torch = None  # optional sync for CUDA timing

from slow_path.service.models.loader import load_model


def load_roi_index(roi_file: str) -> Dict[int, List[dict]]:
    print(f"Loading ROIs from {roi_file}...")
    with open(roi_file, "r") as f:
        all_frames = json.load(f)
    # Convert list of frames to dict: frame_id -> detections
    return {int(fr["frame_id"]): fr.get("detections", []) for fr in all_frames}


def xyxy_to_xywh(b):
    x1, y1, x2, y2 = [int(v) for v in b]
    return [x1, y1, int(x2 - x1), int(y2 - y1)]


def pstats(vals: List[float]) -> Dict[str, float]:
    if not vals:
        return {"count": 0}
    a = sorted(vals)
    n = len(a)
    def p(q: float):
        idx = int(q * (n - 1))
        return float(a[idx])
    return {
        "count": n,
        "p50": p(0.50),
        "p90": p(0.90),
        "p95": p(0.95),
        "max": float(a[-1]),
        "min": float(a[0]),
        "mean": float(sum(a) / n),
    }


def bench(video_path: str,
          roi_file: str,
          model_name: str,
          mode: str,
          limit_frames: int | None,
          per_frame: str,
          prompt: str | None,
          save_latencies: str | None):
    roi_index = load_roi_index(roi_file)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"Error: Could not open video file {video_path}")

    model = load_model(model_name)

    lat_ms: List[float] = []
    samples = 0

    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        if limit_frames and frame_id > limit_frames:
            break

        detections = roi_index.get(frame_id, [])
        if not detections:
            continue

        rois_to_eval: List[dict]
        if per_frame == "first":
            rois_to_eval = [detections[0]]
        else:  # all
            rois_to_eval = detections

        for det in rois_to_eval:
            bbox_xyxy = det.get("bbox_xyxy") or det.get("bbox")
            if bbox_xyxy is None:
                continue
            # Crop ROI if mode == roi, else keep full frame
            if mode == "roi":
                x1, y1, x2, y2 = [int(c) for c in bbox_xyxy]
                if x1 >= x2 or y1 >= y2:
                    continue
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            else:  # full frame
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            image = Image.fromarray(rgb)

            # Accurate GPU timing: synchronize before/after
            if torch is not None and torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model.infer(image, prompt or "")
            if torch is not None and torch.cuda.is_available():
                torch.cuda.synchronize()
            dt_ms = (time.perf_counter() - t0) * 1000.0
            lat_ms.append(dt_ms)
            samples += 1

    cap.release()

    stats = pstats(lat_ms)
    print(f"Mode={mode} | Model={model_name} | Frames={frame_id} | Samples={samples}")
    print("Latency ms:", json.dumps(stats, indent=2))

    if save_latencies:
        with open(save_latencies, "w") as f:
            json.dump({"mode": mode, "model": model_name, "stats": stats, "samples": lat_ms}, f)
        print(f"Saved raw latencies to {save_latencies}")


def main():
    ap = argparse.ArgumentParser(description="Benchmark VLM latency on ROIs vs full frames.")
    ap.add_argument("--video", required=True, help="Path to source video")
    ap.add_argument("--roi_file", default="fast_path/fast_path_rois.json", help="ROI JSON with frame_id+detections")
    ap.add_argument("--model", default="blip", choices=["mock", "blip", "blip2"], help="VLM model to use")
    ap.add_argument("--mode", default="roi", choices=["roi", "full"], help="Process cropped ROI or full frame")
    ap.add_argument("--per_frame", default="first", choices=["first", "all"], help="How many ROIs per frame to run")
    ap.add_argument("--limit_frames", type=int, default=None, help="Limit number of frames processed")
    ap.add_argument("--prompt", type=str, default=None, help="Optional prompt override")
    ap.add_argument("--save_latencies", type=str, default=None, help="Optional JSON path to save raw latencies")
    args = ap.parse_args()

    bench(video_path=args.video,
          roi_file=args.roi_file,
          model_name=args.model,
          mode=args.mode,
          limit_frames=args.limit_frames,
          per_frame=args.per_frame,
          prompt=args.prompt,
          save_latencies=args.save_latencies)


if __name__ == "__main__":
    main()
