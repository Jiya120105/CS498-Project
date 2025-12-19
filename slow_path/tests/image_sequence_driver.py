#!/usr/bin/env python3
import argparse, time, io, base64, requests, sys, os, glob
from PIL import Image
import numpy as np
import cv2

def sorted_paths(root, pattern="*.jpg"):
    paths = glob.glob(os.path.join(root, pattern))
    if not paths:
        paths = glob.glob(os.path.join(root, "*.jpeg")) + glob.glob(os.path.join(root, "*.png"))
    def key(p):
        base = os.path.basename(p)
        nums = "".join(ch if ch.isdigit() else " " for ch in base).split()
        return tuple(int(n) for n in nums) if nums else (base,)
    return sorted(paths, key=key)

def resize_nd(im_rgb, target_wh):
    tw, th = target_wh
    return cv2.resize(im_rgb, (tw, th), interpolation=cv2.INTER_LINEAR)

def to_b64(im_rgb, jpeg_q=85):
    pil = Image.fromarray(im_rgb)
    buf = io.BytesIO(); pil.save(buf, format="JPEG", quality=jpeg_q, optimize=True)
    return base64.b64encode(buf.getvalue()).decode()

def clamp_box(x, y, w, h, W, H):
    x = max(0, min(int(x), W-1))
    y = max(0, min(int(y), H-1))
    w = max(1, min(int(w), W - x))
    h = max(1, min(int(h), H - y))
    return [x, y, w, h]

def detect_motion_rois(prev_gray, gray, min_area=600, max_rois=3):
    diff = cv2.absdiff(prev_gray, gray)
    blur = cv2.GaussianBlur(diff, (5,5), 0)
    _, th = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    th = cv2.dilate(th, None, iterations=2)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if w*h >= min_area:
            boxes.append((x,y,w,h))
    boxes.sort(key=lambda b: b[2]*b[3], reverse=True)
    return boxes[:max_rois]

def grid_rois(W, H, cols=2, rows=2, max_rois=3):
    rois = []
    cw, ch = W // cols, H // rows
    for r in range(rows):
        for c in range(cols):
            x, y = c*cw, r*ch
            w, h = int(cw*0.9), int(ch*0.9)
            rois.append((x + int(cw*0.05), y + int(ch*0.05), w, h))
    return rois[:max_rois]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder containing JPG/PNG frames")
    ap.add_argument("--pattern", default="*.jpg", help="glob pattern (e.g., *.jpg, frame_*.png)")
    ap.add_argument("--base", default="http://127.0.0.1:8008", help="slowpath base URL")
    ap.add_argument("--fps", type=float, default=15.0, help="send at this FPS (approx)")
    ap.add_argument("--resize", default="640x360", help="resize WxH before sending (e.g., 640x360)")
    ap.add_argument("--max-rois", type=int, default=3, help="max ROIs per frame")
    ap.add_argument("--min-area", type=int, default=600, help="min contour area to consider")
    ap.add_argument("--prompt", default='Return JSON: {"label":"<category>","confidence":0.8,"metadata":{}}')
    ap.add_argument("--fallback-grid", action="store_true", help="if no motion found, send a small grid of ROIs")
    args = ap.parse_args()

    try:
        w,h = args.resize.lower().split("x")
        target_wh = (int(w), int(h))
    except Exception:
        print("Invalid --resize; use WxH like 640x360"); return 1

    paths = sorted_paths(args.root, args.pattern)
    if not paths:
        print("No images found in", args.root, "pattern", args.pattern)
        return 1

    tick_url = f"{args.base}/trigger/tick"
    period = 1.0/args.fps if args.fps > 0 else 0

    # load first frame
    im0 = np.array(Image.open(paths[0]).convert("RGB"))
    im0r = resize_nd(im0, target_wh)
    H, W = im0r.shape[0], im0r.shape[1]
    prev_gray = cv2.cvtColor(im0r, cv2.COLOR_RGB2GRAY)

    tstart = time.time()
    for idx, p in enumerate(paths):
        im = np.array(Image.open(p).convert("RGB"))
        imr = resize_nd(im, target_wh)
        gray = cv2.cvtColor(imr, cv2.COLOR_RGB2GRAY)

        boxes = detect_motion_rois(prev_gray, gray, min_area=args.min_area, max_rois=args.max_rois)
        if not boxes and args.fallback_grid:
            boxes = grid_rois(W, H, cols=2, rows=2, max_rois=args.max_rois)

        img_b64 = to_b64(imr, jpeg_q=85)
        rois = [{"track_id": 100+i, "bbox": clamp_box(x,y,w,h,W,H)} for i,(x,y,w,h) in enumerate(boxes)]
        payload = {
            "frame_id": idx,
            "image_b64": img_b64,
            "rois": rois,
            "prompt_hint": args.prompt
        }

        try:
            r = requests.post(tick_url, json=payload, timeout=3)
            if r.status_code != 200:
                print(f"[{idx}] tick status={r.status_code} body={r.text[:160]}")
            elif idx % max(1,int(args.fps)) == 0:
                js = r.json()
                print(f"[t={idx/args.fps:>5.1f}s] sent {len(rois)} rois; enqueued={js.get('count',0)}")
        except Exception as e:
            print(f"[{idx}] tick error:", e)

        prev_gray = gray

        if period > 0:
            elapsed = time.time() - (tstart + (idx+1)*period)
            sleep = period - elapsed
            if sleep > 0:
                time.sleep(sleep)

    print("\nRun complete. Fetching /metrics...")
    try:
        metrics_url = f"{args.base}/metrics"
        m = requests.get(metrics_url, timeout=3).json()
        print("=== METRICS SUMMARY ===")
        w = m.get("worker", {})
        c = m.get("cache", {})
        trig = m.get("trigger", {})
        print("queue_depth:", w.get("queue_depth"))
        print("results_size:", w.get("results_size"))
        print("infer_latency_ms:", w.get("infer_latency_ms"))
        # new worker-level fields
        print("jobs_enqueued_total:", w.get("jobs_enqueued_total"))
        print("jobs_success:", w.get("jobs_success"))
        print("jobs_error:", w.get("jobs_error"))
        print("throughput_jobs_per_sec:", w.get("throughput_jobs_per_sec"))

        # cache summary (moved to top-level 'cache')
        if c:
            print("cache_ok:", c.get("ok"), "cache_fail:", c.get("fail"), "hit_ratio:", c.get("hit_ratio"))
        else:
            # backwards compatibility: older metric name
            cp = w.get("cache_posts") or m.get("cache_posts")
            if cp:
                print("cache_posts:", cp)

        # trigger
        print("trigger:", trig)

        # optional model output stats
        model_out = m.get("model_output", {})
        if model_out:
            print("model_output keys:", list(model_out.keys()))

    except Exception as e:
        print("metrics error:", e)

    return 0

if __name__ == "__main__":
    sys.exit(main())
