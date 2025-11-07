#!/usr/bin/env python3
import time, base64, io, random, argparse, requests
from PIL import Image, ImageDraw

def b64(im):
    buf = io.BytesIO(); im.save(buf, format="JPEG", quality=85, optimize=True)
    return base64.b64encode(buf.getvalue()).decode()

def make_frame(w=640, h=360, t=0, num_boxes=2):
    """Generate a synthetic RGB frame and N moving ROIs."""
    im = Image.new("RGB", (w, h), (30, 30, 30))
    draw = ImageDraw.Draw(im)
    rois = []
    for i in range(num_boxes):
        # simple back-and-forth motion based on t and i
        bx = int((w*0.1) + ((w*0.7) * ((t+i) % 60) / 59.0))
        by = int((h*0.2) + ((h*0.5) * (((t+17*i) % 45) / 44.0)))
        bw = int(w * 0.12)
        bh = int(h * 0.22)
        rois.append({"track_id": 10 + i, "bbox": [bx, by, bw, bh]})
        draw.rectangle([bx, by, bx+bw, by+bh], outline=(230, 230, 230), width=2)
        draw.text((bx+4, by+4), f"id:{10+i}", fill=(240,240,240))
    return im, rois

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8008", help="slowpath base URL")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--secs", type=int, default=30, help="duration seconds")
    ap.add_argument("--boxes", type=int, default=2, help="ROIs per frame")
    ap.add_argument("--prompt", default="Return JSON: {\"label\":\"<category>\",\"confidence\":0.8,\"metadata\":{}}")
    ap.add_argument("--force", action="store_true", help="force enqueue (simulate trigger OFF)")
    args = ap.parse_args()

    tick_url = f"{args.base}/trigger/tick"
    metrics_url = f"{args.base}/metrics"

    print(f"Running short load: {args.secs}s at {args.fps} FPS with {args.boxes} ROIs/frame")
    period = 1.0/args.fps
    t0 = time.time()
    sent = 0
    enq = 0
    for k in range(args.secs*args.fps):
        im, rois = make_frame(num_boxes=args.boxes, t=k)
        payload = {
            "frame_id": k,
            "image_b64": b64(im),
            "rois": rois,
            "prompt_hint": args.prompt
        }
        # If forcing ON (trigger OFF), add a field many implementations check (optional):
        if args.force:
            # Many policies accept a hint; if not used in your code, ignore
            payload["force"] = True
        try:
            r = requests.post(tick_url, json=payload, timeout=2.5)
            js = r.json()
            enq += js.get("count", 0)
            sent += 1
            if k % args.fps == 0:
                print(f"[t={k/args.fps:>4.1f}s] enqueued this second: {js.get('count', 0)} (total enq={enq})")
        except Exception as e:
            print("tick error:", e)
        # maintain cadence
        elapsed = time.time() - (t0 + k*period)
        sleep = period - elapsed
        if sleep > 0:
            time.sleep(sleep)

    print("\nRun complete. Fetching /metrics...")
    try:
        m = requests.get(metrics_url, timeout=3).json()
        print("=== METRICS SUMMARY ===")
        w = m.get("worker", {})
        print("queue_depth:", w.get("queue_depth"))
        print("results_size:", w.get("results_size"))
        print("infer_latency_ms:", w.get("infer_latency_ms"))
        if "batching" in m:
            print("batching:", m["batching"])
        if "jobs" in m:
            print("jobs:", m["jobs"])
        if "trigger" in m:
            print("trigger:", m["trigger"])
        if "parsing" in m:
            print("parsing:", m["parsing"])
        if "cache_posts" in m:
            print("cache_posts:", m["cache_posts"])
    except Exception as e:
        print("metrics error:", e)

if __name__ == "__main__":
    main()
