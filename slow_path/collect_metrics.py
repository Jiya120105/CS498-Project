#!/usr/bin/env python3
import time, csv, argparse, requests

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8008", help="slowpath base URL")
    ap.add_argument("--secs", type=int, default=60, help="duration to collect (seconds)")
    ap.add_argument("--out", default="metrics.csv", help="output CSV path")
    args = ap.parse_args()

    url = f"{args.base}/metrics"
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        # header
        w.writerow([
            "t", "queue_depth", "results_size",
            "infer_count", "infer_p50", "infer_p95", "infer_max",
            "cache_ok", "cache_fail",
            "jobs_enqueued_total",
            "trigger_seen", "trigger_enqueued", "trigger_enqueue_rate"
        ])
        t0 = time.time()
        for _ in range(args.secs):
            try:
                m = requests.get(url, timeout=3).json()
            except Exception as e:
                # write blanks to keep cadence
                now = time.time() - t0
                w.writerow([round(now,2), "", "", "", "", "", "", "", "", "", "", "", ""])
                time.sleep(1.0)
                continue

            # support both flat and nested "worker"/"trigger"
            worker = m.get("worker", m)
            trigger = m.get("trigger", {})

            infer = worker.get("infer_latency_ms") or {}
            now = time.time() - t0
            w.writerow([
                round(now,2),
                worker.get("queue_depth", ""),
                worker.get("results_size", ""),
                infer.get("count", ""),
                infer.get("p50", ""),
                infer.get("p95", ""),
                infer.get("max", ""),
                (worker.get("cache_posts") or {}).get("ok", ""),
                (worker.get("cache_posts") or {}).get("fail", ""),
                worker.get("jobs_enqueued_total", ""),
                trigger.get("seen", ""),
                trigger.get("enqueued", ""),
                trigger.get("enqueue_rate", ""),
            ])
            time.sleep(1.0)

if __name__ == "__main__":
    main()
