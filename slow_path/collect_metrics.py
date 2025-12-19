#!/usr/bin/env python3
import time, csv, argparse, requests, json
from typing import List, Dict, Any, Optional

def _now_s(start: float) -> float:
    return time.time() - start

def collect_metrics(url: str, secs: int) -> List[Dict[str, Any]]:
    """Collect metrics for `secs` seconds from `url` and return a list of
    per-second metric dicts (in order)."""
    rows: List[Dict[str, Any]] = []
    t0 = time.time()
    # sample exactly once per second relative to t0
    for i in range(secs):
        target = t0 + (i + 1)
        try:
            resp = requests.get(url, timeout=3)
            resp.raise_for_status()
            m = resp.json()
        except Exception:
            rows.append({"t": round(_now_s(t0), 2)})
            # sleep to the next whole-second target
            delay = max(0.0, target - time.time())
            time.sleep(delay)
            continue

        worker = m.get("worker", {})
        trigger = m.get("trigger", {})
        cache   = m.get("cache", {})
        infer   = worker.get("infer_latency_ms") or {}

        row = {
            "t": round(_now_s(t0), 2),
            # worker
            "queue_depth": worker.get("queue_depth"),
            "results_size": worker.get("results_size"),
            "infer_count": infer.get("count"),
            "infer_p50": infer.get("p50"),
            "infer_p95": infer.get("p95"),
            "infer_p99": infer.get("p99"),
            "infer_max": infer.get("max"),
            "jobs_enqueued_total": worker.get("jobs_enqueued_total"),
            "jobs_success": worker.get("jobs_success"),
            "jobs_error": worker.get("jobs_error"),
            "throughput_jobs_per_sec": worker.get("throughput_jobs_per_sec"),
            # cache
            "cache_ok": cache.get("ok"),
            "cache_fail": cache.get("fail"),
            "cache_hit_ratio": cache.get("hit_ratio"),
            # trigger
            "trigger_seen": trigger.get("seen"),
            "trigger_enqueued": trigger.get("enqueued"),
            "trigger_enqueue_rate": trigger.get("enqueue_rate"),
            "trigger_changed_scene": trigger.get("changed_scene"),
            "trigger_expired_total": trigger.get("expired_total"),
            "trigger_periodic": trigger.get("periodic"),
            # optional model output (kept as JSON string to avoid dynamic columns)
            "confidence_stats_per_label": json.dumps(
                (m.get("model_output") or {}).get("confidence_stats_per_label", {}),
                separators=(",", ":")
            ),
        }
        rows.append(row)

        # sleep to the next whole-second target
        delay = max(0.0, target - time.time())
        time.sleep(delay)

    return rows


CSV_HEADER = [
    "t",
    # worker
    "queue_depth", "results_size",
    "infer_count", "infer_p50", "infer_p95", "infer_p99", "infer_max",
    "jobs_enqueued_total", "jobs_success", "jobs_error", "throughput_jobs_per_sec",
    # cache
    "cache_ok", "cache_fail", "cache_hit_ratio",
    # trigger
    "trigger_seen", "trigger_enqueued", "trigger_enqueue_rate",
    "trigger_changed_scene", "trigger_expired_total", "trigger_periodic",
    # optional
    "confidence_stats_per_label",
]

def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        for r in rows:
            w.writerow([r.get(col, "") for col in CSV_HEADER])


def plot_metric(rows: List[Dict[str, Any]], metric: str, out: Optional[str] = None) -> None:
    """Plot a single metric series from collected rows."""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib is required to plot metrics. Install it with: pip install matplotlib")
        return

    xs = [r.get("t", 0) for r in rows]
    ys = [
        (float(r.get(metric)) if r.get(metric) is not None and r.get(metric) != "" else float("nan"))
        for r in rows
    ]
    plt.figure(figsize=(8, 4))
    plt.plot(xs, ys, marker="o")
    plt.xlabel("seconds")
    plt.ylabel(metric)
    plt.title(f"Metric: {metric}")
    plt.grid(True)
    if out:
        plt.savefig(out)
        print(f"saved plot to {out}")
    else:
        plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8008", help="slowpath base URL")
    ap.add_argument("--secs", type=int, default=60, help="duration to collect (seconds)")
    ap.add_argument("--out", default="metrics.csv", help="output CSV path")
    ap.add_argument("--plot", action="store_true", help="plot a metric after collection")
    ap.add_argument("--metric", default="queue_depth", help="metric name to plot when --plot is used")
    ap.add_argument("--plot-out", default=None, help="output image path for the plot (PNG)")
    args = ap.parse_args()

    url = f"{args.base}/metrics"
    rows = collect_metrics(url, args.secs)
    write_csv(args.out, rows)
    print(f"Wrote {len(rows)} rows to {args.out}")
    if args.plot:
        plot_metric(rows, args.metric, out=args.plot_out)


if __name__ == "__main__":
    main()
