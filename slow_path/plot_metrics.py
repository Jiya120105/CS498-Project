#!/usr/bin/env python3
import argparse, csv
import matplotlib.pyplot as plt

def read_csv(path):
    rows = []
    with open(path, newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            # convert numeric fields where possible
            d = {}
            for k, v in row.items():
                if v == "" or v is None:
                    d[k] = None
                else:
                    try:
                        d[k] = float(v)
                    except ValueError:
                        d[k] = v
            rows.append(d)
    return rows

def series(rows, key):
    xs = [r["t"] for r in rows if r["t"] is not None and r.get(key) is not None]
    ys = [r.get(key) for r in rows if r["t"] is not None and r.get(key) is not None]
    return xs, ys

def save_line(xs, ys, title, ylabel, out):
    plt.figure()
    plt.plot(xs, ys)  # no colors/styles specified
    plt.title(title)
    plt.xlabel("time (s)")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="metrics.csv", help="metrics CSV from collect_metrics.py")
    args = ap.parse_args()

    rows = read_csv(args.csv)

    # Latency percentiles over time
    xs, ys = series(rows, "infer_p50")
    if xs:
        save_line(xs, ys, "Inference Latency p50", "milliseconds", "latency_p50.png")
    xs, ys = series(rows, "infer_p95")
    if xs:
        save_line(xs, ys, "Inference Latency p95", "milliseconds", "latency_p95.png")
    xs, ys = series(rows, "infer_max")
    if xs:
        save_line(xs, ys, "Inference Latency Max", "milliseconds", "latency_max.png")

    # Queue depth over time
    xs, ys = series(rows, "queue_depth")
    if xs:
        save_line(xs, ys, "Queue Depth", "jobs", "queue_depth.png")

    # Trigger enqueue rate over time
    xs, ys = series(rows, "trigger_enqueue_rate")
    if xs:
        save_line(xs, ys, "Trigger Enqueue Rate", "enqueued / seen", "enqueue_rate.png")

    # Cache cumulative posted OK over time
    xs, ys = series(rows, "cache_ok")
    if xs:
        save_line(xs, ys, "Cache Posts (OK cumulative)", "count", "cache_ok.png")

    # Jobs enqueued total over time
    xs, ys = series(rows, "jobs_enqueued_total")
    if xs:
        save_line(xs, ys, "Jobs Enqueued (cumulative)", "count", "jobs_enqueued_total.png")

    print("Saved plots: latency_p50.png, latency_p95.png, latency_max.png, queue_depth.png, enqueue_rate.png, cache_ok.png, jobs_enqueued_total.png (created where data existed).")

if __name__ == "__main__":
    main()
