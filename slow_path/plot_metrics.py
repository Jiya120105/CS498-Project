#!/usr/bin/env python3
import argparse, csv, os, json
import matplotlib.pyplot as plt

def read_csv(path):
    rows = []
    with open(path, newline="") as f:
        for _, row in enumerate(csv.DictReader(f)):
            d = {}
            for k, v in row.items():
                if v == "" or v is None:
                    d[k] = None
                else:
                    # try number, else keep as string
                    try:
                        d[k] = float(v)
                    except ValueError:
                        d[k] = v
            rows.append(d)
    return rows

def series(rows, key):
    xs = [r["t"] for r in rows if r.get("t") is not None and r.get(key) is not None]
    ys = [r.get(key) for r in rows if r.get("t") is not None and r.get(key) is not None]
    return xs, ys

def _filter_xy(xs, ys):
    """Return lists with only pairs where both x and y are valid numbers."""
    fx, fy = [], []
    for x, y in zip(xs, ys):
        if x is None or y is None:
            continue
        try:
            fx.append(float(x))
            fy.append(float(y))
        except (TypeError, ValueError):
            continue
    return fx, fy

def save_line(xs, ys, title, ylabel, out):
    plt.figure()
    fx, fy = _filter_xy(xs, ys)
    if not fx:  # nothing valid to plot
        plt.close(); return
    plt.plot(fx, fy)  # no explicit colors/styles
    plt.title(title)
    plt.xlabel("time (s)")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()

def save_multi(xs, series_list, title, ylabel, out):
    plt.figure(figsize=(8,3))
    plotted = False
    for name, ys, style in series_list:
        fx, fy = _filter_xy(xs, ys)
        if not fx:
            continue
        plotted = True
        plt.plot(fx, fy, style, label=name)
    if not plotted:
        plt.close(); return
    plt.title(title)
    plt.xlabel("time (s)")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()

def parse_confidence_json(rows):
    if not rows or "confidence_stats_per_label" not in rows[0]:
        return [], {}
    times = []
    per_label = {}
    for r in rows:
        times.append(r.get("t"))
        raw = r.get("confidence_stats_per_label")
        try:
            obj = json.loads(raw) if isinstance(raw, str) else (raw or {})
        except Exception:
            obj = {}
        # Ensure all seen labels get a value this row (even None)
        for lbl in set(per_label.keys()) | set(obj.keys()):
            per_label.setdefault(lbl, {"p50": [], "p95": [], "p99": [], "max": []})
            s = obj.get(lbl) or {}
            per_label[lbl]["p50"].append(s.get("p50"))
            per_label[lbl]["p95"].append(s.get("p95"))
            per_label[lbl]["p99"].append(s.get("p99"))
            per_label[lbl]["max"].append(s.get("max"))
    return times, per_label

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="metrics.csv", help="metrics CSV from collect_metrics.py")
    ap.add_argument("--out-dir", default="plots", help="directory to write plots")
    ap.add_argument("--show", action="store_true", help="show plots interactively after saving")
    ap.add_argument("--labels", default="", help="comma-separated labels to plot confidences for (default: all)")
    args = ap.parse_args()

    rows = read_csv(args.csv)
    out = args.out_dir
    os.makedirs(out, exist_ok=True)

    # Inference latency (p50/p95/p99/max)
    t_series = [r["t"] for r in rows if r.get("t") is not None]
    ip50_x, ip50 = series(rows, "infer_p50")
    ip95_x, ip95 = series(rows, "infer_p95")
    ip99_x, ip99 = series(rows, "infer_p99")
    ipmax_x, ipmax = series(rows, "infer_max")
    if ip50 or ip95 or ip99 or ipmax:
        xs = ip50_x or ip95_x or ip99_x or ipmax_x or t_series
        series_list = []
        if ip50: series_list.append(("p50", ip50, "-o"))
        if ip95: series_list.append(("p95", ip95, "-s"))
        if ip99: series_list.append(("p99", ip99, "-^"))
        if ipmax: series_list.append(("max", ipmax, "--"))
        save_multi(xs, series_list, "Inference Latency (ms)", "ms", os.path.join(out, "infer_latency.png"))

    # Queue depth & results size
    qx, qy = series(rows, "queue_depth")
    rx, ry = series(rows, "results_size")
    if qy or ry:
        xs = qx or rx
        serie = []
        if qy: serie.append(("queue_depth", qy, "-o"))
        if ry: serie.append(("results_size", ry, "-s"))
        save_multi(xs, serie, "Queue & Results", "count", os.path.join(out, "queue_results.png"))

    # Throughput & job counts
    tx, throughput = series(rows, "throughput_jobs_per_sec")
    jx, jobs_success = series(rows, "jobs_success")
    _, jobs_error = series(rows, "jobs_error")
    if throughput or jobs_success or jobs_error:
        xs = tx or jx
        serie = []
        if throughput: serie.append(("throughput_jobs_per_sec", throughput, "-o"))
        if jobs_success: serie.append(("jobs_success", jobs_success, "-s"))
        if jobs_error: serie.append(("jobs_error", jobs_error, "-x"))
        save_multi(xs, serie, "Throughput & Job Counts", "value", os.path.join(out, "throughput_jobs.png"))

    # Cache hit ratio
    chx, ch = series(rows, "cache_hit_ratio")
    if ch:
        save_line(chx, ch, "Cache Hit Ratio", "fraction", os.path.join(out, "cache_hit_ratio.png"))

    # Cache post counters (ok/fail)
    okx, oky = series(rows, "cache_ok")
    flx, fly = series(rows, "cache_fail")
    if oky or fly:
        xs = okx or flx
        serie = []
        if oky: serie.append(("cache_ok", oky, "-o"))
        if fly: serie.append(("cache_fail", fly, "-x"))
        save_multi(xs, serie, "Cache Posts (cumulative)", "count", os.path.join(out, "cache_posts.png"))

    # Trigger enqueue rate
    ex, er = series(rows, "trigger_enqueue_rate")
    if er:
        save_line(ex, er, "Trigger Enqueue Rate", "enq_rate", os.path.join(out, "enqueue_rate.png"))

    # Trigger counters: changed_scene / expired_total / periodic (cumulative)
    csx, csy = series(rows, "trigger_changed_scene")
    etx, ety = series(rows, "trigger_expired_total")
    px, py = series(rows, "trigger_periodic")
    if csy or ety or py:
        xs = csx or etx or px
        serie = []
        if csy: serie.append(("changed_scene", csy, "-o"))
        if ety: serie.append(("ttl_expired_total", ety, "-s"))
        if py:  serie.append(("periodic", py, "-^"))
        save_multi(xs, serie, "Trigger Counters (cumulative)", "count", os.path.join(out, "trigger_counters.png"))

    # Optional: per-label confidence percentiles
    txs, per_label = parse_confidence_json(rows)
    if txs and per_label:
        requested = set([s for s in args.labels.split(",") if s.strip()]) if args.labels else None
        for lbl, stats in per_label.items():
            if requested and lbl not in requested:
                continue
            series_list = []
            if any(v is not None for v in stats["p50"]): series_list.append(("p50", stats["p50"], "-o"))
            if any(v is not None for v in stats["p95"]): series_list.append(("p95", stats["p95"], "-s"))
            if any(v is not None for v in stats["p99"]): series_list.append(("p99", stats["p99"], "-^"))
            if any(v is not None for v in stats["max"]): series_list.append(("max", stats["max"], "--"))
            if series_list:
                save_multi(txs, series_list, f"Confidence Percentiles — {lbl}", "confidence", os.path.join(out, f"conf_{lbl}.png"))

    print("Saved plots to:", out)
    if args.show:
        try:
            plt.show()
        except Exception:
            print("Unable to show interactive plot (headless). Files saved.")

if __name__ == "__main__":
    main()
