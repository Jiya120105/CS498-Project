#!/usr/bin/env python3
"""
This script runs multiple simulations and generates
figures and tables showing semantic cache performance.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from cache_simulation import simulate_cache
import time

def run_comprehensive_benchmarks():
    """Run multiple benchmark scenarios and collect results."""

    print("="*70)
    print("SEMANTIC CACHE - PERFORMANCE REPORT")
    print("="*70)
    print()

    results = {}

    # Scenario 1: Varying frame counts (scalability)
    print(" Benchmark 1: Scalability Test (varying frame counts)")
    print("-" * 70)
    frame_counts = [50, 100, 200, 300]
    scalability_results = []

    for frames in frame_counts:
        print(f"\n  Testing with {frames} frames...")
        cache_stats = simulate_cache(frames=frames, use_cache=True)
        no_cache_stats = simulate_cache(frames=frames, use_cache=False)

        cache_avg_time = np.mean(cache_stats["total_processing_time"])
        no_cache_avg_time = np.mean(no_cache_stats["total_processing_time"])
        speedup = no_cache_avg_time / cache_avg_time
        final_hit_rate = cache_stats['hit_rate'][-1] if cache_stats['hit_rate'] else 0

        scalability_results.append({
            "frames": frames,
            "cache_time_ms": cache_avg_time * 1000,
            "no_cache_time_ms": no_cache_avg_time * 1000,
            "speedup": speedup,
            "hit_rate": final_hit_rate,
            "max_cache_size": max(cache_stats['cache_size'])
        })

        print(f"     Speedup: {speedup:.2f}x, Hit rate: {final_hit_rate:.1%}")

    results["scalability"] = scalability_results

    # Scenario 2: Detailed 200-frame analysis (for report)
    print("\n\n Benchmark 2: Detailed Performance Analysis (200 frames)")
    print("-" * 70)

    cache_stats = simulate_cache(frames=200, use_cache=True)
    no_cache_stats = simulate_cache(frames=200, use_cache=False)

    # Calculate detailed metrics
    cache_times = cache_stats["total_processing_time"]
    no_cache_times = no_cache_stats["total_processing_time"]

    detailed = {
        "frames": 200,
        "cache": {
            "mean_ms": np.mean(cache_times) * 1000,
            "p50_ms": np.percentile(cache_times, 50) * 1000,
            "p95_ms": np.percentile(cache_times, 95) * 1000,
            "p99_ms": np.percentile(cache_times, 99) * 1000,
            "max_ms": np.max(cache_times) * 1000,
            "min_ms": np.min(cache_times) * 1000,
        },
        "no_cache": {
            "mean_ms": np.mean(no_cache_times) * 1000,
            "p50_ms": np.percentile(no_cache_times, 50) * 1000,
            "p95_ms": np.percentile(no_cache_times, 95) * 1000,
            "p99_ms": np.percentile(no_cache_times, 99) * 1000,
            "max_ms": np.max(no_cache_times) * 1000,
            "min_ms": np.min(no_cache_times) * 1000,
        },
        "speedup": {
            "mean": np.mean(no_cache_times) / np.mean(cache_times),
            "p50": np.percentile(no_cache_times, 50) / np.percentile(cache_times, 50),
            "p95": np.percentile(no_cache_times, 95) / np.percentile(cache_times, 95),
            "p99": np.percentile(no_cache_times, 99) / np.percentile(cache_times, 99),
        },
        "cache_performance": {
            "final_hit_rate": cache_stats['hit_rate'][-1] if cache_stats['hit_rate'] else 0,
            "total_hits": sum(cache_stats['hits']),
            "total_misses": sum(cache_stats['misses']),
            "max_cache_size": max(cache_stats['cache_size']),
            "total_time_saved_s": sum(cache_stats['cache_benefit']),
        }
    }

    results["detailed"] = detailed

    # Print summary table
    print("\n  Latency Comparison (ms):")
    print(f"    Metric      | With Cache | Without Cache | Speedup")
    print(f"    ------------|------------|---------------|--------")
    print(f"    Mean        | {detailed['cache']['mean_ms']:10.2f} | {detailed['no_cache']['mean_ms']:13.2f} | {detailed['speedup']['mean']:6.2f}x")
    print(f"    p50         | {detailed['cache']['p50_ms']:10.2f} | {detailed['no_cache']['p50_ms']:13.2f} | {detailed['speedup']['p50']:6.2f}x")
    print(f"    p95         | {detailed['cache']['p95_ms']:10.2f} | {detailed['no_cache']['p95_ms']:13.2f} | {detailed['speedup']['p95']:6.2f}x")
    print(f"    p99         | {detailed['cache']['p99_ms']:10.2f} | {detailed['no_cache']['p99_ms']:13.2f} | {detailed['speedup']['p99']:6.2f}x")

    print("\n  Cache Effectiveness:")
    perf = detailed['cache_performance']
    print(f"    Hit Rate:       {perf['final_hit_rate']:.1%}")
    print(f"    Total Hits:     {perf['total_hits']}")
    print(f"    Total Misses:   {perf['total_misses']}")
    print(f"    Max Cache Size: {perf['max_cache_size']} entries")
    print(f"    Time Saved:     {perf['total_time_saved_s']:.2f}s")

    # Generate visualizations
    print("\n\n Generating visualizations...")
    print("-" * 70)

    generate_report_figures(cache_stats, no_cache_stats, scalability_results)

    # Save raw data
    with open('semantic_cache/results/metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("   Saved raw metrics to: semantic_cache/results/metrics.json")

    # Generate LaTeX table
    generate_latex_tables(results)

    print("\n" + "="*70)
    print("REPORT GENERATION COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  1. semantic_cache/results/metrics.json - Raw data")
    print("  2. semantic_cache/results/latency_comparison.png - Latency comparison")
    print("  3. semantic_cache/results/scalability_analysis.png - Scalability analysis")
    print("  4. semantic_cache/results/cache_behavior.png - Cache behavior over time")
    print("  5. semantic_cache/results/tables.tex - LaTeX tables")
    print()

    return results

def generate_report_figures(cache_stats, no_cache_stats, scalability_results):
    """Generate publication-quality figures for the report."""

    # Figure 1: Latency Comparison Over Time
    plt.figure(figsize=(8, 4))

    frames = cache_stats["frame"]
    cache_times_ms = [t * 1000 for t in cache_stats["total_processing_time"]]
    no_cache_times_ms = [t * 1000 for t in no_cache_stats["total_processing_time"]]

    plt.plot(frames, cache_times_ms, label="With Cache", linewidth=1.5, alpha=0.8)
    plt.plot(frames, no_cache_times_ms, label="Without Cache", linewidth=1.5, alpha=0.8)
    plt.xlabel("Frame Number")
    plt.ylabel("Processing Time (ms)")
    plt.title("Per-Frame Latency Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('semantic_cache/results/latency_comparison.png', dpi=300, bbox_inches='tight')
    print("   Generated: semantic_cache/results/latency_comparison.png")
    plt.close()

    # Figure 2: Scalability
    plt.figure(figsize=(12, 4))

    frame_counts = [r["frames"] for r in scalability_results]
    speedups = [r["speedup"] for r in scalability_results]
    hit_rates = [r["hit_rate"] * 100 for r in scalability_results]

    plt.subplot(1, 3, 1)
    plt.plot(frame_counts, speedups, marker='o', linewidth=2, markersize=8)
    plt.xlabel("Number of Frames")
    plt.ylabel("Speedup Factor")
    plt.title("Scalability: Speedup vs. Frame Count")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.plot(frame_counts, hit_rates, marker='s', linewidth=2, markersize=8, color='green')
    plt.xlabel("Number of Frames")
    plt.ylabel("Hit Rate (%)")
    plt.title("Cache Hit Rate vs. Frame Count")
    plt.ylim([0, 100])
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    cache_times = [r["cache_time_ms"] for r in scalability_results]
    no_cache_times = [r["no_cache_time_ms"] for r in scalability_results]
    plt.plot(frame_counts, cache_times, marker='o', label="With Cache", linewidth=2)
    plt.plot(frame_counts, no_cache_times, marker='s', label="Without Cache", linewidth=2)
    plt.xlabel("Number of Frames")
    plt.ylabel("Avg. Processing Time (ms)")
    plt.title("Processing Time vs. Frame Count")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('semantic_cache/results/scalability_analysis.png', dpi=300, bbox_inches='tight')
    print("   Generated: semantic_cache/results/scalability_analysis.png")
    plt.close()

    # Figure 3: Cache Behavior
    plt.figure(figsize=(12, 8))

    # Hit rate over time
    plt.subplot(2, 2, 1)
    hit_rate_percent = cache_stats["hit_rate_percent"]
    plt.plot(frames, hit_rate_percent, linewidth=2, color='green')
    plt.xlabel("Frame Number")
    plt.ylabel("Hit Rate (%)")
    plt.title("Cache Hit Rate Over Time")
    plt.ylim([0, 100])
    plt.grid(True, alpha=0.3)

    # Cache size over time
    plt.subplot(2, 2, 2)
    plt.plot(frames, cache_stats["cache_size"], linewidth=2, color='blue')
    plt.xlabel("Frame Number")
    plt.ylabel("Cache Size (entries)")
    plt.title("Cache Size Over Time")
    plt.grid(True, alpha=0.3)

    # Cumulative time saved
    plt.subplot(2, 2, 3)
    cumulative_saved = np.cumsum(cache_stats["cache_benefit"])
    plt.plot(frames, cumulative_saved, linewidth=2, color='purple')
    plt.xlabel("Frame Number")
    plt.ylabel("Cumulative Time Saved (s)")
    plt.title("Cumulative Time Saved by Cache")
    plt.grid(True, alpha=0.3)

    # Hit/miss pattern visualization
    plt.subplot(2, 2, 4)
    cum_hits = cache_stats["cumulative_real_hits"]
    cum_misses = cache_stats["cumulative_real_misses"]
    plt.plot(frames, cum_hits, label="Cumulative Hits", linewidth=2)
    plt.plot(frames, cum_misses, label="Cumulative Misses", linewidth=2)
    plt.xlabel("Frame Number")
    plt.ylabel("Count")
    plt.title("Cumulative Hits vs. Misses")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('semantic_cache/results/cache_behavior.png', dpi=300, bbox_inches='tight')
    print("   Generated: semantic_cache/results/cache_behavior.png")
    plt.close()

def generate_latex_tables(results):
    """Generate LaTeX tables for the report."""

    latex = []

    # Table 1: Latency Comparison
    latex.append("% Table 1: Latency Comparison")
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append("\\caption{Per-Frame Latency Comparison (200 frames)}")
    latex.append("\\label{tab:latency}")
    latex.append("\\begin{tabular}{lrrr}")
    latex.append("\\hline")
    latex.append("\\textbf{Metric} & \\textbf{With Cache (ms)} & \\textbf{Without Cache (ms)} & \\textbf{Speedup} \\\\")
    latex.append("\\hline")

    d = results["detailed"]
    latex.append(f"Mean  & {d['cache']['mean_ms']:.2f} & {d['no_cache']['mean_ms']:.2f} & {d['speedup']['mean']:.2f}$\\times$ \\\\")
    latex.append(f"p50   & {d['cache']['p50_ms']:.2f} & {d['no_cache']['p50_ms']:.2f} & {d['speedup']['p50']:.2f}$\\times$ \\\\")
    latex.append(f"p95   & {d['cache']['p95_ms']:.2f} & {d['no_cache']['p95_ms']:.2f} & {d['speedup']['p95']:.2f}$\\times$ \\\\")
    latex.append(f"p99   & {d['cache']['p99_ms']:.2f} & {d['no_cache']['p99_ms']:.2f} & {d['speedup']['p99']:.2f}$\\times$ \\\\")
    latex.append("\\hline")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    latex.append("")

    # Table 2: Cache Effectiveness
    latex.append("% Table 2: Cache Effectiveness Metrics")
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append("\\caption{Semantic Cache Performance Metrics}")
    latex.append("\\label{tab:cache-metrics}")
    latex.append("\\begin{tabular}{lr}")
    latex.append("\\hline")
    latex.append("\\textbf{Metric} & \\textbf{Value} \\\\")
    latex.append("\\hline")

    perf = d["cache_performance"]
    latex.append(f"Hit Rate & {perf['final_hit_rate']*100:.1f}\\% \\\\")
    latex.append(f"Total Hits & {perf['total_hits']} \\\\")
    latex.append(f"Total Misses & {perf['total_misses']} \\\\")
    latex.append(f"Max Cache Size & {perf['max_cache_size']} entries \\\\")
    latex.append(f"Total Time Saved & {perf['total_time_saved_s']:.2f}s \\\\")
    latex.append("\\hline")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    latex.append("")

    # Table 3: Scalability
    latex.append("% Table 3: Scalability Results")
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append("\\caption{Scalability Analysis}")
    latex.append("\\label{tab:scalability}")
    latex.append("\\begin{tabular}{rrrr}")
    latex.append("\\hline")
    latex.append("\\textbf{Frames} & \\textbf{Speedup} & \\textbf{Hit Rate} & \\textbf{Cache Size} \\\\")
    latex.append("\\hline")

    for r in results["scalability"]:
        latex.append(f"{r['frames']} & {r['speedup']:.2f}$\\times$ & {r['hit_rate']*100:.1f}\\% & {r['max_cache_size']} \\\\")

    latex.append("\\hline")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")

    # Save to file
    with open('semantic_cache/results/tables.tex', 'w') as f:
        f.write('\n'.join(latex))

    print("   Generated: semantic_cache/results/tables.tex")

if __name__ == "__main__":
    results = run_comprehensive_benchmarks()
    print("\nFiles are ready!")
