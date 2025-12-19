"""
Analyze Real-Time Benchmark Results

Compares all benchmark approaches and generates comparison tables.

Usage:
    python analyze_results.py
"""

import json
import numpy as np
from pathlib import Path
from tabulate import tabulate


def load_benchmark_results():
    """Load all benchmark result files."""
    results = {}
    result_files = list(Path(".").glob("realtime_benchmark_*.json"))

    if not result_files:
        print("⚠️  No benchmark results found")
        print("   Run benchmark_realtime_system.py with different --approach options first")
        return {}

    for result_file in result_files:
        approach = result_file.stem.replace("realtime_benchmark_", "")
        with open(result_file, 'r') as f:
            results[approach] = json.load(f)

    return results


def print_comparison_table(results):
    """Print comparison table for all approaches."""
    if not results:
        return

    print("\n" + "="*120)
    print("REAL-TIME PERFORMANCE COMPARISON")
    print("="*120 + "\n")

    # Prepare table data
    headers = [
        "Approach",
        "Coverage\n(%)",
        "Dropped\n(%)",
        "VLM Latency\n(ms)",
        "Speedup\nvs Baseline",
        "Accuracy\n(%)",
        "F1\nScore",
        "Cache Hit\n(%)",
        "Effective\nFPS",
        "Tracks/sec"
    ]

    table_data = []

    # Get baseline latency for speedup calculation
    baseline_latency = results.get('vanilla', {}).get('metrics', {}).get('vlm_latency_mean_ms', 1)

    for approach, data in sorted(results.items()):
        metrics = data['metrics']

        speedup = baseline_latency / metrics['vlm_latency_mean_ms'] if metrics['vlm_latency_mean_ms'] > 0 else 0

        row = [
            approach.upper(),
            f"{metrics['evaluation_coverage_pct']:.1f}",
            f"{metrics['drop_rate_pct']:.1f}",
            f"{metrics['vlm_latency_mean_ms']:.0f}",
            f"{speedup:.2f}×",
            f"{metrics['accuracy_pct']:.1f}",
            f"{metrics['f1_score']:.3f}",
            f"{metrics.get('cache_hit_rate_pct', 0):.1f}",
            f"{metrics['effective_fps']:.1f}",
            f"{metrics['throughput_tracks_per_sec']:.1f}"
        ]
        table_data.append(row)

    print(tabulate(table_data, headers=headers, tablefmt="grid"))


def print_detailed_metrics(results):
    """Print detailed metrics for each approach."""
    if not results:
        return

    print("\n" + "="*120)
    print("DETAILED METRICS BY APPROACH")
    print("="*120)

    for approach, data in sorted(results.items()):
        print(f"\n{'─'*120}")
        print(f"APPROACH: {approach.upper()}")
        print(f"{'─'*120}")

        metrics = data['metrics']

        print(f"\n📊 Coverage & Throughput:")
        print(f"   Total tracks detected:  {metrics['total_tracks_detected']}")
        print(f"   Tracks evaluated:       {metrics['tracks_evaluated']} ({metrics['evaluation_coverage_pct']:.1f}%)")
        print(f"   Tracks dropped:         {metrics['tracks_dropped']} ({metrics['drop_rate_pct']:.1f}%)")
        print(f"   Effective FPS:          {metrics['effective_fps']:.1f}")
        print(f"   Throughput:             {metrics['throughput_tracks_per_sec']:.1f} tracks/sec")

        print(f"\n⏱️  Latency Distribution:")
        print(f"   VLM Mean:     {metrics['vlm_latency_mean_ms']:.0f} ms")
        print(f"   VLM Median:   {metrics['vlm_latency_median_ms']:.0f} ms")
        print(f"   VLM P95:      {metrics['vlm_latency_p95_ms']:.0f} ms")
        print(f"   VLM Std Dev:  {metrics['vlm_latency_std_ms']:.0f} ms")
        print(f"   Frame Time:   {metrics['frame_latency_mean_ms']:.1f} ms/frame")

        print(f"\n📈 Queue Statistics:")
        print(f"   Average depth: {metrics['queue_depth_mean']:.1f}")
        print(f"   Max depth:     {metrics['queue_depth_max']} (limit: 50)")

        print(f"\n🎯 Accuracy Metrics:")
        print(f"   Accuracy:   {metrics['accuracy_pct']:.1f}%")
        print(f"   Precision:  {metrics['precision']:.3f}")
        print(f"   Recall:     {metrics['recall']:.3f}")
        print(f"   F1 Score:   {metrics['f1_score']:.3f}")

        if metrics.get('cache_hit_rate_pct', 0) > 0:
            print(f"\n💾 Cache Performance:")
            print(f"   Hit rate:  {metrics['cache_hit_rate_pct']:.1f}%")
            print(f"   Hits:      {metrics['cache_hits']}")
            print(f"   Misses:    {metrics['cache_misses']}")

            if 'cache_stats' in data and data['cache_stats']:
                cache_stats = data['cache_stats']
                print(f"   Cache size: {cache_stats.get('cache_size', 'N/A')}")
                if 'avg_cache_overhead_ms' in cache_stats:
                    print(f"   Avg overhead: {cache_stats['avg_cache_overhead_ms']:.1f} ms")


def print_summary_analysis(results):
    """Print summary analysis and recommendations."""
    if not results:
        return

    print("\n" + "="*120)
    print("SUMMARY ANALYSIS")
    print("="*120)

    baseline_metrics = results.get('vanilla', {}).get('metrics', {})

    if not baseline_metrics:
        print("\n⚠️  No baseline (vanilla) results found for comparison")
        return

    baseline_latency = baseline_metrics.get('vlm_latency_mean_ms', 0)
    baseline_coverage = baseline_metrics.get('evaluation_coverage_pct', 0)
    baseline_accuracy = baseline_metrics.get('accuracy_pct', 100)

    print(f"\n📊 Baseline Performance:")
    print(f"   Latency:  {baseline_latency:.0f} ms/track")
    print(f"   Coverage: {baseline_coverage:.1f}%")
    print(f"   Accuracy: {baseline_accuracy:.1f}%")

    print(f"\n🎯 Best Performers:")

    # Find best speedup
    best_speedup_approach = None
    best_speedup = 0
    for approach, data in results.items():
        if approach == 'vanilla':
            continue
        latency = data['metrics']['vlm_latency_mean_ms']
        speedup = baseline_latency / latency if latency > 0 else 0
        if speedup > best_speedup:
            best_speedup = speedup
            best_speedup_approach = approach

    if best_speedup_approach:
        best_metrics = results[best_speedup_approach]['metrics']
        print(f"\n   Best Speedup: {best_speedup_approach.upper()} ({best_speedup:.2f}×)")
        print(f"      Latency:  {best_metrics['vlm_latency_mean_ms']:.0f} ms")
        print(f"      Coverage: {best_metrics['evaluation_coverage_pct']:.1f}%")
        print(f"      Accuracy: {best_metrics['accuracy_pct']:.1f}%")

    # Find best coverage
    best_coverage_approach = max(results.items(),
                                  key=lambda x: x[1]['metrics']['evaluation_coverage_pct'])
    best_coverage = best_coverage_approach[1]['metrics']['evaluation_coverage_pct']
    print(f"\n   Best Coverage: {best_coverage_approach[0].upper()} ({best_coverage:.1f}%)")

    # Find best accuracy
    best_accuracy_approach = max(results.items(),
                                  key=lambda x: x[1]['metrics']['accuracy_pct'])
    best_accuracy = best_accuracy_approach[1]['metrics']['accuracy_pct']
    print(f"\n   Best Accuracy: {best_accuracy_approach[0].upper()} ({best_accuracy:.1f}%)")

    # Recommendations
    print(f"\n💡 Recommendations:")

    for approach, data in results.items():
        if approach == 'vanilla':
            continue

        metrics = data['metrics']
        speedup = baseline_latency / metrics['vlm_latency_mean_ms'] if metrics['vlm_latency_mean_ms'] > 0 else 0
        coverage = metrics['evaluation_coverage_pct']
        accuracy = metrics['accuracy_pct']

        if speedup >= 1.5 and accuracy >= 85 and coverage >= 80:
            print(f"\n   ✅ {approach.upper()}: Excellent performance")
            print(f"      {speedup:.2f}× speedup with {accuracy:.0f}% accuracy and {coverage:.0f}% coverage")
            print(f"      → Recommended for production use")

        elif speedup >= 1.3 and accuracy >= 75:
            print(f"\n   ✓ {approach.upper()}: Good performance")
            print(f"      {speedup:.2f}× speedup with {accuracy:.0f}% accuracy")
            print(f"      → Consider for production with accuracy tolerance")

        elif speedup < 1.1:
            print(f"\n   ⚠️  {approach.upper()}: Minimal speedup")
            print(f"      Only {speedup:.2f}× faster")
            print(f"      → Needs optimization or may not be worth complexity")

        if accuracy < 75:
            print(f"\n   ⚠️  {approach.upper()}: Low accuracy ({accuracy:.0f}%)")
            print(f"      → Below 75% threshold, needs improvement")

        if coverage < 60:
            print(f"\n   ⚠️  {approach.upper()}: Low coverage ({coverage:.0f}%)")
            print(f"      → Still dropping too many tracks, needs faster VLM")

    # Real-time capability
    print(f"\n🎬 Real-Time Capability (30 FPS target):")
    for approach, data in results.items():
        metrics = data['metrics']
        effective_fps = metrics['effective_fps']
        status = "✅" if effective_fps >= 20 else "⚠️" if effective_fps >= 15 else "❌"
        print(f"   {status} {approach.upper()}: {effective_fps:.1f} FPS")

    print(f"\n{'='*120}\n")


def main():
    print("="*120)
    print("BENCHMARK RESULTS ANALYSIS")
    print("="*120)

    # Load results
    results = load_benchmark_results()

    if not results:
        return

    print(f"\nLoaded {len(results)} benchmark result(s):")
    for approach in sorted(results.keys()):
        print(f"   - {approach}")

    # Generate analysis
    print_comparison_table(results)
    print_detailed_metrics(results)
    print_summary_analysis(results)

    print("✅ Analysis complete")
    print("\nTo add more results, run:")
    print("   python benchmark_realtime_system.py --approach <vanilla|cached|int8|combined>")
    print("="*120 + "\n")


if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        if 'tabulate' in str(e):
            print("⚠️  Missing dependency: tabulate")
            print("   Install with: pip install tabulate")
        else:
            raise
