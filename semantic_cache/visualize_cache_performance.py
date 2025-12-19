import argparse
from cache_simulation import simulate_cache
import matplotlib.pyplot as plt
import numpy as np

def visualize_cache(cache_stats, no_cache_stats, save_path="cache_performance.png"):
    plt.figure(figsize=(15, 10))

    # Plot 1: Processing Times Comparison
    plt.subplot(2, 2, 1)
    frames = cache_stats["frame"]
    plt.plot(frames, cache_stats["total_processing_time"], label="With Cache")
    plt.plot(frames, no_cache_stats["total_processing_time"], label="Without Cache")
    plt.xlabel("Frame")
    plt.ylabel("Processing Time (s)")
    plt.title("Total Processing Time Comparison")
    plt.legend()

    # Plot 2: Cache Performance
    plt.subplot(2, 2, 2)
    frames = cache_stats["frame"]
    # Use cumulative real track hits/misses (more accurate for real tracks only)
    if "cumulative_real_hits" in cache_stats and "cumulative_real_misses" in cache_stats:
        cum_hits = cache_stats["cumulative_real_hits"]
        cum_misses = cache_stats["cumulative_real_misses"]
    else:
        # Fallback to calculating from per-frame stats
        cum_hits = np.cumsum(cache_stats["hits"])
        cum_misses = np.cumsum(cache_stats["misses"])
    
    # Use the cumulative hit rate from stats (calculated for real tracks only)
    hit_rates = cache_stats["hit_rate_percent"]
    
    # Debug: Print first few hit rates to verify
    if len(hit_rates) > 0:
        print(f"\nDebug: First 10 hit rates: {hit_rates[:10]}")
        print(f"Debug: First 10 frames: {frames[:10]}")

    ax = plt.gca()
    ax.plot(frames, cum_hits, label="Cumulative Hits", color="tab:blue")
    ax.plot(frames, cum_misses, label="Cumulative Misses", color="tab:orange")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Cumulative Count")
    ax.set_title("Cache Performance Metrics (Real Tracks Only)")

    ax2 = ax.twinx()
    ax2.plot(frames, hit_rates, label="Hit Rate (%)", linestyle="--", color="green", linewidth=2)
    ax2.set_ylabel("Hit Rate (%)", color="green")
    ax2.tick_params(axis='y', labelcolor="green")
    ax2.set_ylim([0, 100])

    # Combine legends
    lines, labels_ = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels_ + labels2, loc="upper left")

    # Plot 3: Time Saved by Cache
    plt.subplot(2, 2, 3)
    cumulative_benefit = np.cumsum(cache_stats["cache_benefit"])
    plt.plot(frames, cumulative_benefit, label="Cumulative Time Saved")
    plt.xlabel("Frame")
    plt.ylabel("Time Saved (s)")
    plt.title("Cumulative Time Saved by Cache")
    plt.legend()

    # Plot 4: Path Time Distributions
    plt.subplot(2, 2, 4)
    plt.boxplot([
        cache_stats["fast_path_time"],
        cache_stats["slow_path_time"],
        no_cache_stats["slow_path_time"]
    ], tick_labels=["Fast Path", "Slow Path\n(with cache)", "Slow Path\n(no cache)"])
    plt.ylabel("Time (s)")
    plt.title("Processing Time Distribution")

    plt.tight_layout()

    # Save the plot
    plt.savefig(save_path)
    print(f"Visualization saved to {save_path}")

    # Show the plot
    plt.show()

def main(frames=100, mode="visualize", save_path="cache_performance.png"):
    print("Running simulation with cache...")
    cache_stats = simulate_cache(frames=frames, use_cache=True)
    
    print("Running simulation without cache...")
    no_cache_stats = simulate_cache(frames=frames, use_cache=False)

    if mode == "visualize":
        visualize_cache(cache_stats, no_cache_stats, save_path)
    
    # Calculate and print performance metrics
    cache_avg_time = np.mean(cache_stats["total_processing_time"])
    no_cache_avg_time = np.mean(no_cache_stats["total_processing_time"])
    time_saved = no_cache_avg_time - cache_avg_time
    speedup = no_cache_avg_time / cache_avg_time
    
    print("\n" + "="*60)
    print("SIMULATION COMPLETE")
    print("="*60)
    print(f"Simulation ended after processing {frames} frames.")
    print(f"The simulation runs for the specified number of frames ({frames}),")
    print(f"processing each frame sequentially and collecting statistics.")
    print("="*60)
    print("\nPerformance Metrics:")
    print(f"  Average processing time with cache: {cache_avg_time:.3f}s")
    print(f"  Average processing time without cache: {no_cache_avg_time:.3f}s")
    print(f"  Average time saved per frame: {time_saved:.3f}s")
    print(f"  Speedup factor: {speedup:.2f}x")
    # Calculate final cumulative hit rate (already cumulative, so just get the last value)
    if len(cache_stats['hit_rate']) > 0:
        final_hit_rate = cache_stats['hit_rate'][-1]
        print(f"  Cache hit rate (real tracks, cumulative): {final_hit_rate:.2%}")
    else:
        print(f"  Cache hit rate (real tracks): N/A (no data)")
    print(f"  Total time saved: {np.sum(cache_stats['cache_benefit']):.2f}s")
    print(f"  Max cache size: {max(cache_stats['cache_size'])} entries")
    print("="*60)

if __name__ == "__main__":
    import os
    parser = argparse.ArgumentParser(description="Simulate and analyze cache performance.")
    parser.add_argument("--frames", type=int, default=100, help="Number of frames to simulate.")
    parser.add_argument("--mode", choices=["visualize", "measure"], default="visualize", help="Mode of operation: visualize or measure.")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save the visualization.")

    args = parser.parse_args()
    # Default to saving in semantic_cache folder
    if args.save_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.save_path = os.path.join(script_dir, "cache_performance.png")
    main(frames=args.frames, mode=args.mode, save_path=args.save_path)
