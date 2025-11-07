"""
Main evaluation script
Run evaluation on MOT16 sequences
"""
import argparse
import os
import json
from typing import Dict
from .dataset_loader import MOTDataset
from .metrics import evaluate_sequence
from .baselines import run_baseline_comparison
from .visualization import plot_metrics_comparison, create_comparison_video


def evaluate_single_sequence(data_root: str, sequence: str, split: str = "train",
                            output_dir: str = "results"):
    """Evaluate a single sequence"""
    print(f"Loading sequence: {sequence}")
    dataset = MOTDataset(data_root, sequence, split)
    
    if not dataset.gt_data:
        print(f"No ground truth available for {sequence}")
        return None
    
    if not dataset.det_data:
        print(f"No detections available for {sequence}")
        return None
    
    print(f"Sequence info: {dataset.seq_info}")
    print(f"Frames: {dataset.get_num_frames()}, FPS: {dataset.get_frame_rate()}")
    
    # Run baseline comparison
    os.makedirs(output_dir, exist_ok=True)
    results = run_baseline_comparison(dataset, output_dir)
    
    # Save results
    results_file = os.path.join(output_dir, f"{sequence}_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_file}")
    
    # Create plots
    plot_file = os.path.join(output_dir, f"{sequence}_comparison.png")
    plot_metrics_comparison(results, plot_file)
    
    return results


def evaluate_all_sequences(data_root: str, split: str = "train",
                           output_dir: str = "results"):
    """Evaluate all sequences in the dataset"""
    # Get all sequences
    temp_dataset = MOTDataset(data_root, "MOT16-02", split)
    sequences = temp_dataset.get_all_sequences(split)
    
    print(f"Found {len(sequences)} sequences")
    
    all_results = {}
    
    for sequence in sequences:
        print(f"\n{'='*60}")
        print(f"Evaluating {sequence}")
        print(f"{'='*60}")
        
        try:
            results = evaluate_single_sequence(data_root, sequence, split, output_dir)
            if results:
                all_results[sequence] = results
        except Exception as e:
            print(f"Error evaluating {sequence}: {e}")
            continue
    
    # Aggregate results
    if all_results:
        aggregate_file = os.path.join(output_dir, "aggregate_results.json")
        with open(aggregate_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nAggregate results saved to {aggregate_file}")
        
        # Compute averages
        print("\nAverage Results:")
        for method in list(all_results[list(all_results.keys())[0]].keys()):
            mota_avg = sum(r[method].get('MOTA', 0) for r in all_results.values()) / len(all_results)
            idf1_avg = sum(r[method].get('IDF1', 0) for r in all_results.values()) / len(all_results)
            deadline_avg = sum(r[method].get('deadline_hit_rate', 0) for r in all_results.values()) / len(all_results)
            print(f"{method}: MOTA={mota_avg:.3f}, IDF1={idf1_avg:.3f}, Deadline={deadline_avg:.3f}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate MOT tracking system")
    parser.add_argument("--data_root", type=str, default="data",
                       help="Root directory containing MOT16/17 data")
    parser.add_argument("--sequence", type=str, default=None,
                       help="Specific sequence to evaluate (e.g., MOT16-02). If not specified, evaluates all")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"],
                       help="Dataset split to use")
    parser.add_argument("--output_dir", type=str, default="results",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    if args.sequence:
        evaluate_single_sequence(args.data_root, args.sequence, args.split, args.output_dir)
    else:
        evaluate_all_sequences(args.data_root, args.split, args.output_dir)


if __name__ == "__main__":
    main()

