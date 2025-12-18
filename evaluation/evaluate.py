"""
Main evaluation script
Run evaluation on MOT16 sequences using fast path, slow path, or hybrid approach
"""
import argparse
import os
import json
from typing import Dict, Optional
from .dataset_loader import MOTDataset
from .metrics import evaluate_sequence
from .fast_path_integration import FastPathProcessor
from .slow_path_integration import SlowPathProcessor
from .hybrid_integration import HybridProcessor


def evaluate_processor(dataset: MOTDataset, processor, processor_name: str) -> Dict:
    """Evaluate a processor on a dataset"""
    print(f"\nRunning {processor_name}...")
    output = processor.process_sequence(dataset)
    
    # Evaluate metrics
    gt_data = dataset.gt_data if dataset.gt_data else {}
    frame_rate = dataset.get_frame_rate()
    
    eval_results = evaluate_sequence(
        gt_data,
        output['predictions'],
        output.get('processing_times', []),
        frame_rate
    )
    
    # Add processing time stats
    if 'processing_times' in output:
        processing_times = output['processing_times']
        eval_results['avg_processing_time'] = sum(processing_times) / len(processing_times) if processing_times else 0.0
        eval_results['max_processing_time'] = max(processing_times) if processing_times else 0.0
    
    # Add cache stats if available
    if 'cache_stats' in output:
        eval_results['cache_stats'] = output['cache_stats']
    
    return eval_results


def evaluate_single_sequence(data_root: str, sequence: str, split: str = "train",
                            output_dir: str = "results",
                            methods: Optional[list] = None,
                            test_mode: bool = False,
                            max_frames: Optional[int] = None):
    """
    Evaluate a single sequence
    
    Args:
        data_root: Root directory containing MOT16 data
        sequence: Sequence name (e.g., "MOT16-02")
        split: "train" or "test"
        output_dir: Output directory for results
        methods: List of methods to evaluate: ['fast', 'slow', 'hybrid'] or None for all
        test_mode: If True, use mock models and limit processing
        max_frames: Maximum number of frames to process (None for all)
    """
    if test_mode:
        # Set mock mode for slow path
        os.environ['SLOWPATH_MODEL'] = 'mock'
        print("TEST MODE: Using mock models, limiting frame processing")
        if max_frames is None:
            max_frames = 10  # Default to 10 frames in test mode
    
    print(f"Loading sequence: {sequence}")
    dataset = MOTDataset(data_root, sequence, split)
    
    if not dataset.gt_data:
        print(f"No ground truth available for {sequence}")
        return None
    
    print(f"Sequence info: {dataset.seq_info}")
    total_frames = dataset.get_num_frames()
    print(f"Frames: {total_frames}, FPS: {dataset.get_frame_rate()}")
    
    if max_frames and max_frames < total_frames:
        print(f"Limiting processing to first {max_frames} frames (test mode)")
        # Limit dataset to first N frames
        all_frames = sorted(dataset.gt_data.keys()) if dataset.gt_data else []
        if all_frames:
            limited_frames = all_frames[:max_frames]
            # Create a limited dataset view
            limited_gt = {f: dataset.gt_data[f] for f in limited_frames if f in dataset.gt_data}
            dataset.gt_data = limited_gt
            if dataset.det_data:
                limited_det = {f: dataset.det_data[f] for f in limited_frames if f in dataset.det_data}
                dataset.det_data = limited_det
    
    os.makedirs(output_dir, exist_ok=True)
    
    if methods is None:
        methods = ['fast', 'hybrid']
    
    results = {}
    
    # Fast path only
    if 'fast' in methods:
        fast_processor = FastPathProcessor()
        results['fast_path'] = evaluate_processor(dataset, fast_processor, 'Fast Path')
    
    # Hybrid (fast + slow path with cache)
    if 'hybrid' in methods:
        hybrid_processor = HybridProcessor(use_slow_path=True, wait_for_slow_path=False)
        results['hybrid'] = evaluate_processor(dataset, hybrid_processor, 'Hybrid')
    
    # Hybrid with blocking slow path (for comparison)
    if 'hybrid_blocking' in methods:
        hybrid_blocking = HybridProcessor(use_slow_path=True, wait_for_slow_path=True)
        results['hybrid_blocking'] = evaluate_processor(dataset, hybrid_blocking, 'Hybrid (Blocking)')
    
    # Save results
    results_file = os.path.join(output_dir, f"{sequence}_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("Results Summary:")
    print("="*60)
    for method, metrics in results.items():
        print(f"\n{method}:")
        print(f"  MOTA: {metrics.get('MOTA', 0):.3f}")
        print(f"  IDF1: {metrics.get('IDF1', 0):.3f}")
        print(f"  MOTP: {metrics.get('MOTP', 0):.3f}")
        print(f"  Precision: {metrics.get('precision', 0):.3f}")
        print(f"  Recall: {metrics.get('recall', 0):.3f}")
        print(f"  Fragmentation: {metrics.get('fragmentation', 0)}")
        print(f"  MT: {metrics.get('MT_percentage', 0):.1f}% ({metrics.get('mostly_tracked', 0)} tracks)")
        print(f"  ML: {metrics.get('ML_percentage', 0):.1f}% ({metrics.get('mostly_lost', 0)} tracks)")
        print(f"  Avg Track Completeness: {metrics.get('avg_track_completeness', 0):.3f}")
        print(f"  Deadline Hit Rate: {metrics.get('deadline_hit_rate', 0):.3f}")
        if 'avg_processing_time' in metrics:
            print(f"  Avg Processing Time: {metrics['avg_processing_time']*1000:.2f}ms")
        if 'cache_stats' in metrics:
            cache = metrics['cache_stats']
            print(f"  Cache Hit Rate: {cache.get('hit_rate', 0):.2%}")
    
    return results


def evaluate_all_sequences(data_root: str, split: str = "train",
                           output_dir: str = "results",
                           methods: Optional[list] = None,
                           test_mode: bool = False,
                           max_frames: Optional[int] = None):
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
            results = evaluate_single_sequence(data_root, sequence, split, output_dir, methods, test_mode, max_frames)
            if results:
                all_results[sequence] = results
        except Exception as e:
            print(f"Error evaluating {sequence}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Aggregate results
    if all_results:
        aggregate_file = os.path.join(output_dir, "aggregate_results.json")
        with open(aggregate_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nAggregate results saved to {aggregate_file}")
        
        # Compute averages
        print("\n" + "="*60)
        print("Average Results Across All Sequences:")
        print("="*60)
        method_names = list(all_results[list(all_results.keys())[0]].keys())
        for method in method_names:
            mota_avg = sum(r[method].get('MOTA', 0) for r in all_results.values()) / len(all_results)
            idf1_avg = sum(r[method].get('IDF1', 0) for r in all_results.values()) / len(all_results)
            motp_avg = sum(r[method].get('MOTP', 0) for r in all_results.values()) / len(all_results)
            precision_avg = sum(r[method].get('precision', 0) for r in all_results.values()) / len(all_results)
            recall_avg = sum(r[method].get('recall', 0) for r in all_results.values()) / len(all_results)
            frag_avg = sum(r[method].get('fragmentation', 0) for r in all_results.values()) / len(all_results)
            mt_avg = sum(r[method].get('MT_percentage', 0) for r in all_results.values()) / len(all_results)
            ml_avg = sum(r[method].get('ML_percentage', 0) for r in all_results.values()) / len(all_results)
            completeness_avg = sum(r[method].get('avg_track_completeness', 0) for r in all_results.values()) / len(all_results)
            deadline_avg = sum(r[method].get('deadline_hit_rate', 0) for r in all_results.values()) / len(all_results)
            print(f"\n{method}:")
            print(f"  MOTA: {mota_avg:.3f}")
            print(f"  IDF1: {idf1_avg:.3f}")
            print(f"  MOTP: {motp_avg:.3f}")
            print(f"  Precision: {precision_avg:.3f}")
            print(f"  Recall: {recall_avg:.3f}")
            print(f"  Fragmentation: {frag_avg:.1f}")
            print(f"  MT: {mt_avg:.1f}%")
            print(f"  ML: {ml_avg:.1f}%")
            print(f"  Avg Track Completeness: {completeness_avg:.3f}")
            print(f"  Deadline Hit Rate: {deadline_avg:.3f}")
    
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
    parser.add_argument("--methods", type=str, nargs="+", 
                       choices=['fast', 'hybrid', 'hybrid_blocking'],
                       default=None,
                       help="Methods to evaluate (default: fast, hybrid)")
    parser.add_argument("--test", action="store_true",
                       help="Test mode: use mock models and limit to 10 frames (no downloads)")
    parser.add_argument("--max_frames", type=int, default=None,
                       help="Maximum number of frames to process (useful for testing)")
    
    args = parser.parse_args()
    
    if args.sequence:
        evaluate_single_sequence(args.data_root, args.sequence, args.split, args.output_dir, args.methods, args.test, args.max_frames)
    else:
        evaluate_all_sequences(args.data_root, args.split, args.output_dir, args.methods, args.test, args.max_frames)


if __name__ == "__main__":
    main()
