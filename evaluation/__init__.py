"""
Evaluation package for MOT tracking
"""
from .dataset_loader import MOTDataset, BoundingBox
from .metrics import compute_mota, compute_idf1, compute_deadline_hit_rate, evaluate_sequence
from .visualization import visualize_frame, create_comparison_video, plot_metrics_comparison
from .baselines import NaiveVLMBaseline, FastOnlyBaseline, HybridBaseline, run_baseline_comparison

__all__ = [
    'MOTDataset',
    'BoundingBox',
    'compute_mota',
    'compute_idf1',
    'compute_deadline_hit_rate',
    'evaluate_sequence',
    'visualize_frame',
    'create_comparison_video',
    'plot_metrics_comparison',
    'NaiveVLMBaseline',
    'FastOnlyBaseline',
    'HybridBaseline',
    'run_baseline_comparison'
]

