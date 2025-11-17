"""
Evaluation package for MOT tracking
"""
from .dataset_loader import MOTDataset, BoundingBox
from .metrics import compute_mota, compute_idf1, compute_deadline_hit_rate, evaluate_sequence
from .fast_path_integration import FastPathProcessor
from .slow_path_integration import SlowPathProcessor
from .hybrid_integration import HybridProcessor

__all__ = [
    'MOTDataset',
    'BoundingBox',
    'compute_mota',
    'compute_idf1',
    'compute_deadline_hit_rate',
    'evaluate_sequence',
    'FastPathProcessor',
    'SlowPathProcessor',
    'HybridProcessor',
]

