"""
Evaluation Metrics for MOT
Implements MOTA, IDF1, and deadline hit-rate
"""
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
from .dataset_loader import BoundingBox, MOTDataset


def iou(bbox1: Tuple[float, float, float, float], 
        bbox2: Tuple[float, float, float, float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes
    Format: (x, y, w, h)
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    # Calculate intersection
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union


def compute_mota(gt_data: Dict[int, List[BoundingBox]], 
                 pred_data: Dict[int, List[BoundingBox]],
                 iou_threshold: float = 0.5) -> Dict:
    """
    Compute MOTA (Multiple Object Tracking Accuracy)
    
    MOTA = 1 - (FN + FP + IDSW) / GT
    
    Returns:
        Dictionary with MOTA score and component metrics
    """
    all_frames = sorted(set(list(gt_data.keys()) + list(pred_data.keys())))
    
    total_gt = 0
    total_fp = 0
    total_fn = 0
    total_idsw = 0
    
    # Track ID mappings between frames
    id_mapping = {}  # pred_id -> gt_id
    
    for frame in all_frames:
        gt_boxes = gt_data.get(frame, [])
        pred_boxes = pred_data.get(frame, [])
        
        total_gt += len(gt_boxes)
        
        if len(gt_boxes) == 0:
            total_fp += len(pred_boxes)
            continue
        
        if len(pred_boxes) == 0:
            total_fn += len(gt_boxes)
            continue
        
        # Compute IoU matrix
        iou_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))
        for i, gt_box in enumerate(gt_boxes):
            for j, pred_box in enumerate(pred_boxes):
                iou_matrix[i, j] = iou(gt_box.bbox, pred_box.bbox)
        
        # Greedy matching
        matched_gt = set()
        matched_pred = set()
        
        # Sort by IoU descending
        matches = []
        for i in range(len(gt_boxes)):
            for j in range(len(pred_boxes)):
                if iou_matrix[i, j] >= iou_threshold:
                    matches.append((iou_matrix[i, j], i, j))
        
        matches.sort(reverse=True)
        
        for iou_val, i, j in matches:
            if i not in matched_gt and j not in matched_pred:
                matched_gt.add(i)
                matched_pred.add(j)
                
                gt_id = gt_boxes[i].track_id
                pred_id = pred_boxes[j].track_id
                
                # Check for ID switch
                if pred_id in id_mapping:
                    if id_mapping[pred_id] != gt_id:
                        total_idsw += 1
                id_mapping[pred_id] = gt_id
        
        # False negatives: unmatched GT
        total_fn += len(gt_boxes) - len(matched_gt)
        
        # False positives: unmatched predictions
        total_fp += len(pred_boxes) - len(matched_pred)
    
    # Compute MOTA
    if total_gt == 0:
        mota = 0.0
    else:
        mota = 1.0 - (total_fn + total_fp + total_idsw) / total_gt
    
    return {
        'MOTA': mota,
        'total_gt': total_gt,
        'FP': total_fp,
        'FN': total_fn,
        'IDSW': total_idsw,
        'precision': (total_gt - total_fn) / (total_gt - total_fn + total_fp) if (total_gt - total_fn + total_fp) > 0 else 0.0,
        'recall': (total_gt - total_fn) / total_gt if total_gt > 0 else 0.0
    }


def compute_idf1(gt_data: Dict[int, List[BoundingBox]], 
                 pred_data: Dict[int, List[BoundingBox]],
                 iou_threshold: float = 0.5) -> Dict:
    """
    Compute IDF1 (ID F1 Score)
    
    IDF1 = 2 * IDTP / (2 * IDTP + IDFP + IDFN)
    
    Returns:
        Dictionary with IDF1 score and component metrics
    """
    all_frames = sorted(set(list(gt_data.keys()) + list(pred_data.keys())))
    
    # Build tracklets
    gt_tracks = defaultdict(list)  # gt_id -> [(frame, bbox), ...]
    pred_tracks = defaultdict(list)  # pred_id -> [(frame, bbox), ...]
    
    for frame in all_frames:
        for gt_box in gt_data.get(frame, []):
            gt_tracks[gt_box.track_id].append((frame, gt_box))
        for pred_box in pred_data.get(frame, []):
            pred_tracks[pred_box.track_id].append((frame, pred_box))
    
    # Match tracks based on overlap
    idtp = 0  # ID True Positives
    idfp = 0  # ID False Positives
    idfn = 0  # ID False Negatives
    
    matched_pred_tracks = set()
    
    for gt_id, gt_tracklet in gt_tracks.items():
        best_match = None
        best_overlap = 0
        
        for pred_id, pred_tracklet in pred_tracks.items():
            if pred_id in matched_pred_tracks:
                continue
            
            # Compute temporal overlap
            gt_frames = set([f for f, _ in gt_tracklet])
            pred_frames = set([f for f, _ in pred_tracklet])
            overlap_frames = gt_frames & pred_frames
            
            if len(overlap_frames) == 0:
                continue
            
            # Compute average IoU over overlapping frames
            total_iou = 0.0
            count = 0
            
            for frame in overlap_frames:
                gt_box = next(bbox for f, bbox in gt_tracklet if f == frame)
                pred_box = next(bbox for f, bbox in pred_tracklet if f == frame)
                iou_val = iou(gt_box.bbox, pred_box.bbox)
                if iou_val >= iou_threshold:
                    total_iou += iou_val
                    count += 1
            
            if count > 0:
                avg_iou = total_iou / count
                overlap_ratio = len(overlap_frames) / max(len(gt_frames), len(pred_frames))
                score = avg_iou * overlap_ratio
                
                if score > best_overlap:
                    best_overlap = score
                    best_match = pred_id
        
        if best_match is not None and best_overlap > 0.5:
            idtp += len(overlap_frames)
            matched_pred_tracks.add(best_match)
        else:
            idfn += len(gt_tracklet)
    
    # Unmatched predictions are false positives
    for pred_id, pred_tracklet in pred_tracks.items():
        if pred_id not in matched_pred_tracks:
            idfp += len(pred_tracklet)
    
    # Compute IDF1
    if (2 * idtp + idfp + idfn) == 0:
        idf1 = 0.0
    else:
        idf1 = 2 * idtp / (2 * idtp + idfp + idfn)
    
    return {
        'IDF1': idf1,
        'IDTP': idtp,
        'IDFP': idfp,
        'IDFN': idfn
    }


def compute_deadline_hit_rate(processing_times: List[float], 
                              frame_rate: float = 30.0) -> Dict:
    """
    Compute deadline hit rate
    
    Args:
        processing_times: List of processing times per frame (in seconds)
        frame_rate: Target frame rate (FPS)
    
    Returns:
        Dictionary with deadline hit rate and statistics
    """
    if not processing_times:
        return {
            'deadline_hit_rate': 0.0,
            'avg_processing_time': 0.0,
            'max_processing_time': 0.0,
            'min_processing_time': 0.0,
            'frames_processed': 0
        }
    
    deadline = 1.0 / frame_rate
    hit_count = sum(1 for t in processing_times if t <= deadline)
    hit_rate = hit_count / len(processing_times)
    
    return {
        'deadline_hit_rate': hit_rate,
        'avg_processing_time': np.mean(processing_times),
        'max_processing_time': np.max(processing_times),
        'min_processing_time': np.min(processing_times),
        'frames_processed': len(processing_times),
        'deadline': deadline
    }


def evaluate_sequence(gt_data: Dict[int, List[BoundingBox]], 
                     pred_data: Dict[int, List[BoundingBox]],
                     processing_times: List[float] = None,
                     frame_rate: float = 30.0) -> Dict:
    """
    Comprehensive evaluation for a sequence
    
    Returns:
        Dictionary with all metrics
    """
    results = {}
    
    # MOTA
    mota_results = compute_mota(gt_data, pred_data)
    results.update(mota_results)
    
    # IDF1
    idf1_results = compute_idf1(gt_data, pred_data)
    results.update(idf1_results)
    
    # Deadline hit rate
    if processing_times:
        deadline_results = compute_deadline_hit_rate(processing_times, frame_rate)
        results.update(deadline_results)
    
    return results

