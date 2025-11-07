"""
Visualization tools for debugging and analysis
"""
import os
import cv2
import numpy as np
from typing import Dict, List, Optional
from .dataset_loader import BoundingBox, MOTDataset


def draw_bbox(img: np.ndarray, bbox: BoundingBox, color: tuple, 
              label: str = "", thickness: int = 2) -> np.ndarray:
    """
    Draw bounding box on image
    
    Args:
        img: Image array (H, W, 3)
        bbox: BoundingBox object
        color: BGR color tuple
        label: Optional label text
        thickness: Line thickness
    """
    x, y, w, h = bbox.bbox
    x, y, w, h = int(x), int(y), int(w), int(h)
    
    # Draw rectangle
    cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
    
    # Draw label
    if label:
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y = max(y, label_size[1] + 10)
        cv2.rectangle(img, (x, label_y - label_size[1] - 10), 
                     (x + label_size[0], label_y), color, -1)
        cv2.putText(img, label, (x, label_y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return img


def visualize_frame(dataset: MOTDataset, frame: int, 
                   pred_boxes: Optional[List[BoundingBox]] = None,
                   show_gt: bool = True,
                   output_path: Optional[str] = None) -> np.ndarray:
    """
    Visualize a single frame with GT and/or predictions
    
    Args:
        dataset: MOTDataset instance
        frame: Frame number
        pred_boxes: Optional list of predicted bounding boxes
        show_gt: Whether to show ground truth
        output_path: Optional path to save image
    """
    img_path = dataset.get_frame_image_path(frame)
    if not img_path or not os.path.exists(img_path):
        raise ValueError(f"Image not found for frame {frame}")
    
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not load image: {img_path}")
    
    # Draw ground truth in green
    if show_gt:
        gt_boxes = dataset.get_gt_for_frame(frame)
        for gt_box in gt_boxes:
            img = draw_bbox(img, gt_box, (0, 255, 0), 
                          label=f"GT-{gt_box.track_id}", thickness=2)
    
    # Draw predictions in red
    if pred_boxes:
        for pred_box in pred_boxes:
            img = draw_bbox(img, pred_box, (0, 0, 255), 
                          label=f"P-{pred_box.track_id}", thickness=2)
    
    if output_path:
        cv2.imwrite(output_path, img)
    
    return img


def create_comparison_video(dataset: MOTDataset,
                           gt_data: Dict[int, List[BoundingBox]],
                           pred_data: Dict[int, List[BoundingBox]],
                           output_path: str,
                           fps: float = 10.0,
                           max_frames: Optional[int] = None):
    """
    Create a comparison video showing GT vs predictions
    
    Args:
        dataset: MOTDataset instance
        gt_data: Ground truth data dictionary
        pred_data: Prediction data dictionary
        output_path: Output video path
        fps: Output video FPS
        max_frames: Maximum number of frames to process
    """
    all_frames = sorted(set(list(gt_data.keys()) + list(pred_data.keys())))
    if max_frames:
        all_frames = all_frames[:max_frames]
    
    if not all_frames:
        print("No frames to process")
        return
    
    # Get first frame to determine video size
    first_img_path = dataset.get_frame_image_path(all_frames[0])
    if not first_img_path:
        print(f"Could not find image for frame {all_frames[0]}")
        return
    
    sample_img = cv2.imread(first_img_path)
    if sample_img is None:
        print(f"Could not load sample image")
        return
    
    h, w = sample_img.shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    for frame in all_frames:
        try:
            img = visualize_frame(dataset, frame, 
                                 pred_boxes=pred_data.get(frame, []),
                                 show_gt=True)
            out.write(img)
        except Exception as e:
            print(f"Error processing frame {frame}: {e}")
            continue
    
    out.release()
    print(f"Video saved to {output_path}")


def plot_metrics_comparison(results_dict: Dict[str, Dict], 
                           output_path: Optional[str] = None):
    """
    Create bar plots comparing metrics across different methods
    
    Args:
        results_dict: Dictionary mapping method names to metric results
        output_path: Optional path to save plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return
    
    methods = list(results_dict.keys())
    
    # Extract metrics
    mota_scores = [results_dict[m].get('MOTA', 0) for m in methods]
    idf1_scores = [results_dict[m].get('IDF1', 0) for m in methods]
    deadline_rates = [results_dict[m].get('deadline_hit_rate', 0) for m in methods]
    
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # MOTA
    axes[0].bar(methods, mota_scores, color='blue', alpha=0.7)
    axes[0].set_ylabel('MOTA')
    axes[0].set_title('MOTA Comparison')
    axes[0].set_ylim([0, 1])
    axes[0].tick_params(axis='x', rotation=45)
    
    # IDF1
    axes[1].bar(methods, idf1_scores, color='green', alpha=0.7)
    axes[1].set_ylabel('IDF1')
    axes[1].set_title('IDF1 Comparison')
    axes[1].set_ylim([0, 1])
    axes[1].tick_params(axis='x', rotation=45)
    
    # Deadline Hit Rate
    axes[2].bar(methods, deadline_rates, color='red', alpha=0.7)
    axes[2].set_ylabel('Deadline Hit Rate')
    axes[2].set_title('Deadline Hit Rate Comparison')
    axes[2].set_ylim([0, 1])
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()

