"""
Fast path integration for evaluation
Processes frames using YOLOv8 + ByteTrack
"""
import cv2
import time
import numpy as np
from typing import Dict, List
from ultralytics import YOLO
from .dataset_loader import BoundingBox, MOTDataset


class FastPathProcessor:
    """Fast path processor using YOLOv8 + ByteTrack"""
    
    def __init__(self, model_name: str = 'yolov8n.pt', tracker_config: str = "bytetrack.yaml"):
        """
        Args:
            model_name: YOLO model name (e.g., 'yolov8n.pt')
            tracker_config: Tracker config file name
        """
        self.model = YOLO(model_name)
        self.tracker_config = tracker_config
    
    def process_frame(self, frame_image: np.ndarray) -> tuple:
        """
        Process a single frame
        
        Args:
            frame_image: BGR image array (from cv2.imread)
            
        Returns:
            tuple: (detections_list, processing_time_seconds)
                detections_list: List of BoundingBox objects
        """
        start_time = time.perf_counter()
        
        # Run tracking
        results = self.model.track(frame_image, persist=True, tracker=self.tracker_config, verbose=False)
        
        processing_time = time.perf_counter() - start_time
        
        detections = []
        result = results[0]
        
        if result.boxes.id is not None:
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.cpu().numpy().astype(int)
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            for i in range(len(track_ids)):
                # Convert from xyxy to xywh format
                x1, y1, x2, y2 = boxes_xyxy[i]
                x = float(x1)
                y = float(y1)
                w = float(x2 - x1)
                h = float(y2 - y1)
                
                bbox = BoundingBox(
                    frame=0,  # Will be set by caller
                    track_id=int(track_ids[i]),
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                    confidence=float(confidences[i]),
                    class_id=int(class_ids[i])
                )
                detections.append(bbox)
        
        return detections, processing_time
    
    def process_sequence(self, dataset: MOTDataset) -> Dict:
        """
        Process entire sequence
        
        Returns:
            Dictionary with 'predictions' and 'processing_times'
        """
        all_frames = sorted(dataset.det_data.keys()) if dataset.det_data else []
        if not all_frames:
            # If no detections, try to get frames from GT
            all_frames = sorted(dataset.gt_data.keys()) if dataset.gt_data else []
        
        predictions = {}
        processing_times = []
        total_frames = len(all_frames)
        
        for idx, frame in enumerate(all_frames):
            if (idx + 1) % 100 == 0 or idx == 0:
                print(f"  Processing frame {idx + 1}/{total_frames} (frame {frame})...")
            img_path = dataset.get_frame_image_path(frame)
            if not img_path:
                # Skip if image not found
                processing_times.append(0.0)
                predictions[frame] = []
                continue
            
            # Load image
            frame_image = cv2.imread(img_path)
            if frame_image is None:
                processing_times.append(0.0)
                predictions[frame] = []
                continue
            
            # Process frame
            detections, proc_time = self.process_frame(frame_image)
            
            # Update frame numbers
            for det in detections:
                det.frame = frame
            
            predictions[frame] = detections
            processing_times.append(proc_time)
        
        print(f"  Completed processing {total_frames} frames")
        return {
            'predictions': predictions,
            'processing_times': processing_times
        }

