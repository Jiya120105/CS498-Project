"""
MOT16/17 Dataset Loader
Loads ground truth and detection data from MOT format
"""
import os
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class BoundingBox:
    """Bounding box representation"""
    frame: int
    track_id: int
    x: float
    y: float
    w: float
    h: float
    confidence: float = 1.0
    class_id: int = 1  # 1 = pedestrian for MOT
    
    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        """Returns (x, y, w, h)"""
        return (self.x, self.y, self.w, self.h)
    
    @property
    def area(self) -> float:
        """Bounding box area"""
        return self.w * self.h


class MOTDataset:
    """MOT16/17 dataset loader"""
    
    def __init__(self, data_root: str, sequence: str, split: str = "train"):
        """
        Args:
            data_root: Root directory containing MOT16 or MOT17 folders
            sequence: Sequence name (e.g., "MOT16-02")
            split: "train" or "test"
        """
        self.data_root = data_root
        self.sequence = sequence
        self.split = split
        
        # Build paths
        self.seq_dir = os.path.join(data_root, f"MOT16/{split}/{sequence}")
        self.img_dir = os.path.join(self.seq_dir, "img1")
        self.gt_path = os.path.join(self.seq_dir, "gt", "gt.txt")
        self.det_path = os.path.join(self.seq_dir, "det", "det.txt")
        self.seqinfo_path = os.path.join(self.seq_dir, "seqinfo.ini")
        
        # Load sequence info
        self.seq_info = self._load_seqinfo()
        
        # Load data
        self.gt_data = self._load_gt() if os.path.exists(self.gt_path) else None
        self.det_data = self._load_det() if os.path.exists(self.det_path) else None
    
    def _load_seqinfo(self) -> Dict:
        """Load sequence information from seqinfo.ini"""
        info = {}
        if os.path.exists(self.seqinfo_path):
            with open(self.seqinfo_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        # Try to convert numeric values
                        try:
                            if '.' in value:
                                info[key] = float(value)
                            else:
                                info[key] = int(value)
                        except ValueError:
                            info[key] = value
        return info
    
    def _load_gt(self) -> Dict[int, List[BoundingBox]]:
        """
        Load ground truth data
        Format: frame, id, x, y, w, h, confidence, class, visibility
        """
        gt_dict = {}
        if not os.path.exists(self.gt_path):
            return gt_dict
        
        with open(self.gt_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 6:
                    continue
                
                frame = int(float(parts[0]))
                track_id = int(float(parts[1]))
                x = float(parts[2])
                y = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])
                confidence = float(parts[6]) if len(parts) > 6 else 1.0
                class_id = int(float(parts[7])) if len(parts) > 7 else 1
                visibility = float(parts[8]) if len(parts) > 8 else 1.0
                
                # Only include visible pedestrians (class=1, visibility>0.25)
                if class_id == 1 and visibility > 0.25:
                    bbox = BoundingBox(frame, track_id, x, y, w, h, confidence, class_id)
                    if frame not in gt_dict:
                        gt_dict[frame] = []
                    gt_dict[frame].append(bbox)
        
        return gt_dict
    
    def _load_det(self) -> Dict[int, List[BoundingBox]]:
        """
        Load detection data
        Format: frame, id, x, y, w, h, confidence, ...
        """
        det_dict = {}
        if not os.path.exists(self.det_path):
            return det_dict
        
        with open(self.det_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 6:
                    continue
                
                frame = int(float(parts[0]))
                track_id = int(float(parts[1]))  # Usually -1 for detections
                x = float(parts[2])
                y = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])
                confidence = float(parts[6]) if len(parts) > 6 else 1.0
                
                bbox = BoundingBox(frame, track_id, x, y, w, h, confidence)
                if frame not in det_dict:
                    det_dict[frame] = []
                det_dict[frame].append(bbox)
        
        return det_dict
    
    def get_frame_image_path(self, frame: int) -> Optional[str]:
        """Get path to image for a given frame"""
        frame_num = str(frame).zfill(6)
        ext = self.seq_info.get('imExt', '.jpg')
        img_path = os.path.join(self.img_dir, f"{frame_num}{ext}")
        return img_path if os.path.exists(img_path) else None
    
    def get_gt_for_frame(self, frame: int) -> List[BoundingBox]:
        """Get ground truth boxes for a frame"""
        return self.gt_data.get(frame, [])
    
    def get_det_for_frame(self, frame: int) -> List[BoundingBox]:
        """Get detection boxes for a frame"""
        return self.det_data.get(frame, [])
    
    def get_num_frames(self) -> int:
        """Get total number of frames"""
        return self.seq_info.get('seqLength', 0)
    
    def get_frame_rate(self) -> float:
        """Get frame rate"""
        return self.seq_info.get('frameRate', 30.0)
    
    def get_all_sequences(self, split: str = None) -> List[str]:
        """Get list of all sequences in the dataset"""
        split = split or self.split
        mot_dir = os.path.join(self.data_root, f"MOT16/{split}")
        if os.path.exists(mot_dir):
            return [d for d in os.listdir(mot_dir) 
                   if os.path.isdir(os.path.join(mot_dir, d))]
        return []

