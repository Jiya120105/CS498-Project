"""
Slow path integration for evaluation
Uses the slow path service client to call VLM
"""
import cv2
import base64
import io
import time
import numpy as np
from typing import Dict, List, Optional
from PIL import Image
import sys
import os

# Add project root to path to import slow_path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from slow_path.service.api import ServiceClient
from .dataset_loader import BoundingBox, MOTDataset


class SlowPathProcessor:
    """Slow path processor using VLM service"""
    
    def __init__(self):
        """Initialize the slow path service client"""
        self.client = ServiceClient()
    
    def process_roi(self, frame_id: int, track_id: int, bbox: List[int], 
                   image_b64: str, prompt_hint: Optional[str] = None) -> Dict:
        """
        Process a single ROI through the slow path
        
        Args:
            frame_id: Frame number
            track_id: Track ID
            bbox: Bounding box as [x, y, w, h]
            image_b64: Base64 encoded image
            prompt_hint: Optional prompt hint
            
        Returns:
            Dictionary with job_id and result (when ready)
        """
        result = self.client.infer(
            frame_id=frame_id,
            track_id=track_id,
            bbox=bbox,
            image_b64=image_b64,
            prompt_hint=prompt_hint
        )
        return result
    
    def trigger_tick(self, frame_id: int, image_b64: str, rois: List[dict],
                    prompt_hint: Optional[str] = None) -> Dict:
        """
        Trigger slow path processing for multiple ROIs
        
        Args:
            frame_id: Frame number
            image_b64: Base64 encoded full frame
            rois: List of dicts with 'track_id' and 'bbox' [x,y,w,h]
            prompt_hint: Optional prompt hint
            
        Returns:
            Dictionary with enqueued track IDs
        """
        result = self.client.trigger_tick(
            frame_id=frame_id,
            image_b64=image_b64,
            rois=rois,
            prompt_hint=prompt_hint
        )
        return result
    
    def get_result(self, job_id: str) -> Dict:
        """Get result for a job ID"""
        return self.client.result(job_id)
    
    def wait_for_result(self, job_id: str, timeout: float = 30.0, 
                       poll_interval: float = 0.1) -> Optional[Dict]:
        """
        Wait for a job to complete
        
        Args:
            job_id: Job ID to wait for
            timeout: Maximum time to wait (seconds)
            poll_interval: Time between polls (seconds)
            
        Returns:
            Result dictionary if completed, None if timeout
        """
        start_time = time.time()
        last_status_print = 0
        while time.time() - start_time < timeout:
            elapsed = time.time() - start_time
            result = self.get_result(job_id)
            status = result.get('status', 'unknown')
            
            # Print status every 2 seconds
            if elapsed - last_status_print >= 2.0:
                print(f"      Job {job_id[:8]}... status: {status} (waited {elapsed:.1f}s)")
                last_status_print = elapsed
            
            if status == 'done':
                return result
            elif status == 'error':
                return result
            time.sleep(poll_interval)
        print(f"      Job {job_id[:8]}... timed out after {timeout}s")
        return None
    
    def close(self):
        """Close the service client"""
        self.client.close()
    
    @staticmethod
    def image_to_b64(image: Image.Image) -> str:
        """Convert PIL Image to base64 string"""
        buf = io.BytesIO()
        image.save(buf, format='JPEG')
        return base64.b64encode(buf.getvalue()).decode()
    
    @staticmethod
    def cv2_to_b64(frame: np.ndarray) -> str:
        """Convert OpenCV frame (BGR) to base64 string"""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        return SlowPathProcessor.image_to_b64(image)

