import cv2, numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass
class TriggerConfig:
    every_N: int = 10            # run slow-path every 10 frames
    diff_thresh: float = 8.0     # mean abs-diff threshold (0..255)
    min_gap: int = 2             # at least 5 frames between triggers per track
    cooldown: int = 1            # avoid re-triggering immediately after a diff trigger
    ttl_frames: int = 15         # explicit semantic TTL in frames

class TriggerPolicy:
    def __init__(self, cfg: TriggerConfig = TriggerConfig()):
        self.cfg = cfg
        self.state: Dict[int, Tuple[int, int]] = {} # track_id -> (last_frame, last_trigger_frame)

    def _roi_gray(self, frame_rgb: np.ndarray, bbox):
        x,y,w,h = bbox
        x2, y2 = x+w, y+h
        x,y = max(0,x), max(0,y)
        roi = frame_rgb[y:y2, x:x2]
        if roi.size == 0: return None
        return cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

    def should_enqueue(self, frame_id: int, frame_rgb: np.ndarray, bbox, track_id: int, prev_gray_cache, last_semantic_tick: dict):
        cfg = self.cfg
        last_frame, last_trig = self.state.get(track_id, (-10**9, -10**9))

        # periodic trigger
        periodic = (frame_id % cfg.every_N == 0) and (frame_id - last_trig >= cfg.min_gap)

        # scene-change trigger (with cooldown)
        g = self._roi_gray(frame_rgb, bbox)
        if g is None:
            self.state[track_id] = (frame_id, last_trig)
            return False, prev_gray_cache  # no ROI

        prev_g = prev_gray_cache.get(track_id)
        diff = 0.0
        changed = False
        if prev_g is not None and prev_g.shape == g.shape:
            diff = float(np.mean(np.abs(g.astype(np.int16) - prev_g.astype(np.int16))))
            changed = (diff >= cfg.diff_thresh) and (frame_id - last_trig >= cfg.cooldown)


        # TTL expiry: force refresh if too old
        last = last_semantic_tick.get(track_id, -10**9)
        expired = (frame_id - last) >= self.cfg.ttl_frames

        should = expired or periodic or changed
        self.state[track_id] = (frame_id, frame_id if should else last_trig)
        prev_gray_cache[track_id] = g

        reason = {
            "expired": expired,
            "everyN": periodic,
            "scene_change": changed,
        }
        return should, prev_gray_cache, reason
