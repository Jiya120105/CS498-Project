import cv2, numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass
class TriggerConfig:
    every_N: int = 10            # run slow-path every 15 frames
    diff_thresh: float = 8.0     # mean abs-diff threshold (0..255)
    min_gap: int = 2             # at least 5 frames between triggers per track
    cooldown: int = 1            # avoid re-triggering immediately after a diff trigger
    ttl_frames: int = 14         # explicit semantic TTL in frames
    strategies: list = None      # ["periodic", "new", "scene_change"]

class TriggerPolicy:
    def __init__(self, cfg: TriggerConfig = TriggerConfig()):
        self.cfg = cfg
        if self.cfg.strategies is None:
            self.cfg.strategies = ["periodic", "new", "scene_change"]
        # per-track book-keeping: track_id -> (last_frame, last_trigger_frame)
        self.state: Dict[int, Tuple[int, int]] = {}

    def _roi_gray(self, frame_rgb: np.ndarray, bbox):
        x,y,w,h = bbox
        x2, y2 = x+w, y+h
        x,y = max(0,x), max(0,y)
        roi = frame_rgb[y:y2, x:x2]
        if roi.size == 0: return None
        return cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

    def should_enqueue(self, frame_id: int, frame_rgb: np.ndarray, bbox, track_id: int, prev_gray_cache, last_semantic_tick: dict, is_resolved: bool = False):
        cfg = self.cfg
        last_frame, last_trig = self.state.get(track_id, (-10**9, -10**9))
        
        # Strategies check
        use_periodic = "periodic" in cfg.strategies
        use_change = "scene_change" in cfg.strategies
        # "new" is implicit: if track_id not in state, it's new.

        # 1. New Object Check (Implicit)
        is_new = track_id not in self.state
        
        # 2. Periodic Refresh
        # If resolved (sticky), we SKIP periodic refresh.
        periodic = False
        if use_periodic and not is_resolved:
            periodic = (frame_id % cfg.every_N == 0) and (frame_id - last_trig >= cfg.min_gap)

        # 3. Scene-Change Detection
        g = self._roi_gray(frame_rgb, bbox)
        if g is None:
            self.state[track_id] = (frame_id, last_trig)
            return False, prev_gray_cache, {}  # no ROI

        prev_g = prev_gray_cache.get(track_id)
        diff = 0.0
        changed = False
        if use_change and prev_g is not None and prev_g.shape == g.shape:
            diff = float(np.mean(np.abs(g.astype(np.int16) - prev_g.astype(np.int16))))
            # Even if resolved, a MAJOR change might warrant a re-check, 
            # but for "sticky" logic, we usually assume the ID persists. 
            # We'll allow change to trigger updates if it's really big, or respect the sticky rule.
            # User requirement: "once a vlm said no or yes... shouldn't run vlm on it again"
            # implying we assume tracking holds ID correctly.
            # However, if 'scene_change' is explicitly requested, we honor it.
            if not is_resolved:
                 changed = (diff >= cfg.diff_thresh) and (frame_id - last_trig >= cfg.cooldown)

        # 4. TTL Expiry
        # If resolved, we ignore TTL (Sticky).
        expired = False
        if not is_resolved:
            last = last_semantic_tick.get(track_id, -10**9)
            expired = (frame_id - last) >= self.cfg.ttl_frames

        should = (is_new) or (expired) or (periodic) or (changed)
        
        # If strictly "new" strategy only, disable others
        if "new" in cfg.strategies and len(cfg.strategies) == 1:
             should = is_new

        self.state[track_id] = (frame_id, frame_id if should else last_trig)
        prev_gray_cache[track_id] = g

        reason = {
            "new": is_new,
            "expired": expired,
            "everyN": periodic,
            "scene_change": changed,
        }
        return should, prev_gray_cache, reason
