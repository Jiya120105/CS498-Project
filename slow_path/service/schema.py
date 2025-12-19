from pydantic import BaseModel, Field
from typing import Dict, List

class CacheRecord(BaseModel):
    track_id: int
    label: str
    bbox: List[int] = Field(..., min_items=4, max_items=4)  # [x,y,w,h]
    confidence: float
    timestamp: int  # frame index
    ttl: int        # in frames
    metadata: Dict[str, object] = {}
