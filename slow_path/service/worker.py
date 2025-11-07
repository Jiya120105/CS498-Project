import asyncio, time
from dataclasses import dataclass
from typing import Optional, Dict, Any
from PIL import Image
import io, base64

from .models.loader import load_model
from .schema import CacheRecord
from .client import post_cache_record
from .timing import TimerStats


@dataclass
class Job:
    job_id: str
    frame_id: int
    track_id: int
    bbox: list
    image_b64: str
    prompt: str

class Worker:
    def __init__(self):
        self.queue: "asyncio.Queue[Job]" = asyncio.Queue()
        self.results: Dict[str, Dict[str, Any]] = {}
        self.model = load_model("blip")
        self.lat = TimerStats()
        self.infer_lat_ms = []     # last N samples
        self.cache_ok = 0
        self.cache_fail = 0
        self.jobs_enqueued_total = 0

    async def run(self):
        while True:
            job = await self.queue.get()
            try:
                img = Image.open(io.BytesIO(base64.b64decode(job.image_b64))).convert("RGB")
                start = time.perf_counter()
                out = self.model.infer(img, job.prompt)
                infer_ms = (time.perf_counter()-start)*1000
                self.lat.observe_ms(infer_ms)
                if len(self.lat.samples)%10==0:
                    print("[slowpath] infer_ms stats:", self.lat.percentiles())

                self.infer_lat_ms.append(infer_ms)
                if len(self.infer_lat_ms) > 200:  # keep a short window
                    self.infer_lat_ms.pop(0)

                record = CacheRecord(
                    track_id=job.track_id,
                    label=out["label"],
                    bbox=job.bbox,
                    confidence=float(out["confidence"]),
                    timestamp=job.frame_id,
                    ttl=5,
                    metadata=out.get("metadata", {})
                ).model_dump()
                ok = post_cache_record(record)
                if ok: self.cache_ok += 1
                else: self.cache_fail += 1

                self.results[job.job_id] = {"status":"done","record":record, "posted": ok}
            except Exception as e:
                self.results[job.job_id] = {"status":"error","error":str(e)}
            finally:
                self.queue.task_done()
