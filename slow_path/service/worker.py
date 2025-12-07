import asyncio, time, os
from dataclasses import dataclass
from typing import Optional, Dict, Any
from PIL import Image
import io, base64

from .models.loader import load_model
from .schema import CacheRecord
from semantic_cache.semantic_cache import CacheEntry, SemanticCache
from .client import post_cache_record
from .timing import TimerStats


# If the environment variable USE_LOCAL_SEMANTIC_CACHE is set to "1", the worker will
# store results in a local SemanticCache instance instead of posting to
# the external testing cache HTTP endpoint. 

_use_local_cache = os.getenv("USE_LOCAL_SEMANTIC_CACHE", "0") == "1"
_local_cache = None
if _use_local_cache:
    try:
        _local_cache = SemanticCache()
    except Exception:
        _local_cache = None


def get_local_cache():
    """Return the in-process SemanticCache instance if enabled, else None.

    Other modules can call this to access the same cache instance used by
    the worker when `USE_LOCAL_SEMANTIC_CACHE=1`.
    """
    return _local_cache


@dataclass
class Job:
    job_id: str
    frame_id: int
    track_id: int
    bbox: list
    frame_rgb: list
    prompt: str

class Worker:
    def __init__(self):
        self.queue: "asyncio.Queue[Job]" = asyncio.Queue()
        self.results: Dict[str, Dict[str, Any]] = {}
        # Allow selecting the model via env var SLOWPATH_MODEL (mock/blip/blip2/llama)
        model_name = os.getenv("SLOWPATH_MODEL", "llama")
        try:
            self.model = load_model(model_name)
            print(f"[slowpath] loaded model: {model_name}")
        except Exception as e:
            # fallback to a mock model if chosen model fails to load
            print(f"[slowpath] failed to load model '{model_name}': {e}; falling back to 'mock'")
            self.model = load_model("mock")
        self.lat = TimerStats()
        self.infer_lat_ms = []     # last N samples
        self.cache_ok = 0
        self.cache_fail = 0
        self.jobs_enqueued_total = 0

    async def run(self):
        while True:
            job = await self.queue.get()
            try:
                x, y, w, h = job.bbox
                crop = job.frame_rgb[y:y+h, x:x+w]
                img = Image.fromarray(crop)
                buf = io.BytesIO()
                img.save(buf, format="JPEG")
                crop_b64 = base64.b64encode(buf.getvalue()).decode()

                start = time.perf_counter()
                out = self.model.infer(crop_b64, job.prompt)
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

                # If a local semantic cache is configured, put the entry there
                # and treat that as a successful post. Otherwise call the
                # external testing cache HTTP endpoint.
                ok = False
                if _local_cache is not None:
                    try:
                        ce = CacheEntry.from_vlm_output(job.track_id, out, job.bbox, job.frame_id)
                        _local_cache.put(ce)
                        ok = True
                    except Exception:
                        ok = False
                else:
                    if get_local_cache is not None:
                        entry = CacheEntry(
                            track_id=job.track_id,
                            label=out["label"],
                            bbox=job.bbox,
                            confidence=float(out["confidence"]),
                            timestamp=job.frame_id,
                        ).model_dump()
                        ok = get_local_cache.put(entry)
                    else: 
                        ok = post_cache_record(record)
                if ok: self.cache_ok += 1
                else: self.cache_fail += 1

                self.results[job.job_id] = {"status":"done","record":record, "posted": ok}
            except Exception as e:
                self.results[job.job_id] = {"status":"error","error":str(e)}
            finally:
                self.queue.task_done()
