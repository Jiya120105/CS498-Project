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
    def __init__(self, max_queue_size: int = 50):
        # Bounded queue to provide deterministic backpressure
        qmax = int(os.getenv("SLOWPATH_QUEUE_MAX", str(max_queue_size)))
        self.queue: "asyncio.Queue[Job]" = asyncio.Queue(maxsize=qmax)
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
        self.jobs_dropped_total = 0
        # De-duplication and recency tracking
        self.pending_tracks: set[int] = set()     # tracks currently enqueued/processing
        self.last_semantic_tick: Dict[int, int] = {}  # track_id -> frame_id when result was produced
        self.duplicates_skipped = 0
        self.resolved_skipped = 0

    async def try_enqueue(self, job: Job) -> tuple[bool, str]:
        """Try to enqueue a job with duplicate suppression and bounded backpressure.

        Returns (ok, reason). reason in {"enqueued", "duplicate_inflight", "dropped_oldest", "queue_full"}.
        """
        tid = job.track_id
        # Hard sticky policy: once resolved, never run again for this track
        if tid in self.last_semantic_tick:
            self.resolved_skipped += 1
            return False, "already_resolved"
        # If a job for this track is already in-flight or queued, skip enqueuing a duplicate
        if tid in self.pending_tracks:
            self.duplicates_skipped += 1
            return False, "duplicate_inflight"

        # Mark as pending before attempting to put to avoid races
        self.pending_tracks.add(tid)
        try:
            try:
                self.queue.put_nowait(job)
                self.jobs_enqueued_total += 1
                return True, "enqueued"
            except asyncio.QueueFull:
                # Queue is full. Prefer freshness: drop the oldest job in queue (if any)
                try:
                    dropped_job = self.queue.get_nowait()
                    self.results[dropped_job.job_id] = {
                        "status": "error",
                        "error": "dropped_due_to_backpressure"
                    }
                    self.jobs_dropped_total += 1
                    # Put the new (fresh) job
                    self.queue.put_nowait(job)
                    self.jobs_enqueued_total += 1
                    return True, "dropped_oldest"
                except asyncio.QueueEmpty:
                    # Rare race: queue became empty — try again
                    self.queue.put_nowait(job)
                    self.jobs_enqueued_total += 1
                    return True, "enqueued"
                except asyncio.QueueFull:
                    # Could not recover; drop this new job
                    self.results[job.job_id] = {"status": "error", "error": "queue_full"}
                    return False, "queue_full"
        except Exception:
            # If anything fails, make sure to remove pending flag
            self.pending_tracks.discard(tid)
            raise

    def submit_job(self, job: Job) -> None:
        """Compatibility wrapper for existing callers.

        Schedules an asynchronous enqueue attempt and returns immediately.
        """
        async def _do():
            try:
                await self.try_enqueue(job)
            except Exception:
                # swallow errors in fire-and-forget path
                pass
        try:
            asyncio.create_task(_do())
        except RuntimeError:
            # In case there's no running loop (shouldn't happen in our runner),
            # fall back to a synchronous attempt.
            loop = asyncio.get_event_loop()
            loop.create_task(_do())

    async def run(self):
        print("[Worker] Loop started.")
        try:
            while True:
                # Pull one job and process sequentially (no batching)
                jb = await self.queue.get()
                try:
                    x, y, w, h = jb.bbox
                    crop = jb.frame_rgb[y:y+h, x:x+w]
                    img = Image.fromarray(crop)

                    start = time.perf_counter()
                    out = await asyncio.to_thread(self.model.infer, img, jb.prompt)
                    infer_ms = (time.perf_counter()-start)*1000
                    self.lat.observe_ms(infer_ms)
                    self.infer_lat_ms.append(infer_ms)
                    if len(self.infer_lat_ms) > 200:
                        self.infer_lat_ms.pop(0)

                    record = CacheRecord(
                        track_id=jb.track_id,
                        label=out.get("label",""),
                        bbox=jb.bbox,
                        confidence=float(out.get("confidence", 0.0)),
                        timestamp=jb.frame_id,
                        ttl=5,
                        metadata=out.get("metadata", {})
                    ).model_dump()

                    ok = False
                    if _local_cache is not None:
                        try:
                            ce = CacheEntry.from_vlm_output(jb.track_id, out, jb.bbox, jb.frame_id)
                            _local_cache.put(ce)
                            ok = True
                        except Exception:
                            ok = False
                    if ok: self.cache_ok += 1
                    else: self.cache_fail += 1
                    self.results[jb.job_id] = {"status":"done","record":record, "posted": ok}
                    self.last_semantic_tick[jb.track_id] = jb.frame_id
                except Exception as e:
                    print(f"[Worker] Error processing job {jb.job_id}: {e}")
                    import traceback
                    traceback.print_exc()
                    self.results[jb.job_id] = {"status":"error","error":str(e)}
                finally:
                    self.pending_tracks.discard(jb.track_id)
                    self.queue.task_done()
        except Exception as e:
            print(f"[Worker] CRITICAL FAILURE: {e}")
            import traceback
            traceback.print_exc()
