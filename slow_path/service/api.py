import asyncio, uuid, base64, io
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel, Field
from PIL import Image
import numpy as np

from .worker import Worker, Job
from .triggers.policy import TriggerPolicy, TriggerConfig

app = FastAPI(title="SlowPath VLM")
worker = Worker()
_loop_started = False

TRIGGER_SEEN_ROIS = 0
TRIGGER_ENQUEUED_ROIS = 0

class InferReq(BaseModel):
    frame_id: int
    track_id: int
    bbox: list[int] = Field(..., min_items=4, max_items=4)  # [x,y,w,h]
    image_b64: str  # base64-encoded RGB image (ROI or full frame)
    prompt_hint: Optional[str] = None


class TriggerConfigDTO(BaseModel):
    every_N: int
    diff_thresh: float
    min_gap: int
    cooldown: int

@app.get("/config/get")
def get_config():
    cfg = trigger.cfg
    return {"trigger": {
        "every_N": cfg.every_N,
        "diff_thresh": cfg.diff_thresh,
        "min_gap": cfg.min_gap,
        "cooldown": cfg.cooldown
    }}

@app.post("/config/set")
def set_config(new: TriggerConfigDTO):
    trigger.cfg.every_N    = new.every_N
    trigger.cfg.diff_thresh = new.diff_thresh
    trigger.cfg.min_gap     = new.min_gap
    trigger.cfg.cooldown    = new.cooldown
    return {"ok": True, "trigger": new.model_dump()}

def _percentiles(vals):
    if not vals: return {}
    a = sorted(vals); n=len(a)
    def p(q): 
        idx = int(q*(n-1))
        return float(a[idx])
    return {"count": n, "p50": p(0.5), "p95": p(0.95), "max": a[-1]}

@app.get("/metrics")
def metrics():
    return {
        "worker": {"queue_depth": worker.queue.qsize(),
        "results_size": len(worker.results),
        "infer_latency_ms": _percentiles(worker.infer_lat_ms),
        "cache_posts": {"ok": worker.cache_ok, "fail": worker.cache_fail},
        "jobs_enqueued_total": worker.jobs_enqueued_total},
        "trigger": {
            "seen": TRIGGER_SEEN_ROIS,
            "enqueued": TRIGGER_ENQUEUED_ROIS,
            "enqueue_rate": (
                TRIGGER_ENQUEUED_ROIS / TRIGGER_SEEN_ROIS if TRIGGER_SEEN_ROIS else 0.0
            )
            }
    }

@app.on_event("startup")
async def startup():
    global _loop_started
    if not _loop_started:
        asyncio.create_task(worker.run())
        _loop_started = True

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/infer")
async def infer(req: InferReq):
    job_id = str(uuid.uuid4())
    prompt = (req.prompt_hint or
              'Return JSON: {"label": "<category>", "confidence": <0..1>, "metadata": {...}}')
    await worker.queue.put(Job(job_id, req.frame_id, req.track_id, req.bbox, req.image_b64, prompt))
    return {"job_id": job_id}

@app.get("/result")
def result(job_id: str):
    return worker.results.get(job_id, {"status":"pending"})

trigger = TriggerPolicy(TriggerConfig())
_prev_gray_cache = {}  # track_id -> gray ROI

class TickReq(BaseModel):
    frame_id: int
    image_b64: str                     # full frame RGB
    rois: list[dict] = Field(default_factory=list)  # [{track_id:int, bbox:[x,y,w,h]}]
    prompt_hint: str | None = None

@app.post("/trigger/tick")
async def trigger_tick(req: TickReq):

    # decode full frame once
    img = Image.open(io.BytesIO(base64.b64decode(req.image_b64))).convert("RGB")
    frame_rgb = np.array(img)  # HWC, RGB

    enqueued = []
    for roi in req.rois:
        TRIGGER_SEEN_ROIS += 1
        tid = int(roi["track_id"])
        bbox = [int(v) for v in roi["bbox"]]
        should, _ = trigger.should_enqueue(req.frame_id, frame_rgb, bbox, tid, _prev_gray_cache)
        if should:
            TRIGGER_ENQUEUED_ROIS += 1
            # crop to send a lighter payload to /infer
            x,y,w,h = bbox
            crop = img.crop((x,y,x+w,y+h))
            buf = io.BytesIO(); crop.save(buf, format="JPEG")
            roi_b64 = base64.b64encode(buf.getvalue()).decode()
            jid = str(uuid.uuid4())

            await worker.queue.put(
                Job(job_id=jid,
                    frame_id=req.frame_id,
                    track_id=tid,
                    bbox=bbox,
                    image_b64=roi_b64,
                    prompt=req.prompt_hint or "")
            )
            
            enqueued.append({"track_id": tid, "job_id": jid})
            worker.jobs_enqueued_total += 1

    return {"enqueued_track_ids": enqueued, "count": len(enqueued)}
