# CS498-Project

## API Documentation for Slow Path

### POST /infer
Body: { frame_id:int, track_id:int, bbox:[x,y,w,h], image_b64:str }
Resp: { job_id:str }

### GET /result?job_id=...
Resp: { status:"pending"|"done"|"error", record?:CacheRecord }

### POST /trigger/tick
Body: { frame_id:int, image_b64:str, rois:[{track_id:int,bbox:[x,y,w,h]}], prompt_hint?:str }
Resp: { enqueued:[{track_id:int,job_id:str}], count:int }

### POST /cache/put 
Body: CacheRecord
Resp: { ok: true }

### GET /config/get  |  POST /config/set
Tune TriggerConfig at runtime

### GET /metrics
Queue depth, latency percentiles, cache post counts, total enqueued jobs
