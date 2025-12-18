# CS498-Project Slowpath

## Compile & Run (Detailed instructions)

### 1) Environment & dependencies

- Create a virtual environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

- The repository includes a top-level `requirements.txt`. If you need GPU-enabled packages, install them according to your platform and CUDA version.

### 2) Optional: run the cache stub (local HTTP cache)

```bash
# runs a lightweight HTTP endpoint to receive cache PUTs
python slow_path/cache_stub.py
```

Default stub: `http://127.0.0.1:8010`.

### 3) Run the slow-path server (HTTP / HTTPS)

The slow-path service is a FastAPI app. Typical development run uses `uvicorn`:

```bash
# run locally (HTTP)
USE_LOCAL_SEMANTIC_CACHE=1 SLOWPATH_MODEL=llama uvicorn slow_path.service.api:app --host 127.0.0.1 --port 8008 --reload

# with external cache stub
CACHE_BASE_URL=http://127.0.0.1:8010 SLOWPATH_MODEL=blip uvicorn slow_path.service.api:app --host 127.0.0.1 --port 8008
```

### 4) Quick verification

```bash
curl -sS http://127.0.0.1:8008/health
# expected: {"ok": true}
```

## HTTPS API Overview (endpoints summary)

### GET /health
- Simple liveness check. Response: `{ "ok": true }`

### POST /infer
- Purpose: enqueue a single inference job for one ROI.
- Request JSON:

```json
{
  "frame_id": 123,
  "track_id": 5,
  "bbox": [x,y,w,h],
  "image_b64": "<base64-encoded RGB image>",
  "prompt_hint": "optional prompt"
}
```

  - Response JSON: `{ "job_id": "<uuid>" }`


### GET /result?job_id=...
- Purpose: fetch status/result for an enqueued job.
- Response: `{ status:"pending"|"done"|"error", record?:CacheRecord }`

### POST /trigger/tick
- Purpose: non-blocking tick used by tracking pipelines. The server decides which ROIs to enqueue based on trigger policy.
  - Request JSON:

```json
{
  "frame_id": 123,
  "image_b64": "<base64 full-frame>",
  "rois": [{ "track_id": 5, "bbox": [x,y,w,h] }, ...],
  "prompt_hint": "optional"
}
```

  - Response: `{ enqueued:[{track_id:int,job_id:str}], count:int }` 

### GET /config/get  |  POST /config/set
- Purpose: read/update the runtime trigger configuration.
- `POST /config/set` accepts a JSON with fields `every_N`, `diff_thresh`, `min_gap`, `cooldown`, `ttl_frames`.

### GET /metrics
- Purpose: return worker / trigger / cache statistics (latency percentiles, queue depth, counts).

## Programmatic / In-process API

The module exposes a small programmatic surface documented in slow_path/service/api.py:

- `ServiceClient` — a synchronous, in-process client (wraps FastAPI TestClient). Useful for unit tests or embedding the service in-process.

Example:

```python
from slow_path.service.api import ServiceClient
svc = ServiceClient()
res = svc.infer(frame_id=1, track_id=2, bbox=[0,0,32,32], image_b64=my_b64)
job_id = res["job_id"]
print(svc.result(job_id))
svc.close()
```

- `enqueue_infer(...)` — async helper to push a job directly to the worker queue (useful from an asyncio context when the worker loop is running).

## Model configuration

The slow-path worker has following available models are:

- `mock` — a mock VLM that returns fixed labels instantly (latency ~40ms). Useful for testing the pipeline.
- `blip` — BLIP image captioning model (Salesforce/blip-image-captioning-base). Lightweight, runs on CPU or GPU. Returns a label extracted from the caption.
- `llama` — Meta's Llama 3.2 Vision (meta-llama/Llama-3.2-11B-Vision-Instruct). Requires significant GPU

### Model selection

The model is loaded once when the worker starts with environment variable SLOWPATH_MODEL.If no variable provided, default `llama`, while exceptions fall back to `mock`.

### Model input and output

**Input:**
- A PIL Image (RGB, any size).
- 
- A prompt string (optional, can be empty or provide task-specific instructions).

**Output:**
- A JSON dict with keys:
  - `label` — semantic category (string), e.g., "person", "car", "background".
  - `confidence` — float in [0, 1], model's confidence in the label.
  - `metadata` — optional dict with auxiliary info (raw model output, description, etc.).

Example model output:

```json
{
  "label": "person",
  "confidence": 0.92,
  "metadata": {
    "raw": "A person in blue clothing",
    "labels": ["person"]
  }
}
```

## Running the server and tests (quick start)

### Start a local cache stub (optional)

```bash
# optional: run the included cache stub to capture cache POSTs
python slow_path/cache_stub.py
```

By default the stub listens on `http://127.0.0.1:8010` (see the stub file top comments).

### Start the slow-path server

Two common modes:

- In-process SemanticCache (good for single-process dev/testing)
- External cache HTTP endpoint (useful for multi-process or distributed setups)

Examples:

```bash
# In-process cache (single-process testing)
USE_LOCAL_SEMANTIC_CACHE=1 SLOWPATH_MODEL=llama uvicorn slow_path.service.api:app --host 127.0.0.1 --port 8008 --reload

# Use external cache stub
CACHE_BASE_URL=http://127.0.0.1:8010 SLOWPATH_MODEL=blip uvicorn slow_path.service.api:app --host 127.0.0.1 --port 8008
```

### Run the load/test generator (`run_test.py`)

This is a simple synthetic generator useful for quick load tests:

```bash
python slow_path/tests/run_test.py --base http://127.0.0.1:8008 --fps 15 --secs 30 --boxes 2
```

Helpful flags:
- `--base` — slow-path base URL (default `http://127.0.0.1:8008`)
- `--fps` — frames per second to simulate
- `--secs` — duration in seconds
- `--boxes` — ROIs per frame
- `--force` — force enqueue (simulate bypassing the trigger policy)

### Run the image sequence driver on your dataset

This script streams a directory of images to the slow-path service as frames and ROIs.

```bash
python image_sequence_driver.py \
	--root /absolute/path/to/images \
	--pattern "*.jpg" \
	--base http://127.0.0.1:8008 \
	--fps 15 \
	--resize 640x360 \
	--max-rois 3 \
	--fallback-grid
```

Common driver flags: `--root`, `--pattern`, `--base`, `--fps`, `--resize`, `--max-rois`, `--fallback-grid`.

### Collecting metrics to CSV and plotting
Collect per-second metrics to a CSV using `collect_metrics.py` - so run while running :

```bash
python slow_path/collect_metrics.py --base http://127.0.0.1:8008 --secs 60 --out slow_path/metrics.csv
```
### Collect metrics and plot results

- Plot saved figures (uses `plot_metrics.py`):

```bash
python slow_path/collect_metrics.py --base http://127.0.0.1:8008 --secs 60 --out slow_path/metrics.csv
python slow_path/plot_metrics.py --csv slow_path/metrics.csv --out-dir slow_path/plots
```