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