# tests/test_integration_local_cache.py
import base64
import io
import time
from PIL import Image
from slow_path.service import ServiceClient, get_local_cache
from semantic_cache import CacheEntry

def make_test_b64_jpeg(color=(128, 64, 32), size=(32, 32)):
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()

def main():
    print("[cache_test] Starting ServiceClient...")
    svc = ServiceClient()
    try:
        image_b64 = make_test_b64_jpeg()
        track_id = 123
        print("[cache_test] Enqueuing job...")
        resp = svc.infer(frame_id=1, track_id=track_id, bbox=[0,0,10,10], image_b64=image_b64, prompt_hint="test")
        job_id = resp.get("job_id")
        print(f"[cache_test] Job ID: {job_id}")
        assert job_id

        # wait for the background worker to process
        time.sleep(0.5)

        # check job result (optional)
        r = svc.result(job_id)
        print(f"[cache_test] Job result: {r}")

        # access local cache and check for an entry
        cache = get_local_cache()
        if cache is None:
            print("[cache_test] Local cache not enabled (USE_LOCAL_SEMANTIC_CACHE=1?)")
            return
        entry = cache.get(track_id, current_frame=2)
        if entry is not None:
            print("[cache_test] Cache entry found:", entry)
        else:
            print("[cache_test] Worker did not write an entry to the local semantic cache")
    finally:
        svc.close()

if __name__ == "__main__":
    main()