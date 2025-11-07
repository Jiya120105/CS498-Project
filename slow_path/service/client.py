import os, time, json, httpx
CACHE_BASE = os.getenv("CACHE_BASE_URL", "http://127.0.0.1:8010")

def post_cache_record(record: dict, timeout=2.0, retries=3) -> bool:
    url = f"{CACHE_BASE}/cache/put"
    for i in range(retries):
        try:
            r = httpx.post(url, json=record, timeout=timeout)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.2 * (2**i))  # backoff
    return False
