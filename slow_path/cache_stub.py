from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any
import uvicorn, time

app = FastAPI(title="CacheStub")
STORE = []  # append-only for demo

class CacheRecord(BaseModel):
    track_id: int; label: str; bbox: list[int]; confidence: float
    timestamp: int; ttl: int; metadata: Dict[str, Any] = {}

@app.post("/cache/put")
def cache_put(rec: CacheRecord):
    STORE.append({"ts": time.time(), **rec.model_dump()})
    return {"ok": True, "size": len(STORE)}

@app.get("/cache/stats")
def stats():
    return {"count": len(STORE), "last": (STORE[-1] if STORE else None)}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8010)
