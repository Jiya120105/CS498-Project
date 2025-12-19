# BASIC TEST WITH REAL IMAGE
import time, requests, base64, io
from PIL import Image

img = Image.open("slow_path/tests/000004.jpg").convert("RGB")
img.thumbnail((640, 640))
buf = io.BytesIO(); img.save(buf, format="JPEG")
b64 = base64.b64encode(buf.getvalue()).decode()

t0 = time.perf_counter()

r = requests.post("http://127.0.0.1:8008/infer",
                  json={"frame_id":1,"track_id":101,"bbox":[0,0,img.width,img.height],"image_b64":b64},
                  timeout=(2, 60))
job_id = r.json()["job_id"]

out = None
while True:
    out = requests.get("http://127.0.0.1:8008/result",
                       params={"job_id": job_id},
                       timeout=(2, 20)).json()
    if out["status"] != "pending":
        break
    time.sleep(0.5)

t1 = time.perf_counter()
print(f"Wall time (submit → done): {(t1 - t0)*1000:.1f} ms")
print(out)


# TEST TRIGGER - CHANGE COOLDOWN OR GAP
import io, base64, requests, time
from PIL import Image, ImageDraw

def b64(im):
    buf = io.BytesIO(); im.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()

# Make two frames (320x240). Same ROI box [50,60,70,120].
f1 = Image.new("RGB",(320,240),(40,40,40))
d1 = ImageDraw.Draw(f1)
d1.rectangle([50,60,120,180], outline=(200,200,200), width=3)

# Second frame: fill the ROI area with a bright color to guarantee big diff
f2 = f1.copy()
d2 = ImageDraw.Draw(f2)
d2.rectangle([50,60,120,180], fill=(255,50,50))  # big change inside ROI

roi = {"track_id": 7, "bbox": [50,60,70,120]}

# 1) Frame 15 should trigger due to periodic every_N=15
r = requests.post("http://127.0.0.1:8008/trigger/tick", json={
    "frame_id": 15, "image_b64": b64(f1), "rois": [roi]
})
print("tick15 status:", r.status_code)
try:
    print("tick15 json:", r.json())
except Exception:
    print("tick15 text:", r.text)

# 2) Frame 18 should trigger due to scene change (big ROI difference)
r = requests.post("http://127.0.0.1:8008/trigger/tick", json={
    "frame_id": 18, "image_b64": b64(f2), "rois": [roi]
})
print("tick18 status:", r.status_code)
try:
    print("tick18 json:", r.json())
except Exception:
    print("tick18 text:", r.text)

# Optional: give the worker a moment and check any result
time.sleep(1.5)



# TESTING CACHE 
from PIL import Image
import io, base64, requests, time

# 1) server up: uvicorn service.api:app --port 8008
# 2) cache stub up: python cache_stub.py (port 8010)
img = Image.new("RGB",(128,128),(180,60,60))
buf = io.BytesIO(); img.save(buf, format="JPEG")
b64 = base64.b64encode(buf.getvalue()).decode()

r = requests.post("http://127.0.0.1:8008/infer", json={
    "frame_id": 1, "track_id": 42, "bbox":[0,0,128,128], "image_b64": b64
})
job_id = r.json()["job_id"]

for i in range(20):
    out = requests.get("http://127.0.0.1:8008/result", params={"job_id": job_id}).json()
    if out["status"] != "pending":
        break
    time.sleep(0.2)
    print(i)

print("worker result:", out)
print("cache stats:", requests.get("http://127.0.0.1:8010/cache/stats").json())
