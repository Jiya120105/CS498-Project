# BASIC TEST WITH REAL IMAGE
from PIL import Image
import io, base64, requests, time

# 1) Load your actual image
img = Image.open("game1.webp").convert("RGB")

# (optional) if the image is huge, shrink to speed up VLM
img.thumbnail((640, 640))

# 2) Encode to base64
buf = io.BytesIO(); img.save(buf, format="JPEG")
b64 = base64.b64encode(buf.getvalue()).decode()

# 3) POST /infer (whole image as the ROI)
r = requests.post("http://127.0.0.1:8008/infer", json={
    "frame_id": 1,
    "track_id": 101,
    "bbox": [0, 0, img.width, img.height],
    "image_b64": b64
})
job_id = r.json()["job_id"]
print("Job ID:", job_id)

# 4) Poll /result
for i in range(20):
    out = requests.get("http://127.0.0.1:8008/result", params={"job_id": job_id}).json()
    if out["status"] != "pending":
        break
    time.sleep(0.5)
    print(i)

print(out)  # expect {'status':'done','record':{...}}



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
