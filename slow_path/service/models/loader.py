import time
import json
from typing import Literal, Dict, Any
from transformers import BlipProcessor, BlipForConditionalGeneration, Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import torch
from ..json_utils import try_parse_json, coerce_to_schema, normalize_label, extract_categories
from ..categories import CATEGORIES

ModelName = Literal["mock", "blip"]  # add "blip2", "minigpt4" later

class MockVLM:
    def __init__(self, latency_ms: int = 40): self.latency_ms = latency_ms
    def infer(self, image: Image.Image, prompt: str) -> Dict[str, Any]:
        time.sleep(self.latency_ms / 1000.0)
        return {"label":"player","confidence":0.92,"metadata":{"jersey_color":"blue"}}
    
SYSTEM_INSTR = (
    "Describe the image briefly, then format the answer as a single JSON object "
    "with keys: label, confidence, metadata. "
    "Allowed labels: " + ", ".join(CATEGORIES) + ". "
    "For example: {\"label\":\"player\",\"confidence\":0.9,\"metadata\":{\"desc\":\"a man kicking a ball\"}}. "
    "Return only the JSON object."
)


prompt = 'Identify what is in the image and output only JSON with keys: label, confidence, metadata.'


class Blip2FlanModel:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-flan-t5-xl",
            torch_dtype=torch.float16
        ).to(self.device)

    def infer(self, image, prompt: str = None):
        full_prompt = SYSTEM_INSTR if prompt is None else prompt + "\n" + SYSTEM_INSTR
        inputs = self.processor(image, full_prompt, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=80, do_sample=False)
        text = self.processor.decode(out[0], skip_special_tokens=True).strip()
        return {"label": "background", "confidence": 0.5, "metadata": {"raw": text}}



class BlipModel:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(self.device)
    
    def infer(self, image: Image.Image, prompt: str):
        # don't pass an instruction prompt to captioning model
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=20, do_sample=False)
        caption = self.processor.decode(out[0], skip_special_tokens=True)

        labels = extract_categories(caption)
        label = labels[0] if labels is not None else 'background'
        conf = 0.7 if label != "background" else 0.6
        return {"label": label, "confidence": conf, "metadata": {"raw": caption, "labels": labels}}



def load_model(name: ModelName = "mock"):
    if name == "mock":
        return MockVLM()
    if name == "blip":
        return BlipModel()
    if name == "blip2":
        return Blip2FlanModel()
    raise ValueError(f"Unknown model: {name}")