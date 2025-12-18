import time, torch
from typing import Literal, Dict, Any
from transformers import BlipProcessor, BlipForConditionalGeneration, Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
from ..json_utils import extract_categories, json_extraction
from ..categories import CATEGORIES
from transformers import AutoProcessor, MllamaForConditionalGeneration

ModelName = Literal["mock", "blip", "blip2", "llama"]  # add "blip2", "minigpt4" later

class MockVLM:
    def __init__(self, latency_ms: int = 40): self.latency_ms = latency_ms
    def infer(self, image: Image.Image, prompt: str) -> Dict[str, Any]:
        time.sleep(self.latency_ms / 1000.0)
        return {"label":"player","confidence":0.92,"metadata":{"jersey_color":"blue"}}

SYSTEM_INSTR = (
    "Describe the image briefly, then format the answer as a single JSON object "
    "with keys: label, confidence, metadata. "
    "For example: {\"label\":\"pedestrian\",\"confidence\":0.9,\"metadata\":{\"desc\":\"a man kicking a ball\"}}. "
    "Return only the JSON object."
)

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
    
class LlamaModel:
    def __init__(self, device=None):
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        model_name="meta-llama/Llama-3.2-11B-Vision-Instruct"
        # model_name = "meta-llama/Llama-3.1-8B-Vision-Instruct"

        dtype = torch.float32 if self.device == "mps" else torch.float16 # handling precision for mps

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map = None
        ).to(self.device).eval()

    def infer(self, image, prompt, max_new_tokens=256, temperature=0.2):
        # Build messages via chat template (system + user with an image)
        messages = [
            {"role": "system", "content": SYSTEM_INSTR},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt or "Describe the image."}
            ]}
        ]
        chat = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        inputs = self.processor(
            text=chat,
            images=image,
            return_tensors="pt",
            text_kwargs={"add_special_tokens": False}
        )
        inputs = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                  for k, v in inputs.items()}

        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )

        # Decode whole sequence and strip everything before the assistant turn.
        decoded = self.processor.batch_decode(out, skip_special_tokens=True)[0]

        # Heuristic: split at the last assistant tag if present
        for tag in ["<|assistant|>", "assistant\n", "Assistant:"]:
            if tag in decoded:
                decoded = decoded.split(tag)[-1].strip()
        
        return json_extraction(decoded)

def load_model(name: ModelName = "mock"):
    if name == "mock":
        return MockVLM()
    if name == "blip":
        return BlipModel()
    if name == "blip2":
        return Blip2FlanModel()
    if name == "llama":
        return LlamaModel()
    raise ValueError(f"Unknown model: {name}")