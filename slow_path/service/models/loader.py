import time, json, re, math, torch, os
from typing import Literal, Dict, Any
from transformers import BlipProcessor, BlipForConditionalGeneration, Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
from ..json_utils import try_parse_json, coerce_to_schema, normalize_label, extract_categories
from ..categories import CATEGORIES
from transformers import AutoModelForVision2Seq, AutoProcessor, MllamaForConditionalGeneration

ModelName = Literal["mock", "blip", "blip2", "llama"]  # add "blip2", "minigpt4" later

class MockVLM:
    def __init__(self, latency_ms: int = 500): self.latency_ms = latency_ms
    def infer(self, image: Image.Image, prompt: str) -> Dict[str, Any]:
        time.sleep(self.latency_ms / 1000.0)
        return {"label":"player","confidence":0.92,"metadata":{"jersey_color":"blue"}}
    def batch_infer(self, images, prompts):
        return [self.infer(im, pr) for im, pr in zip(images, prompts)]

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
    def batch_infer(self, images, prompts):
        return [self.infer(im, pr) for im, pr in zip(images, prompts)]


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
    def batch_infer(self, images, prompts):
        return [self.infer(im, pr) for im, pr in zip(images, prompts)]
    
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

        # With device_map="auto", DO NOT manually move inputs to a single device.
        # Accelerate will dispatch correctly from CPU tensors.
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
        
        # --- Robust JSON extraction & normalization ---
        # Grab the first JSON object in the string (handles any stray tokens)
        m = re.search(r"\{.*\}", decoded, flags=re.DOTALL)
        s = m.group(0) if m else decoded.strip()

        try:
            obj = json.loads(s)
        except Exception:
            # Fallback if the model didn't return valid JSON
            obj = {"label": s[:64], "confidence": 0.0, "metadata": {"raw": s}}

        # Normalize fields
        label = str(obj.get("label", "")).strip()
        conf = obj.get("confidence", 0.0)
        try:
            conf = float(conf)
        except Exception:
            conf = 0.0
        # clamp to [0,1]
        if math.isnan(conf) or math.isinf(conf):
            conf = 0.0
        conf = max(0.0, min(1.0, conf))

        metadata = obj.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {"meta": str(metadata)}

        return {"label": label, "confidence": conf, "metadata": metadata}
    def batch_infer(self, images, prompts):
        return [self.infer(im, pr) for im, pr in zip(images, prompts)]

class SmolVLM:
    def __init__(self, device=None):
        # Allow explicit device override via env SLOWPATH_DEVICE (e.g., 'cpu', 'cuda', 'cuda:1')
        dev_override = os.getenv("SLOWPATH_DEVICE")
        if device is None and dev_override:
            device = dev_override
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[SmolVLM] Loading model on {self.device}...")
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-500M-Instruct")
        # For decoder-only models, use left padding to avoid warning and ensure correct generation
        try:
            if hasattr(self.processor, "tokenizer") and hasattr(self.processor.tokenizer, "padding_side"):
                self.processor.tokenizer.padding_side = "left"
        except Exception:
            pass
        
        # Load Model
        # Optimized loading: 500M model is small enough for FP16 on almost any GPU.
        # Avoid bitsandbytes overhead for small models.
        if self.device == "cuda":
            print("[SmolVLM] Loading in float16 on CUDA (fast path)...")
            self.model = AutoModelForVision2Seq.from_pretrained(
                "HuggingFaceTB/SmolVLM-500M-Instruct",
                torch_dtype=torch.float16,
                _attn_implementation="eager"
            ).to(self.device)
        else:
            # CPU fallback
            self.model = AutoModelForVision2Seq.from_pretrained(
                "HuggingFaceTB/SmolVLM-500M-Instruct",
                _attn_implementation="eager"
            ).to(self.device)
            
        print(f"[SmolVLM] Model loaded. Device: {self.model.device}")

    def infer(self, image: Image.Image, prompt: str) -> Dict[str, Any]:
        # Construct message
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt or "Describe this image."}
                ]
            },
        ]
        
        # Preprocess
        text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=text_prompt, images=[image], return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate
        generated_ids = self.model.generate(**inputs, max_new_tokens=50)
        generated_texts = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )
        
        # Parse result (The model returns the full prompt + answer usually, or just answer depending on template)
        # SmolVLM instruction tuned usually returns the Assistant response.
        raw_text = generated_texts[0]
        
        # Simple heuristic to extract the answer part if it repeats prompt
        if "Assistant:" in raw_text:
            answer = raw_text.split("Assistant:")[-1].strip()
        else:
            answer = raw_text.strip()
            
        # Extract label (Yes/No or Category)
        # We assume the prompt asked for "Yes or No"
        label = "Unknown"
        conf = 0.0
        
        lower_ans = answer.lower()
        if "yes" in lower_ans:
            label = "Yes"
            conf = 0.95
        elif "no" in lower_ans:
            label = "No"
            conf = 0.95
        else:
            # If it's a description, treat as label
            label = answer[:50] # truncated
            conf = 0.5

        return {"label": label, "confidence": conf, "metadata": {"raw": answer}}

    def batch_infer(self, images, prompts):
        # Build chat prompts per sample
        messages_list = [
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": pr or "Describe this image."}]}]
            for pr in prompts
        ]
        chats = [self.processor.apply_chat_template(m, add_generation_prompt=True) for m in messages_list]
        inputs = self.processor(text=chats, images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        gen_ids = self.model.generate(**inputs, max_new_tokens=50)
        texts = self.processor.batch_decode(gen_ids, skip_special_tokens=True)

        outs = []
        for raw_text in texts:
            answer = raw_text.split("Assistant:")[-1].strip() if "Assistant:" in raw_text else raw_text.strip()
            lower_ans = answer.lower()
            label = "Unknown"; conf = 0.0
            if "yes" in lower_ans:
                label, conf = "Yes", 0.95
            elif "no" in lower_ans:
                label, conf = "No", 0.95
            else:
                label, conf = answer[:50], 0.5
            outs.append({"label": label, "confidence": conf, "metadata": {"raw": answer}})
        return outs

def load_model(name: ModelName = "mock"):
    if name == "mock":
        return MockVLM()
    if name == "blip":
        return BlipModel()
    if name == "blip2":
        return Blip2FlanModel()
    if name == "llama":
        return LlamaModel()
    if name == "smol":
        return SmolVLM()
    raise ValueError(f"Unknown model: {name}")
