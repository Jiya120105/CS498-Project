"""
Cached VLM Wrapper

Wraps SmolVLM to use vision embedding cache for repeated/similar ROIs.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from PIL import Image
import numpy as np
import hashlib
from .embedding_cache import VisionEmbeddingCache


class CachedSmolVLM:
    """SmolVLM wrapper with vision embedding caching."""

    def __init__(self, base_model, processor, cache_size: int = 200, similarity_threshold: float = 0.98,
                 device: str = "cuda", enable_background_validation: bool = True):
        """
        Initialize cached VLM.

        Args:
            base_model: The base SmolVLM model
            processor: The processor for the model
            cache_size: Max cache entries
            similarity_threshold: Similarity threshold for cache hits (default: 0.98)
            device: Device to run on
            enable_background_validation: If True, verify cached results in background
        """
        self.model = base_model
        self.processor = processor
        self.device = device
        self.cache = VisionEmbeddingCache(max_size=cache_size, similarity_threshold=similarity_threshold)
        self.enable_background_validation = enable_background_validation

        # Cache to store actual answers for validation
        self.answer_cache = {}  # track_id -> answer

        # Statistics
        self.cache_hit_speedup_ms = []
        self.total_inferences = 0
        self.background_validations = 0
        self.cache_corrections = 0
        self.cache_accuracy_matches = 0

        # Background validation queue
        self.validation_queue = []  # List of (image, prompt, track_id, cached_answer)

    def _compute_image_embedding(self, image: Image.Image) -> torch.Tensor:
        """
        Compute a simple embedding for the image using resized pixel values.

        This is much faster than running the vision encoder, and sufficient
        for detecting similar ROIs in video tracking.
        """
        # Resize to small size for fast comparison
        img_small = image.resize((32, 32), Image.Resampling.BILINEAR)

        # Convert to numpy and normalize
        img_array = np.array(img_small).astype(np.float32) / 255.0

        # Flatten to 1D vector
        img_flat = img_array.flatten()

        # Convert to torch tensor
        embedding = torch.from_numpy(img_flat).unsqueeze(0).to(self.device)

        return embedding

    def _run_actual_inference(self, image: Image.Image, prompt: str) -> tuple[str, float]:
        """Run actual VLM inference and return parsed answer."""
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=text_prompt, images=[image], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=50)

        generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]

        if "Assistant:" in generated_text:
            answer = generated_text.split("Assistant:")[-1].strip()
        else:
            answer = generated_text.strip()

        # Parse answer
        label = "Unknown"
        conf = 0.5

        lower_ans = answer.lower()
        if "yes" in lower_ans:
            label, conf = "Yes", 0.95
        elif "no" in lower_ans:
            label, conf = "No", 0.95
        else:
            label, conf = answer[:50], 0.5

        return label, conf

    def _validate_cache_entry_background(self, image: Image.Image, prompt: str, track_id: int, cached_answer: str):
        """
        Background validation: Run actual inference and update cache if answer differs.
        This should be called asynchronously/in background thread in production.
        """
        self.background_validations += 1

        # Run actual inference
        actual_label, actual_conf = self._run_actual_inference(image, prompt)

        # Compare with cached answer
        if actual_label == cached_answer:
            self.cache_accuracy_matches += 1
        else:
            # Cache was wrong! Update it
            self.cache_corrections += 1
            # Update answer cache
            if track_id is not None:
                self.answer_cache[track_id] = actual_label

            print(f"[Cache Correction] Track {track_id}: {cached_answer} → {actual_label}")

    def infer(self, image: Image.Image, prompt: str, track_id: Optional[int] = None,
              skip_background_validation: bool = False) -> Dict[str, Any]:
        """
        Run inference with vision embedding caching.

        Args:
            image: Input image
            prompt: Text prompt
            track_id: Optional track ID for cache indexing

        Returns:
            Dict with 'label', 'confidence', 'metadata', and 'cache_hit' flag
        """
        self.total_inferences += 1
        cache_hit = False
        use_cached_answer = False

        # Compute lightweight image embedding for similarity check
        query_embedding = self._compute_image_embedding(image)

        # Check if we have a cached answer for this track
        if track_id is not None and track_id in self.answer_cache:
            # Direct answer cache hit!
            cache_hit = True
            use_cached_answer = True
            label = self.answer_cache[track_id]
            conf = 0.95

            # Schedule background validation if enabled
            if self.enable_background_validation and not skip_background_validation:
                # In production, this would be added to a background queue/thread
                # For now, we'll add to a list and process periodically
                self.validation_queue.append((image.copy(), prompt, track_id, label))

                # Process validation queue periodically (every 50 items)
                if len(self.validation_queue) >= 50:
                    self._process_validation_queue()

            return {
                "label": label,
                "confidence": conf,
                "metadata": {"raw": label, "cache_hit": cache_hit, "from_answer_cache": True}
            }

        # Check similarity-based cache
        cached_embedding = self.cache.get(query_embedding, track_id=track_id)
        if cached_embedding is not None:
            cache_hit = True

        # Run actual VLM inference
        label, conf = self._run_actual_inference(image, prompt)

        # Store result in caches
        if not cache_hit:
            self.cache.put(query_embedding, track_id=track_id)

        # Store answer for future fast lookups
        if track_id is not None:
            self.answer_cache[track_id] = label

        return {
            "label": label,
            "confidence": conf,
            "metadata": {"raw": label, "cache_hit": cache_hit, "from_answer_cache": False}
        }

    def _process_validation_queue(self):
        """Process pending background validations (in real implementation, this runs in separate thread)."""
        print(f"\n[Background Validation] Processing {len(self.validation_queue)} items...")

        for image, prompt, track_id, cached_answer in self.validation_queue:
            self._validate_cache_entry_background(image, prompt, track_id, cached_answer)

        # Clear queue
        self.validation_queue.clear()

        # Print stats
        if self.background_validations > 0:
            accuracy = self.cache_accuracy_matches / self.background_validations * 100
            print(f"[Background Validation] Accuracy: {accuracy:.1f}% ({self.cache_accuracy_matches}/{self.background_validations})")
            print(f"[Background Validation] Corrections: {self.cache_corrections}")

    def get_cache_stats(self) -> Dict[str, float]:
        """Get cache statistics."""
        stats = self.cache.get_stats()
        stats.update({
            "answer_cache_size": len(self.answer_cache),
            "background_validations": self.background_validations,
            "cache_corrections": self.cache_corrections,
            "cache_accuracy": self.cache_accuracy_matches / self.background_validations * 100 if self.background_validations > 0 else 0.0
        })
        return stats

    def clear_cache(self):
        """Clear the vision embedding cache."""
        self.cache.clear()
        self.answer_cache.clear()
        self.validation_queue.clear()
        self.background_validations = 0
        self.cache_corrections = 0
        self.cache_accuracy_matches = 0
