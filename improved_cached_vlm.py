"""
Improved Cached VLM with Real Vision Encoder Embeddings

Uses actual vision encoder features from SmolVLM layers 6-7
instead of simple 32×32 pixel embeddings.

This should provide much better semantic similarity matching.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
from PIL import Image
import numpy as np
from collections import OrderedDict
import time


class ImprovedVisionEmbeddingCache:
    """LRU cache using real vision encoder embeddings for similarity matching."""

    def __init__(self, max_size: int = 200, similarity_threshold: float = 0.93):
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold

        # Cache storage
        self.cache: OrderedDict[str, Dict] = OrderedDict()
        self.track_index: Dict[int, str] = {}

        # Statistics
        self.hits = 0
        self.misses = 0
        self.total_lookups = 0
        self.evictions = 0

    def _compute_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        """Compute cosine similarity between two embeddings."""
        # Flatten if needed
        if emb1.dim() > 2:
            emb1 = emb1.flatten(start_dim=1)
        if emb2.dim() > 2:
            emb2 = emb2.flatten(start_dim=1)

        # Mean pooling if sequence
        if emb1.dim() == 2 and emb1.size(0) > 1:
            emb1 = emb1.mean(dim=0, keepdim=True)
        if emb2.dim() == 2 and emb2.size(0) > 1:
            emb2 = emb2.mean(dim=0, keepdim=True)

        # Cosine similarity
        similarity = F.cosine_similarity(emb1, emb2, dim=-1)
        return similarity.item()

    def get(self, query_embedding: torch.Tensor, track_id: Optional[int] = None) -> Optional[str]:
        """
        Look up cached entry similar to query.

        Returns:
            Cached answer (label) if found, None otherwise
        """
        self.total_lookups += 1

        # Fast path: exact track_id match
        if track_id is not None and track_id in self.track_index:
            cache_key = self.track_index[track_id]
            if cache_key in self.cache:
                entry = self.cache[cache_key]
                # Move to end (LRU)
                self.cache.move_to_end(cache_key)
                entry['access_count'] += 1
                self.hits += 1
                return entry['answer']

        # Slow path: similarity search across all cached embeddings
        best_match = None
        best_similarity = -1.0

        for key, entry in self.cache.items():
            sim = self._compute_similarity(query_embedding, entry['embedding'])
            if sim > best_similarity:
                best_similarity = sim
                best_match = entry

        if best_match is not None and best_similarity >= self.similarity_threshold:
            # Cache hit via similarity!
            self.hits += 1
            best_match['access_count'] += 1
            # Move to end (LRU)
            for key, entry in self.cache.items():
                if entry is best_match:
                    self.cache.move_to_end(key)
                    break
            return best_match['answer']

        # Cache miss
        self.misses += 1
        return None

    def put(self, embedding: torch.Tensor, answer: str, track_id: Optional[int] = None):
        """Store an embedding and its answer in the cache."""
        # Generate cache key
        if track_id is not None:
            cache_key = f"track_{track_id}"
        else:
            cache_key = f"emb_{len(self.cache)}_{time.time()}"

        # Evict if at capacity
        if len(self.cache) >= self.max_size:
            evicted_key, evicted_entry = self.cache.popitem(last=False)
            self.evictions += 1
            # Remove from track index
            if evicted_entry.get('track_id') is not None:
                self.track_index.pop(evicted_entry['track_id'], None)

        # Create and store entry
        entry = {
            'track_id': track_id,
            'embedding': embedding.detach().clone() if isinstance(embedding, torch.Tensor) else embedding,
            'answer': answer,
            'timestamp': time.time(),
            'access_count': 0
        }

        self.cache[cache_key] = entry

        # Update track index
        if track_id is not None:
            self.track_index[track_id] = cache_key

    def get_stats(self) -> Dict[str, float]:
        """Get cache statistics."""
        hit_rate = (self.hits / self.total_lookups * 100) if self.total_lookups > 0 else 0.0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total_lookups": self.total_lookups,
            "hit_rate": hit_rate,
            "cache_size": len(self.cache),
            "evictions": self.evictions,
            "max_size": self.max_size
        }

    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        self.track_index.clear()
        self.hits = 0
        self.misses = 0
        self.total_lookups = 0
        self.evictions = 0


class ImprovedCachedVLM:
    """
    Improved cached VLM using real vision encoder embeddings.

    Extracts embeddings from SmolVLM vision encoder layers 6-7
    for better semantic similarity matching.
    """

    def __init__(self, base_model, processor, cache_size: int = 200,
                 similarity_threshold: float = 0.93, device: str = "cuda",
                 embedding_layers: list = [6, 7]):
        """
        Args:
            base_model: SmolVLM model
            processor: SmolVLM processor
            cache_size: Max cache entries
            similarity_threshold: Cosine similarity threshold (0.90-0.98)
            device: Device to run on
            embedding_layers: Which vision encoder layers to extract from
        """
        self.model = base_model
        self.processor = processor
        self.device = device
        self.embedding_layers = embedding_layers
        self.cache = ImprovedVisionEmbeddingCache(
            max_size=cache_size,
            similarity_threshold=similarity_threshold
        )

        # Hook storage for extracting embeddings
        self.layer_outputs = {}

        # Register hooks to extract vision encoder layer outputs
        self._register_hooks()

        # Statistics
        self.total_inferences = 0
        self.cache_overhead_ms = 0

    def _register_hooks(self):
        """Register forward hooks to extract vision encoder layer outputs."""
        vision_layers = self.model.model.vision_model.encoder.layers

        def get_hook(layer_idx):
            def hook(module, input, output):
                # Store the output (hidden states)
                self.layer_outputs[layer_idx] = output[0] if isinstance(output, tuple) else output
            return hook

        for layer_idx in self.embedding_layers:
            if layer_idx < len(vision_layers):
                vision_layers[layer_idx].register_forward_hook(get_hook(layer_idx))

    def _extract_vision_embedding(self, image: Image.Image) -> torch.Tensor:
        """
        Extract vision encoder embedding from specified layers.

        Returns a pooled embedding from layers 6-7 of vision encoder.
        """
        # Prepare image input (minimal processing for embedding extraction)
        inputs = self.processor(images=[image], return_tensors="pt")
        # Fix: Must include pixel_attention_mask for Idefics3/SmolVLM
        inputs = {k: v.to(self.device) for k, v in inputs.items() if k in ['pixel_values', 'pixel_attention_mask']}

        # Flatten inputs (Batch, NumImages, ...) -> (Batch*NumImages, ...)
        # The vision model expects flattened 4D inputs, while the processor returns 5D.
        if 'pixel_values' in inputs and inputs['pixel_values'].ndim == 5:
            # Fix: Flatten AND cast to model dtype (FP16)
            inputs['pixel_values'] = inputs['pixel_values'].flatten(start_dim=0, end_dim=1).to(self.model.dtype)
        
        if 'pixel_attention_mask' in inputs and inputs['pixel_attention_mask'].ndim == 4:
            inputs['pixel_attention_mask'] = inputs['pixel_attention_mask'].flatten(start_dim=0, end_dim=1)

        # Clear previous outputs
        self.layer_outputs.clear()

        # Forward pass through vision model only (not full VLM)
        with torch.no_grad():
            _ = self.model.model.vision_model(**inputs)

        # Combine embeddings from specified layers
        embeddings = []
        for layer_idx in self.embedding_layers:
            if layer_idx in self.layer_outputs:
                layer_out = self.layer_outputs[layer_idx]
                # Mean pool over sequence dimension
                pooled = layer_out.mean(dim=1)  # [batch, hidden_dim]
                embeddings.append(pooled)

        if len(embeddings) == 0:
            raise RuntimeError("Failed to extract vision embeddings")

        # Concatenate or average embeddings from multiple layers
        if len(embeddings) > 1:
            combined = torch.cat(embeddings, dim=-1)
        else:
            combined = embeddings[0]

        return combined

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

    def infer(self, image: Image.Image, prompt: str, track_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Run inference with vision embedding caching.

        Returns:
            Dict with 'label', 'confidence', 'metadata'
        """
        self.total_inferences += 1

        # Extract vision embedding for similarity check
        cache_start = time.perf_counter()
        vision_embedding = self._extract_vision_embedding(image)

        # Check cache for similar embedding
        cached_answer = self.cache.get(vision_embedding, track_id=track_id)
        cache_time = (time.perf_counter() - cache_start) * 1000
        self.cache_overhead_ms += cache_time

        if cached_answer is not None:
            # Cache hit!
            return {
                "label": cached_answer,
                "confidence": 0.95,
                "metadata": {
                    "cache_hit": True,
                    "from_similarity": True,
                    "cache_overhead_ms": cache_time
                }
            }

        # Cache miss - run actual VLM inference
        label, conf = self._run_actual_inference(image, prompt)

        # Store in cache
        self.cache.put(vision_embedding, label, track_id=track_id)

        return {
            "label": label,
            "confidence": conf,
            "metadata": {
                "cache_hit": False,
                "from_similarity": False,
                "cache_overhead_ms": cache_time
            }
        }

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = self.cache.get_stats()
        stats['total_cache_overhead_ms'] = self.cache_overhead_ms
        stats['avg_cache_overhead_ms'] = (self.cache_overhead_ms / self.total_inferences
                                          if self.total_inferences > 0 else 0)
        return stats

    def clear_cache(self):
        """Clear the cache."""
        self.cache.clear()
        self.cache_overhead_ms = 0
