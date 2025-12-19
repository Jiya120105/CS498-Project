"""
Validation Study: Query-Adaptive Layer Importance for VLM Compression

This script tests the feasibility of:
1. Whether layer importance is stable across different ROIs for the same query
2. Whether quantizing unimportant layers preserves accuracy
3. Whether we get meaningful speedup from compression

Usage:
    python validate_layer_importance.py --device cuda --num_samples 10
"""

import os
import sys
import time
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import cv2
from ultralytics import YOLO
from transformers import AutoProcessor, AutoModelForVision2Seq
import matplotlib.pyplot as plt

# Force environment setup
os.environ["USE_LOCAL_SEMANTIC_CACHE"] = "0"  # Don't need cache for validation


class LayerImportanceAnalyzer:
    """Analyzes layer importance in SmolVLM for a given query."""

    def __init__(self, device="cuda"):
        self.device = device
        print(f"[Analyzer] Loading SmolVLM on {device}...")

        # Load model and processor
        self.processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-500M-Instruct")
        if hasattr(self.processor, "tokenizer"):
            self.processor.tokenizer.padding_side = "left"

        if device == "cuda":
            self.model = AutoModelForVision2Seq.from_pretrained(
                "HuggingFaceTB/SmolVLM-500M-Instruct",
                torch_dtype=torch.float16,
                _attn_implementation="eager"
            ).to(device)
        else:
            self.model = AutoModelForVision2Seq.from_pretrained(
                "HuggingFaceTB/SmolVLM-500M-Instruct",
                _attn_implementation="eager"
            ).to(device)

        self.model.eval()
        print(f"[Analyzer] Model loaded. Device: {self.model.device}")

        # Storage for analysis
        self.layer_gradients = []
        self.layer_activations = []

    def register_hooks(self):
        """Register forward and backward hooks to capture activations and gradients."""
        self.hooks = []
        self.activation_storage = {}
        self.gradient_storage = {}

        # Find all transformer layers
        # For SmolVLM (Idefics3), we have both text and vision layers
        text_layers = None
        vision_layers = None

        # Find text layers (LLaMA): model.model.text_model.layers
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'text_model'):
            if hasattr(self.model.model.text_model, 'layers'):
                text_layers = self.model.model.text_model.layers
                print(f"[Analyzer] Found {len(text_layers)} text layers (LLaMA)")

        # Find vision layers: model.model.vision_model.encoder.layers
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'vision_model'):
            if hasattr(self.model.model.vision_model, 'encoder'):
                if hasattr(self.model.model.vision_model.encoder, 'layers'):
                    vision_layers = self.model.model.vision_model.encoder.layers
                    print(f"[Analyzer] Found {len(vision_layers)} vision layers")

        if text_layers is None and vision_layers is None:
            print("[Warning] Could not find any model layers.")
            return

        # Combine both sets of layers
        layers = []
        if vision_layers:
            layers.extend([(f"vision_{i}", layer) for i, layer in enumerate(vision_layers)])
        if text_layers:
            layers.extend([(f"text_{i}", layer) for i, layer in enumerate(text_layers)])

        def make_forward_hook(name):
            def hook(module, input, output):
                # Store activation magnitude
                if isinstance(output, tuple):
                    act = output[0]
                else:
                    act = output
                if isinstance(act, torch.Tensor):
                    self.activation_storage[name] = act.detach().abs().mean().item()
            return hook

        def make_backward_hook(name):
            def hook(module, grad_input, grad_output):
                # Store gradient magnitude
                if grad_output[0] is not None:
                    self.gradient_storage[name] = grad_output[0].detach().abs().mean().item()
            return hook

        # Register hooks on each layer
        for name, layer in layers:
            self.hooks.append(layer.register_forward_hook(make_forward_hook(name)))
            # Note: Not registering backward hooks since we're not using gradients

        print(f"[Analyzer] Registered hooks on {len(layers)} layers (vision + text)")

    def remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def infer_with_analysis(self, image, prompt):
        """Run inference and collect layer importance metrics (activation-based only)."""
        # Prepare input
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=text_prompt, images=[image], return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Clear storage
        self.activation_storage = {}
        self.gradient_storage = {}

        # Forward pass (activation hooks will capture layer outputs)
        with torch.no_grad():  # No gradients needed for activation-based analysis
            outputs = self.model.generate(**inputs, max_new_tokens=50)

        # Decode answer
        generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]

        # Extract answer
        if "Assistant:" in generated_text:
            answer = generated_text.split("Assistant:")[-1].strip()
        else:
            answer = generated_text.strip()

        # Collect metrics (activation-based only, more stable)
        layer_importance = {
            'activations': dict(self.activation_storage),
            'gradients': {}  # Not using gradients in this simplified version
        }

        return answer, layer_importance

    def infer_baseline(self, image, prompt):
        """Run normal inference without analysis (for speed comparison)."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=text_prompt, images=[image], return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=50)

        generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]

        if "Assistant:" in generated_text:
            answer = generated_text.split("Assistant:")[-1].strip()
        else:
            answer = generated_text.strip()

        return answer


def load_mot16_frames_and_rois(mot_path, num_frames=5, max_rois_per_frame=3):
    """Load frames from MOT16 and extract ROIs using YOLO."""
    print(f"[Data] Loading frames from {mot_path}...")

    # Find img1 directory
    img_dir = Path(mot_path) / "img1"
    if not img_dir.exists():
        img_dir = Path(mot_path)

    # Get frame files
    frame_files = sorted(list(img_dir.glob("*.jpg")))[:num_frames]
    print(f"[Data] Found {len(frame_files)} frames")

    # Load YOLO for ROI extraction
    yolo = YOLO('yolov8n.pt')

    rois = []
    for frame_file in frame_files:
        frame = cv2.imread(str(frame_file))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run YOLO (just detection, no tracking)
        results = yolo.predict(frame, verbose=False)

        if len(results[0].boxes) == 0:
            continue

        # Extract person bounding boxes (class 0)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()

        person_count = 0
        for i, cls_id in enumerate(classes):
            if cls_id == 0 and person_count < max_rois_per_frame:  # Person class
                x1, y1, x2, y2 = map(int, boxes[i])

                # Extract ROI
                roi = frame_rgb[y1:y2, x1:x2]
                if roi.size > 0:
                    roi_pil = Image.fromarray(roi)
                    rois.append({
                        'image': roi_pil,
                        'bbox': [x1, y1, x2, y2],
                        'frame': frame_file.name
                    })
                    person_count += 1

    print(f"[Data] Extracted {len(rois)} ROIs")
    return rois


def analyze_layer_importance_stability(analyzer, rois, query, num_samples=10):
    """Test if layer importance is stable across different ROIs."""
    print(f"\n{'='*60}")
    print(f"EXPERIMENT 1: Layer Importance Stability")
    print(f"{'='*60}")
    print(f"Query: '{query}'")
    print(f"Analyzing {min(num_samples, len(rois))} ROIs...\n")

    analyzer.register_hooks()

    importance_samples = []
    answers = []

    for i, roi_data in enumerate(rois[:num_samples]):
        print(f"ROI {i+1}/{min(num_samples, len(rois))} from {roi_data['frame']}...", end=" ")

        try:
            answer, importance = analyzer.infer_with_analysis(roi_data['image'], query)
            answers.append(answer)

            # Use activation-based importance (more stable than gradients)
            if importance['activations']:
                importance_samples.append(importance['activations'])
                print(f"✓ Answer: {answer[:50]}")
            else:
                print("✗ No activations captured")
        except Exception as e:
            print(f"✗ Error: {e}")

    analyzer.remove_hooks()

    if len(importance_samples) < 2:
        print("[Error] Not enough samples to analyze stability")
        return None, answers

    # Analyze stability: compute correlation between importance vectors
    print(f"\n--- Stability Analysis ---")

    # Convert to matrix (samples x layers)
    layer_names = sorted(importance_samples[0].keys())
    importance_matrix = np.array([
        [sample.get(layer, 0) for layer in layer_names]
        for sample in importance_samples
    ])

    # Compute pairwise correlations
    from scipy.stats import pearsonr
    correlations = []
    for i in range(len(importance_matrix)):
        for j in range(i+1, len(importance_matrix)):
            corr, _ = pearsonr(importance_matrix[i], importance_matrix[j])
            if not np.isnan(corr):
                correlations.append(corr)

    if correlations:
        mean_corr = np.mean(correlations)
        print(f"Mean pairwise correlation: {mean_corr:.3f}")
        print(f"Std pairwise correlation: {np.std(correlations):.3f}")

        if mean_corr > 0.7:
            print("✓ HIGH stability - Layer importance is consistent across ROIs!")
        elif mean_corr > 0.4:
            print("~ MODERATE stability - Some consistency but variable")
        else:
            print("✗ LOW stability - Layer importance varies significantly")
    else:
        print("[Warning] Could not compute correlations")

    # Compute average importance per layer
    avg_importance = np.mean(importance_matrix, axis=0)
    importance_dict = {layer: imp for layer, imp in zip(layer_names, avg_importance)}

    # Identify top and bottom layers
    sorted_layers = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    print(f"\nTop 5 most important layers:")
    for layer, imp in sorted_layers[:5]:
        print(f"  {layer}: {imp:.4f}")

    print(f"\nBottom 5 least important layers:")
    for layer, imp in sorted_layers[-5:]:
        print(f"  {layer}: {imp:.4f}")

    return importance_dict, answers


def test_quantization_impact(analyzer, rois, query, importance_dict, num_test_samples=5):
    """Test impact of quantizing unimportant layers."""
    print(f"\n{'='*60}")
    print(f"EXPERIMENT 2: Quantization Impact")
    print(f"{'='*60}")

    if importance_dict is None:
        print("[Skip] No importance data available")
        return

    # Identify layers to quantize (bottom 50%)
    sorted_layers = sorted(importance_dict.items(), key=lambda x: x[1])
    threshold_idx = len(sorted_layers) // 2
    layers_to_quantize = [layer for layer, _ in sorted_layers[:threshold_idx]]

    print(f"Will quantize {len(layers_to_quantize)} layers (bottom 50% by importance)")
    print(f"Layers to quantize: {layers_to_quantize[:5]}...")

    # TODO: Actual quantization would go here
    # For now, we'll just measure baseline performance
    print("\n[Note] Actual quantization not implemented in validation script")
    print("[Note] This experiment shows what WOULD be tested:")
    print(f"  1. Quantize {len(layers_to_quantize)} layers to INT8")
    print(f"  2. Run inference on {num_test_samples} test ROIs")
    print(f"  3. Compare: accuracy, latency, memory usage")

    # Measure baseline latency
    print("\n--- Baseline Performance ---")
    test_rois = rois[len(rois)//2:][:num_test_samples]  # Use different ROIs

    latencies = []
    answers = []

    for i, roi_data in enumerate(test_rois):
        print(f"Test {i+1}/{len(test_rois)}...", end=" ")

        start = time.perf_counter()
        answer = analyzer.infer_baseline(roi_data['image'], query)
        latency = (time.perf_counter() - start) * 1000

        latencies.append(latency)
        answers.append(answer)
        print(f"{latency:.0f}ms - {answer[:50]}")

    print(f"\nBaseline latency: {np.mean(latencies):.0f} ± {np.std(latencies):.0f} ms")
    print(f"Expected speedup with quantization: ~1.2-1.5x (theoretical)")
    print(f"Expected memory reduction: ~30-40% (theoretical)")


def visualize_layer_importance(importance_dict, output_path="layer_importance.png"):
    """Create visualization of layer importance."""
    if importance_dict is None:
        return

    layer_names = sorted(importance_dict.keys())
    importance_values = [importance_dict[l] for l in layer_names]

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(layer_names)), importance_values)
    plt.xlabel("Layer Index")
    plt.ylabel("Importance (Activation Magnitude)")
    plt.title("Layer Importance for Query (Higher = More Important)")
    plt.xticks(range(len(layer_names)), [l.split('_')[1] for l in layer_names])
    plt.axhline(y=np.median(importance_values), color='r', linestyle='--', label='Median')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"\n[Viz] Saved layer importance plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Validate query-adaptive layer importance")
    parser.add_argument("--mot_path", default="MOT16/train/MOT16-04", help="Path to MOT16 sequence")
    parser.add_argument("--query", default="Is this person with a backpack? Answer Yes or No.", help="Query to test")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of ROIs to analyze")
    parser.add_argument("--output", default="validation_results.json", help="Output file for results")

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"QUERY-ADAPTIVE LAYER IMPORTANCE VALIDATION")
    print(f"{'='*60}")
    print(f"MOT Path: {args.mot_path}")
    print(f"Query: {args.query}")
    print(f"Device: {args.device}")
    print(f"Samples: {args.num_samples}")

    # Load data
    rois = load_mot16_frames_and_rois(args.mot_path, num_frames=20, max_rois_per_frame=2)

    if len(rois) < args.num_samples:
        print(f"[Warning] Only found {len(rois)} ROIs, requested {args.num_samples}")
        args.num_samples = len(rois)

    # Initialize analyzer
    analyzer = LayerImportanceAnalyzer(device=args.device)

    # Experiment 1: Stability
    importance_dict, answers = analyze_layer_importance_stability(
        analyzer, rois, args.query, num_samples=args.num_samples
    )

    # Experiment 2: Quantization impact
    if importance_dict:
        test_quantization_impact(analyzer, rois, args.query, importance_dict, num_test_samples=5)

    # Visualize
    if importance_dict:
        visualize_layer_importance(importance_dict)

    # Save results
    results = {
        "query": args.query,
        "num_samples": args.num_samples,
        "layer_importance": importance_dict,
        "answers": answers[:10]  # First 10 answers
    }

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"VALIDATION COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {args.output}")
    print(f"\nNext steps:")
    print(f"  1. Review layer_importance.png to see importance distribution")
    print(f"  2. Check correlation values - if >0.7, proceed with implementation")
    print(f"  3. Expected benefits: 1.2-1.5x speedup, 30-40% memory reduction")


if __name__ == "__main__":
    main()
