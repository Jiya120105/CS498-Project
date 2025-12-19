import json
import os
import torch
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor
from adaptive_int8_vlm import AdaptiveINT8VLM

DATASET_DIR = Path("mot16_dataset")
METADATA_FILE = DATASET_DIR / "dataset_metadata.json"
GROUND_TRUTH_FILE = DATASET_DIR / "ground_truth.json"
QUERY = "Is this person with a backpack? Answer Yes or No."

def load_dataset():
    if not METADATA_FILE.exists() or not GROUND_TRUTH_FILE.exists():
        raise FileNotFoundError("Dataset not found. Run create_mot16_dataset.py first.")

    with open(METADATA_FILE, 'r') as f:
        metadata = json.load(f)
    
    with open(GROUND_TRUTH_FILE, 'r') as f:
        gt = json.load(f)
    
    tracks = metadata['tracks']
    tracks.sort(key=lambda x: (x['video'], x['frame_id']))
    
    return tracks, gt['tracks']

def main():
    print("BENCHMARK: ADAPTIVE INT8 QUANTIZATION")
    print("Loading dataset...")
    tracks, ground_truth = load_dataset()
    print(f"✓ Loaded {len(tracks)} tracks")

    print("\nInitializing AdaptiveINT8VLM...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-500M-Instruct")
    
    vlm = AdaptiveINT8VLM(
        processor, 
        device=device,
        profiling_samples=10,
        inference_samples=50,
        quantization_ratio=0.5,
        correlation_threshold=0.95
    )

    correct = 0
    total = 0
    
    print("\nStarting Inference Stream...")

    for i, track in enumerate(tracks):
        track_id = track['track_id']
        roi_path = DATASET_DIR / track['roi_path']
        
        if not roi_path.exists():
            continue
            
        image = Image.open(roi_path).convert("RGB")
        result = vlm.infer(image, QUERY, track_id=track_id)
        
        if track_id in ground_truth:
            gt_label = ground_truth[track_id]['label']
            if result['label'] == gt_label:
                correct += 1
            total += 1
        
        phase = result['metadata'].get('phase', 'unknown')
        
        if (i + 1) % 10 == 0:
            stats = vlm.get_stats()
            print(f"Track {i+1}/{len(tracks)} | Phase: {phase} | Acc: {correct/total*100:.1f}% | Quantized: {stats['quantized_layers']} layers")

    print("FINAL RESULTS")
    stats = vlm.get_stats()
    accuracy = correct / total * 100 if total > 0 else 0
    
    print(f"Total Tracks:      {len(tracks)}")
    print(f"Accuracy:          {accuracy:.1f}%")
    print(f"Re-profilings:     {stats['re_profiling_count']}")
    print(f"Re-quantizations:  {stats['re_quantization_count']}")
    print(f"Avg Correlation:   {stats['avg_correlation']:.4f}")
    
    output = {
        "accuracy": accuracy,
        "total_tracks": len(tracks),
        "stats": stats
    }
    
    with open("adaptive_quantization_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nResults saved to adaptive_quantization_results.json")

if __name__ == "__main__":
    main()
