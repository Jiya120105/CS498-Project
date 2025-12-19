# Query-Adaptive Layer Compression - Validation Plan

## Research Hypothesis

**Core Idea:** For repetitive video queries (same question, different ROIs), not all VLM layers contribute equally. We can identify less important layers during initial samples and compress them for subsequent inferences without significant accuracy loss.

## What We're Testing

### Experiment 1: Layer Importance Stability
**Question:** Is layer importance consistent across different ROIs for the same query?

**Method:**
- Extract 5-10 person ROIs from MOT16 frames
- Run SmolVLM inference on each with query: "Is this person with a backpack?"
- Hook into all 44 layers (12 vision + 32 text/LLaMA layers)
- Measure activation magnitudes per layer
- Compute pairwise correlation between importance vectors

**Success Criteria:**
- Mean correlation > 0.7 = HIGH stability → proceed with implementation
- Mean correlation 0.4-0.7 = MODERATE → may work but needs tuning
- Mean correlation < 0.4 = LOW stability → approach won't work

### Experiment 2: Compression Impact (Theoretical)
**Question:** What speedup/memory savings can we expect?

**Baseline Measurements:**
- Per-inference latency on CPU (ms)
- Memory footprint (MB)
- Semantic accuracy (% correct labels)

**Expected Results with Quantization:**
- Bottom 50% layers quantized (FP16 → INT8): ~1.3-1.5× speedup, ~30-40% memory reduction
- Bottom 75% layers quantized: ~1.8-2.0× speedup, ~50-60% memory reduction, may lose 5-10% accuracy

## Model Architecture (SmolVLM/Idefics3)

```
Idefics3ForConditionalGeneration
├── model.vision_model.encoder.layers (12 layers)
│   └── Vision Transformer layers processing image ROIs
├── model.text_model.layers (32 layers)
│   └── LLaMA decoder layers processing text + vision tokens
└── connector (fuses vision and text)
```

**Key Insight:** For simple Yes/No queries, we hypothesize that:
- Early vision layers (0-5) are critical (extracting basic features)
- Late vision layers (10-11) may be less critical
- Middle text layers (10-25) may have redundancy for simple queries
- Early and late text layers likely critical (prompt understanding + answer generation)

## If Validation Succeeds

### Next Steps:
1. Implement `AdaptiveSmolVLM` class in `loader.py`
2. Add profiling phase (K=10 samples) at system startup
3. Quantize unimportant layers after profiling
4. Integrate with `run_system.py`
5. Evaluate on manufacturing/retail scenarios

### Metrics to Track:
- Profiling overhead (time to analyze K samples)
- Compression ratio (memory saved)
- Speedup (latency improvement)
- Accuracy preservation (semantic correctness)
- Cache hit rate (how often same layers are unimportant)

## If Validation Fails

### Fallback Options:
1. **Static compression:** Pre-analyze which layers are generally unimportant (across many queries) and always compress them
2. **Query complexity heuristic:** Use query length/structure to predict which layers to compress (simple query = more aggressive)
3. **Hybrid approach:** Combine with embedding cache (Direction 1 from earlier discussion)
4. **Batch optimization:** Focus on cross-track KV sharing instead (Direction 2)

## Current Status

Running validation script with:
- Device: CPU (for initial test)
- Samples: 5 ROIs from MOT16-04
- Query: "Is this person with a backpack? Answer Yes or No."
- Layers analyzed: 12 vision + 32 text = 44 total

**Expected runtime:** 3-5 minutes on CPU

**Output files:**
- `validation_results.json` - Raw importance data and answers
- `layer_importance.png` - Visualization of which layers matter most

## Decision Point

After reviewing results, we'll decide:
- ✅ **Correlation > 0.7:** Proceed with full implementation
- ⚠️ **Correlation 0.4-0.7:** Run more tests with different queries before deciding
- ❌ **Correlation < 0.4:** Pivot to alternative approach

