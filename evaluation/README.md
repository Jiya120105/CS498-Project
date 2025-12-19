# Evaluation

This directory contains the evaluation framework for the MOT tracking system.

## Quick Start

### Basic Usage

```bash
# Evaluate a single sequence
python -m evaluation.evaluate --sequence MOT16-02

# Evaluate all sequences
python -m evaluation.evaluate

# Test mode (no downloads, processes 10 frames)
python -m evaluation.evaluate --test --sequence MOT16-02
```

### Using the Helper Script

```bash
# Evaluate single sequence
./run_eval.sh MOT16-02

# Evaluate all sequences
./run_eval.sh

# Test mode
./run_eval.sh MOT16-02 data results test
```

## Command Line Options

- `--data_root`: Root directory containing MOT16/17 data (default: `data`)
- `--sequence`: Specific sequence to evaluate (e.g., `MOT16-02`). If omitted, evaluates all sequences
- `--split`: Dataset split - `train` or `test` (default: `train`)
- `--output_dir`: Output directory for results (default: `results`)
- `--methods`: Methods to evaluate - `fast`, `hybrid`, `hybrid_blocking` (default: `fast`, `hybrid`)
- `--test`: Test mode - uses mock models and limits to 10 frames (no downloads)
- `--max_frames`: Maximum number of frames to process (useful for testing)

## Evaluation Methods

- **fast**: Fast path only (YOLOv8 + ByteTrack)
- **hybrid**: Hybrid approach with fast path + slow path (non-blocking)
- **hybrid_blocking**: Hybrid approach with blocking slow path (for comparison)

## Output

Results are saved as JSON files in the output directory:
- Single sequence: `{sequence}_results.json`
- All sequences: `aggregate_results.json`

Metrics computed:
- MOTA, IDF1, MOTP
- Precision, Recall
- Track fragmentation
- Mostly tracked/lost percentages
- Average track completeness
- Deadline hit rate
- Processing times
- Cache statistics (for hybrid methods)

## Requirements

- MOT16/17 dataset in `data/` directory
- Dependencies from `requirements.txt`
- Slow path service runs in-process (no separate server needed)

## Test Mode

Test mode is useful for quick verification without downloading large models:

```bash
python -m evaluation.evaluate --test --sequence MOT16-02
```

This will:
- Use mock VLM models (no HuggingFace downloads)
- Process only 10 frames by default
- Verify the evaluation pipeline works correctly
