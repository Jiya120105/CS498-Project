#!/bin/bash
# Helper script to run evaluations with proper setup

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}CS498 Project - Evaluation Runner${NC}"
echo "=========================================="

# Check if cache stub is running
if ! curl -s http://127.0.0.1:8010/cache/stats > /dev/null 2>&1; then
    echo -e "${YELLOW}Warning: Cache stub server not detected on port 8010${NC}"
    echo -e "${YELLOW}Starting cache stub in background...${NC}"
    cd slow_path
    python cache_stub.py &
    CACHE_PID=$!
    cd ..
    sleep 2
    echo -e "${GREEN}Cache stub started (PID: $CACHE_PID)${NC}"
    echo -e "${YELLOW}Note: You may need to kill it manually: kill $CACHE_PID${NC}"
else
    echo -e "${GREEN}Cache stub server detected and running${NC}"
fi

# Parse arguments
SEQUENCE="${1:-}"
DATA_ROOT="${2:-data}"
OUTPUT_DIR="${3:-results}"

# Enable shared semantic cache for proper coordination between worker and hybrid processor
export USE_LOCAL_SEMANTIC_CACHE=1

if [ -z "$SEQUENCE" ]; then
    echo -e "${GREEN}Running evaluation on all sequences...${NC}"
    python -m evaluation.evaluate --data_root "$DATA_ROOT" --output_dir "$OUTPUT_DIR"
else
    echo -e "${GREEN}Running evaluation on sequence: $SEQUENCE${NC}"
    python -m evaluation.evaluate --sequence "$SEQUENCE" --data_root "$DATA_ROOT" --output_dir "$OUTPUT_DIR"
fi

echo -e "${GREEN}Evaluation complete!${NC}"
echo "Results saved to: $OUTPUT_DIR/"

