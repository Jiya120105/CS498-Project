#!/bin/bash
# Helper script to run evaluations with proper setup

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}CS498 Project - Evaluation Runner${NC}"
echo "=========================================="

# Parse arguments
SEQUENCE="${1:-}"
DATA_ROOT="${2:-data}"
OUTPUT_DIR="${3:-results}"
TEST_MODE="${4:-}"

# Enable shared semantic cache for proper coordination between worker and hybrid processor
export USE_LOCAL_SEMANTIC_CACHE=1

# For test mode, use mock models
if [ "$TEST_MODE" = "--test" ] || [ "$TEST_MODE" = "test" ]; then
    export SLOWPATH_MODEL=mock
    echo -e "${YELLOW}Test mode: Using mock models${NC}"
    TEST_FLAG="--test"
else
    TEST_FLAG=""
fi

# Note: The slow path service uses ServiceClient which works in-process
# No need to start a separate server for evaluation
echo -e "${GREEN}Using in-process slow path service${NC}"

if [ -z "$SEQUENCE" ]; then
    echo -e "${GREEN}Running evaluation on all sequences...${NC}"
    python -m evaluation.evaluate --data_root "$DATA_ROOT" --output_dir "$OUTPUT_DIR" $TEST_FLAG
else
    echo -e "${GREEN}Running evaluation on sequence: $SEQUENCE${NC}"
    python -m evaluation.evaluate --sequence "$SEQUENCE" --data_root "$DATA_ROOT" --output_dir "$OUTPUT_DIR" $TEST_FLAG
fi

echo -e "${GREEN}Evaluation complete!${NC}"
echo "Results saved to: $OUTPUT_DIR/"

