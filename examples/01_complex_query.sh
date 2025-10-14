#!/bin/bash
# Example: Multi-hop query about historical events

set -e

QUESTION="Who was the first person to walk on the moon and when did they return to Earth?"
INDEX_DIR="data/index"
OUTPUT_DIR="examples/output"

mkdir -p "$OUTPUT_DIR"

echo "Running CoRAG on complex query..."
echo "Question: $QUESTION"
echo

# Run multi-step retrieval
corag-search \
  --question "$QUESTION" \
  --index-dir "$INDEX_DIR" \
  --max-steps 6 \
  --k 8 \
  --temperature 0.2 \
  --output "$OUTPUT_DIR/trace.json"

# Generate answer
corag-answer \
  --trace "$OUTPUT_DIR/trace.json" \
  --output "$OUTPUT_DIR/answer.md" \
  --temperature 0.3

echo
echo "Results saved to $OUTPUT_DIR/"
echo "  - trace.json: Full retrieval trace"
echo "  - answer.md: Generated answer with citations"
echo

cat "$OUTPUT_DIR/answer.md"
