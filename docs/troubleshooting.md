# Troubleshooting & Performance Tuning

## Common Issues

### Index Building Fails

**Symptom**: Out of memory during embedding or indexing

**Solutions**:
- Reduce `--batch-size` for embedding (try 16 or 8)
- Process corpus in batches with `--max-docs`
- Use CPU instead of GPU if VRAM limited
- Consider IVF index for large corpora (>100k chunks)

### Retrieval Too Slow

**Symptom**: >30s per question

**Solutions**:
- Reduce `--max-steps` (try 4 instead of 6)
- Reduce `--k` (try 5 instead of 8)
- Use IVF index instead of Flat for large indices
- Increase `similarity_threshold` to avoid redundant queries (try 0.90)

### Poor Answer Quality

**Symptom**: Low EM/F1 scores

**Diagnosis**:
1. Check retrieval: Are relevant chunks retrieved? (inspect trace JSON)
2. Check decomposition: Are sub-queries appropriate?
3. Check synthesis: Does answer use retrieved information?

**Solutions**:
- Increase `--k` to retrieve more chunks
- Lower `similarity_threshold` to allow more query variations
- Increase `--max-steps` for more thorough retrieval
- Try different embedding model (e.g., E5 vs DPR)
- Adjust chunking: smaller chunks for precision, larger for context

### LLM Errors

**Symptom**: JSON parsing failures, rate limit errors

**Solutions**:
- JSON parsing: Prompts have fallback logic; check logs for patterns
- Rate limits: Reduce concurrency or add delays in controller
- Timeout errors: Increase max_tokens or reduce context length

## Performance Tuning

### Latency vs Quality Tradeoff

| Configuration | Latency | Quality | Use Case |
|---------------|---------|---------|----------|
| k=3, steps=3  | Fast    | Lower   | Quick answers, simple queries |
| k=8, steps=6  | Medium  | Good    | Balanced (default) |
| k=12, steps=10| Slow    | Best    | Complex multi-hop, research |

### Index Type Selection

- **Flat**: Exact search, small-to-medium corpora (<100k chunks)
- **IVF**: Approximate search, large corpora (>100k chunks)
  - Set `nlist = sqrt(n_chunks)` as starting point
  - Higher nlist = more accurate but slower

### Chunk Size Tuning

- **Small chunks (256 tokens)**: Higher precision, more chunks needed
- **Medium chunks (512 tokens)**: Balanced (default)
- **Large chunks (1024 tokens)**: More context, but may dilute relevance

### Embedding Model Selection

| Model | Dim | Speed | Quality |
|-------|-----|-------|---------|
| msmarco-distilbert-base-v4 | 768 | Fast | Good |
| all-MiniLM-L6-v2 | 384 | Faster | Decent |
| E5-base-v2 | 768 | Fast | Better |
| BGE-base-en-v1.5 | 768 | Fast | Better |

## Debugging

### Enable Verbose Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Inspect Retrieval Trace

```bash
corag-search ... --output trace.json
cat trace.json | jq '.steps[] | {query: .query, num_chunks: .num_chunks}'
```

### Check Retrieved Chunks

```python
import json
with open("trace.json") as f:
    trace = json.load(f)
for step in trace["steps"]:
    print(f"Query: {step['query']}")
    for chunk in step["chunks"][:3]:
        print(f"  - {chunk['text'][:100]}...")
```

### Measure Component Latency

Add timing logs in pipeline:
```python
import time
start = time.time()
# ... operation ...
logger.info(f"Operation took {time.time() - start:.2f}s")
```

## Resource Requirements

### Minimum
- CPU: 2 cores
- RAM: 4 GB
- Storage: 1 GB (small index)

### Recommended
- CPU: 8 cores or GPU
- RAM: 16 GB
- Storage: 10 GB+ (for large corpora)

### Large Scale (>1M chunks)
- CPU: 16+ cores or GPU
- RAM: 64 GB+
- Storage: 100 GB+
- Consider distributed FAISS

## Contact

For issues not covered here, please open an issue on GitHub with:
- Full error message and stack trace
- Command or code that triggered the issue
- Environment (Python version, OS, hardware)
- Minimal reproducible example if possible
