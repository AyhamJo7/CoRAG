# Evaluation Methodology

## Datasets

### HotpotQA
- Multi-hop question answering dataset
- Requires reasoning across 2+ Wikipedia articles
- 113k questions with supporting facts annotations
- Types: bridge (connecting entities) and comparison

### 2WikiMultihopQA
- Explicitly multi-hop with structured reasoning paths
- Wikipedia-based with 2+ hop requirements
- Smaller but more challenging than HotpotQA

## Metrics

### Answer Quality
- **Exact Match (EM)**: Normalized exact string match
- **F1 Score**: Token-level overlap between prediction and ground truth
- Both computed after lowercasing, removing articles (a, an, the), and normalizing punctuation

### Retrieval Efficiency
- **Avg Steps**: Mean number of retrieval iterations
- **Avg Chunks**: Mean total chunks retrieved
- **Avg Unique Chunks**: Mean unique chunks (after deduplication)
- **Retrieval Recall**: % of gold supporting documents retrieved

### Efficiency
- **Avg Latency**: Mean wall-clock time per question (seconds)
- **Total Tokens**: LLM tokens used (prompt + completion)

## Running Evaluations

```bash
corag-eval \
  --dataset hotpotqa \
  --split validation \
  --index-dir data/index \
  --max-steps 6 \
  --k 8 \
  --max-examples 100 \
  --output reports/hotpotqa_results
```

Output files:
- `hotpotqa_results.json`: Aggregate metrics
- `hotpotqa_results.csv`: Per-example details

## Adding New Datasets

Implement a loader in `corag/evaluation/datasets.py`:

```python
def load_custom_dataset(self, split, max_examples):
    for item in dataset:
        yield EvalExample(
            id=item["id"],
            question=item["question"],
            answer=item["answer"],
            supporting_facts=item.get("supporting_facts", []),
        )
```

Register in `DatasetLoader.load_dataset()`.
