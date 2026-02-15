# GSM8K Math Benchmark — Framework Comparison

Benchmarks three LLM orchestration frameworks on the GSM8K math dataset, comparing accuracy, latency, and throughput across different context-window strategies.

## Frameworks Tested

| Framework | Implementation | LLM Calls/Question |
|-----------|---------------|---------------------|
| **LlamaIndex** | `PromptTemplate` + `CompletionResponse` pipeline | 1 |
| **LangChain** | `PromptTemplate` → `LLM` via LCEL (`prompt \| llm`) | 1 |
| **LangChain-Reasoning** | `SequentialChain` of 4 `LLMChain` steps (Parse → Plan → Execute → Verify) | 4 |

All three share the same backend (`vLLM` via OpenAI-compatible API) and the same context management layer.

## Architecture

```
                        ┌─────────────────────────┐
                        │   GSM8K Test Questions   │
                        └────────────┬────────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                ▼                 ▼
             LlamaIndex        LangChain      LangChain-Reasoning
           (1 LLM call)     (1 LLM call)    (4-step SequentialChain)
                    │                │                 │
                    └────────┬───────┘                 │
                             ▼                         ▼
                   ┌──────────────────┐    ┌────────────────────┐
                   │ Context Manager  │    │  Context Manager    │
                   │ (fixed/adaptive) │    │  (half budget for   │
                   └────────┬─────────┘    │   execute step)     │
                            │              └─────────┬──────────┘
                            ▼                        ▼
                   ┌──────────────────────────────────────┐
                   │  vLLM Server (OpenAI-compatible API) │
                   │  Models: MetaMath-Mistral-7B,        │
                   │  DeepSeek-R1-8B, Phi-2, Mistral-7B   │
                   └──────────────────────────────────────┘
```

### Context Strategies

- **Fixed** — Truncates whole QA pairs from the head, preserving the question at the tail.
- **Adaptive** — Math-BERT embeddings + MMR diversity selection + operation-type matching. Selects the most relevant few-shot examples for each query, then reranks by recency bias.

Both strategies are tested at 512, 1024, 2048, and 4096 token budgets.

## Setup

```bash
pip install -r requirements.txt
```

The vLLM server starts automatically on first use. Models are downloaded from HuggingFace on demand.

## Usage

```bash
# Full benchmark — all models × frameworks × context configs × 12 questions
python load_benchmark.py

# Custom run
python load_benchmark.py \
  --models metamath-mistral \
  --frameworks llamaindex langchain langchain_reasoning \
  --context-lengths 1024 2048 \
  --context-strategies fixed adaptive \
  --num-questions 48

# Quick single-model test
python load_benchmark.py \
  --debug --models metamath-mistral --frameworks llamaindex \
  --context-lengths 1024 --context-strategies adaptive --num-questions 5
```

Ctrl+C saves partial results and cleans up the vLLM server.

### CLI Flags

| Flag | Values | Default |
|------|--------|---------|
| `--models` | `phi2` `deepseek-r1-distill` `metamath-mistral` `mistral-7b-quantized` | all |
| `--frameworks` | `llamaindex` `langchain` `langchain_reasoning` | all |
| `--context-lengths` | `512` `1024` `2048` `4096` | all |
| `--context-strategies` | `fixed` `adaptive` | both |
| `--num-questions` | 1–1319 (GSM8K test set size) | 12 |
| `--debug` | flag | off |

### Quick Demo

```bash
python compare_frameworks.py
```

## Output

Each run produces:

- **JSON** — `math_benchmark_framework_comparison_YYYYMMDD_HHMMSS.json` with per-test accuracy, latency, throughput, token counts, and system metrics.
- **PNG** — 9-chart dashboard: framework accuracy/latency/throughput, context strategy comparison, context length scaling, model comparison, strategy×length heatmap, framework×model heatmap, best overall config.

## File Reference

| File | Purpose |
|------|---------|
| `load_benchmark.py` | Benchmark runner — CLI, GSM8K loading, metrics, charts |
| `math_tutor.py` | Direct LLM baseline — prompt building, answer extraction, evaluation |
| `math_tutor_llamaindex.py` | LlamaIndex framework — `PromptTemplate` + `CompletionResponse` |
| `math_tutor_langchain.py` | LangChain framework — `PromptTemplate` + LCEL chain |
| `math_tutor_langchain_reasoning.py` | LangChain 4-step reasoning — `SequentialChain` of `LLMChain`s |
| `context_manager.py` | Fixed truncation + adaptive Math-BERT/MMR selection |
| `batching_model_manager.py` | vLLM server lifecycle, OpenAI API client, tokenizer |
| `few_shot_examples.py` | 76 GSM8K-style chain-of-thought examples |
| `compare_frameworks.py` | Quick side-by-side framework demo |
| `start_vllm_server.sh` | Manual vLLM server launch script |
| `requirements.txt` | Python dependencies |

## Key Design Decisions

1. **Answer extraction uses the LAST "the answer is" occurrence** — correctly handles the reasoning chain where intermediate steps also contain the trigger phrase.
2. **Numerical comparison** — `"39.0"` equals `"39"` in accuracy scoring.
3. **Reasoning chain gets half the context budget** for few-shot examples because its execute prompt also carries parsed facts and plan from prior steps.
4. **Adaptive strategy caches Math-BERT embeddings** on disk (`.npz`) so only the query embedding is computed per question.
5. **Single vLLM server at a time** — models are swapped between test groups; GPU memory is fully released between models.
