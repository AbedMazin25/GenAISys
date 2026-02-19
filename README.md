# LLM Framework Benchmarking Suite

Benchmarking suite comparing LLM orchestration frameworks (LangChain and LlamaIndex) across two domains: **math reasoning** and **retrieval-augmented generation (RAG)**. Both apps use [vLLM](https://github.com/vllm-project/vllm) as the inference backend for high-throughput GPU-accelerated generation.

## Projects

### MATH_final — Math Reasoning Benchmark

Compares three framework configurations on the [GSM8K](https://huggingface.co/datasets/openai/gsm8k) math dataset:

| Framework | Description |
|-----------|-------------|
| **LlamaIndex** | Single LLM call via `PromptTemplate` + `CompletionResponse` |
| **LangChain** | Single LLM call via `PromptTemplate` + LCEL chain |
| **LangChain-Reasoning** | 4-step `SequentialChain` (Parse → Plan → Execute → Verify) |

Tests accuracy, latency, and throughput across context-window strategies (fixed vs adaptive) and lengths (512–4096 tokens). Adaptive context uses Math-BERT embeddings with MMR diversity selection.

**Available models:** Phi-2, DeepSeek-R1-Distill-Llama-8B, MetaMath-Mistral-7B, Mistral-7B-Instruct

```bash
# Full benchmark
python MATH_final/load_benchmark.py

# Custom run
python MATH_final/load_benchmark.py \
  --models metamath-mistral \
  --frameworks llamaindex langchain \
  --context-lengths 1024 2048 \
  --context-strategies adaptive \
  --num-questions 48

# Quick framework comparison on a single question
python MATH_final/compare_frameworks.py
```

**Key files:**

| File | Purpose |
|------|---------|
| `load_benchmark.py` | Main benchmark runner with CLI interface |
| `math_tutor_llamaindex.py` | LlamaIndex implementation |
| `math_tutor_langchain.py` | LangChain LCEL implementation |
| `math_tutor_langchain_reasoning.py` | LangChain 4-step reasoning chain |
| `context_manager.py` | Fixed truncation and adaptive (Math-BERT) context strategies |
| `batching_model_manager.py` | vLLM server lifecycle and GPU memory management |
| `few_shot_examples.py` | 76 GSM8K-style chain-of-thought examples |
| `compare_frameworks.py` | Quick side-by-side demo script |

---

### vllm_deployment_final — RAG Benchmark

Compares LangChain and LlamaIndex RAG pipelines using Python 3.13 documentation as the knowledge base. Measures throughput, latency, GPU utilization, and power consumption across concurrency levels.

**Pipeline:** Document loading → Chunking → Embedding (BAAI/bge-small-en) → ChromaDB vector store → Top-k retrieval → vLLM generation

**Available models:** Llama 2 7B Chat, DeepSeek Coder 7B, Mistral 7B Instruct, Phi-2, Nvidia Nemotron Nano 8B

```bash
# Full benchmark
python vllm_deployment_final/vllm_benchmark.py

# Debug run
python vllm_deployment_final/vllm_benchmark.py --debug

# Interactive Python tutor
python vllm_deployment_final/python_tutor.py --framework langchain --model mistral-7b

# Standalone vLLM server
python vllm_deployment_final/vllm_server.py server --model llama2-7b --port 8000
```

**Key files:**

| File | Purpose |
|------|---------|
| `vllm_benchmark.py` | Main benchmark runner |
| `rag_langchain.py` | LangChain RAG pipeline with ChromaDB |
| `rag_llamaindex.py` | LlamaIndex RAG pipeline with ChromaDB |
| `vllm_server.py` | FastAPI server wrapping vLLM's AsyncLLMEngine |
| `vllm_model_manager.py` | vLLM server lifecycle and GPU memory management |
| `python_tutor.py` | Interactive chatbot supporting both frameworks |
| `average_benchmark_runs.py` | Utility to average multiple benchmark JSON runs |

## Setup

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA support
- HuggingFace account (for gated models like Llama 2)

### Installation

```bash
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```
HF_TOKEN=your_huggingface_token
```

Required for gated models (e.g., Llama 2). Models are downloaded from HuggingFace on first use.

### Data

- **MATH_final**: GSM8K dataset is loaded automatically via the `datasets` library.
- **vllm_deployment_final**: Requires Python 3.13 docs at `python-3.13-docs-text/` (sibling to this repo), or pass `--docs-path` to override.

## Output

Both benchmarks produce:
- **JSON results** with per-configuration metrics (accuracy, latency, throughput, GPU stats)
- **PNG charts** visualizing comparisons across frameworks, models, and configurations
