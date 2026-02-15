# vLLM RAG Deployment - LangChain vs LlamaIndex Benchmark

## Overview

This project benchmarks **Retrieval-Augmented Generation (RAG)** using two popular frameworks — **LangChain** and **LlamaIndex** — with **vLLM** as the high-throughput LLM inference backend.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Benchmark Runner                           │
│  For each (model, framework, concurrency):                      │
│    1. Batch retrieve contexts    (parallel threads)             │
│    2. Build RAG prompts          (template + context + question)│
│    3. Fire all prompts at vLLM   (async HTTP)                  │
│    4. vLLM batches on GPU        (automatic continuous batching)│
└────────────┬───────────────────────────────┬────────────────────┘
             │                               │
     ┌───────▼───────┐             ┌─────────▼────────┐
     │  LangChain    │             │  LlamaIndex      │
     │  RAG Pipeline │             │  RAG Pipeline     │
     │               │             │                   │
     │ DirectoryLoader│            │ SimpleDirectoryReader│
     │ RecursiveText  │            │ SentenceSplitter  │
     │   Splitter     │            │ ChromaVectorStore │
     │ Chroma         │            │ VectorStoreIndex  │
     │ Retriever      │            │ Retriever         │
     └───────┬───────┘             └─────────┬────────┘
             │ retrieved context              │ retrieved context
             └───────────────┬────────────────┘
                             │ RAG prompts
                    ┌────────▼────────┐
                    │   vLLM Server   │
                    │  (FastAPI +     │
                    │  AsyncLLMEngine)│
                    │                 │
                    │ Continuous      │
                    │ Batching on GPU │
                    └─────────────────┘
```

## Files

| File | Description |
|---|---|
| `vllm_server.py` | FastAPI server wrapping vLLM AsyncLLMEngine |
| `vllm_model_manager.py` | Auto-starts/stops vLLM server on Linux; sync + async generation |
| `rag_langchain.py` | LangChain RAG pipeline (load, chunk, embed, retrieve, batch query) |
| `rag_llamaindex.py` | LlamaIndex RAG pipeline (load, parse, embed, retrieve, batch query) |
| `vllm_benchmark.py` | Full benchmark using real RAG pipelines + system monitoring |
| `python_tutor.py` | Interactive RAG chatbot supporting both frameworks |
| `requirements.txt` | Python dependencies |

## How It Works

### RAG Pipeline (per framework)

1. **Document Loading**: Python 3.13 documentation (`.txt` files) is loaded
2. **Chunking**: Documents are split into ~200-token chunks
3. **Embedding**: Chunks are embedded using `BAAI/bge-small-en`
4. **Vector Store**: Embeddings are stored in ChromaDB
5. **Retrieval**: Top-5 similar chunks are retrieved per question
6. **Prompt Building**: Retrieved context is injected into a RAG prompt template
7. **Generation**: The augmented prompt is sent to vLLM for inference

### Batched RAG (benchmark mode)

For `N` concurrent requests:
- **Step 1**: Retrieve contexts for all N questions in parallel (ThreadPoolExecutor)
- **Step 2**: Build N RAG-augmented prompts
- **Step 3**: Send all N prompts to vLLM simultaneously (async HTTP with aiohttp)
- **Step 4**: vLLM's continuous batching engine processes them on the GPU

### Models Tested

| Model | Parameters | HuggingFace ID |
|---|---|---|
| LLaMA 2 7B | 7B | `meta-llama/Llama-2-7b-chat-hf` |
| Mistral 7B | 7B | `mistralai/Mistral-7B-Instruct-v0.1` |
| Nvidia Nemotron | 8B | `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` |
| Phi-2 | 2.7B | `microsoft/phi-2` |

## Usage

### Prerequisites

```bash
pip install -r requirements.txt
```

Python documentation files should be at `../python-3.13-docs-text/` relative to this folder.

### Interactive Tutor

```bash
# LangChain RAG + vLLM
python python_tutor.py --framework langchain --model llama2-7b

# LlamaIndex RAG + vLLM
python python_tutor.py --framework llamaindex --model mistral-7b
```

### Run Benchmark

```bash
# Full benchmark (all models, all concurrency levels)
python vllm_benchmark.py

# Quick debug run
python vllm_benchmark.py --debug

# Specific model
python vllm_benchmark.py --model llama2-7b
```

### Output

- `vllm_rag_benchmark_YYYYMMDD_HHMMSS.json` — raw results
- `vllm_rag_benchmark_charts_YYYYMMDD_HHMMSS.png` — performance charts

## Key Metrics Collected

- **Throughput**: Requests/second, Tokens/second
- **Latency**: Average, Median, P95, P99, Min, Max
- **GPU**: Utilization %, Memory (MB), Power (W)
- **CPU**: Utilization %
- **Memory**: System RAM usage (MB)
