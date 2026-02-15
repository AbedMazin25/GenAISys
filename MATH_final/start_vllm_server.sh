#!/bin/bash
# Launch vLLM as an OpenAI-compatible API server on Linux.
#
# Examples:
#   ./start_vllm_server.sh meta-math/MetaMath-Mistral-7B
#   ./start_vllm_server.sh deepseek-ai/DeepSeek-R1-Distill-Llama-8B 8000 8192

set -e

MODEL_ID="${1:?Usage: $0 <model_id> [port] [max_model_len]}"
PORT="${2:-8000}"
MAX_MODEL_LEN="${3:-8192}"

# Use spawn for multiprocessing (required for Thunder Compute / environments that don't support fork with GPU)
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

echo "Starting vLLM server..."
echo "  Model:         $MODEL_ID"
echo "  Port:          $PORT"
echo "  Max model len: $MAX_MODEL_LEN"

# Kill any existing vLLM server on this port
if command -v fuser &>/dev/null && fuser "$PORT/tcp" &>/dev/null; then
    echo "  Killing existing server on port $PORT..."
    fuser -k "$PORT/tcp" 2>/dev/null || true
    sleep 2
fi

# Launch vLLM OpenAI-compatible server
# - gpu_memory_utilization=0.85 leaves room for KV cache
# - dtype=auto lets vLLM pick the best dtype for the model
# - trust_remote_code for models that need it (Phi, etc.)
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_ID" \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization 0.85 \
    --dtype auto \
    --trust-remote-code \
    --disable-log-requests \
    --enforce-eager
