"""
vLLM Inference Server

FastAPI server wrapping vLLM's AsyncLLMEngine for high-throughput batched inference.
vLLM automatically batches concurrent requests for optimal GPU utilization.

Usage:
    python vllm_server.py server --model llama2-7b --port 8000
"""

import sys
import os
import gc
import asyncio
import signal
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Load environment variables from .env file (project root)
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env'))

# Use 'spawn' instead of 'fork' for multiprocessing (required for Thunder Compute / environments that don't support fork with GPU)
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

try:
    from vllm import LLM, SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.utils import random_uuid
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import uvicorn
except ImportError as e:
    print(f"ERROR: Missing dependencies: {e}")
    print("Please install: pip install vllm fastapi uvicorn pydantic")
    sys.exit(1)

from typing import List, Optional

# HuggingFace Access Token (for gated models like Llama 2)
# Set in .env file, see example.env
HF_ACCESS_TOKEN = os.environ.get('HF_TOKEN', '')

# Model configuration - set from command line
MODEL_NAME = None
GPU_MEMORY_UTIL = 0.85
MAX_MODEL_LEN = None  # None = use model default
async_engine = None

# Available models
AVAILABLE_MODELS = {
    "llama2-7b": "meta-llama/Llama-2-7b-chat-hf",
    "deepseek-7b": "deepseek-ai/deepseek-coder-7b-instruct",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.1",
    "phi2": "microsoft/phi-2",
    "nvidia/llama3-1": "nvidia/Llama-3.1-Nemotron-Nano-8B-v1",
}


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 256
    temperature: Optional[float] = 0.7


class ChatRequest(BaseModel):
    messages: List[dict]
    max_tokens: Optional[int] = 256
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global async_engine, MODEL_NAME

    if MODEL_NAME is None:
        MODEL_NAME = AVAILABLE_MODELS["phi2"]

    if HF_ACCESS_TOKEN:
        os.environ['HF_TOKEN'] = HF_ACCESS_TOKEN
        os.environ['HUGGING_FACE_HUB_TOKEN'] = HF_ACCESS_TOKEN

    print(f"Loading model: {MODEL_NAME}")
    print("vLLM will automatically batch concurrent requests for optimal performance")

    engine_kwargs = dict(
        model=MODEL_NAME,
        trust_remote_code=True,
        max_num_seqs=256,
        gpu_memory_utilization=GPU_MEMORY_UTIL,
        enforce_eager=True,
    )
    if MAX_MODEL_LEN is not None:
        engine_kwargs["max_model_len"] = MAX_MODEL_LEN
    engine_args = AsyncEngineArgs(**engine_kwargs)
    async_engine = AsyncLLMEngine.from_engine_args(engine_args)
    print("Model loaded! Server ready for batched requests.")

    yield

    # ── Graceful shutdown: release engine and GPU memory ──
    print("Shutting down vLLM engine...")
    if async_engine is not None:
        try:
            # Shutdown the engine (stops EngineCore workers)
            if hasattr(async_engine, 'shutdown'):
                async_engine.shutdown()
            elif hasattr(async_engine, 'abort_all'):
                await async_engine.abort_all()
        except Exception as e:
            print(f"  Engine shutdown warning: {e}")
        async_engine = None

    # Force GPU memory release
    try:
        import torch
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
            gc.collect()
            free, total = torch.cuda.mem_get_info()
            print(f"  GPU memory after cleanup: {free/1024**3:.1f} GiB free / {total/1024**3:.1f} GiB total")
    except Exception as e:
        print(f"  GPU cleanup warning: {e}")

    print("Shutdown complete.")


app = FastAPI(title="vLLM Inference Server", lifespan=lifespan)


@app.get("/")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "model": MODEL_NAME}


@app.post("/generate")
async def generate(request: GenerateRequest):
    """Generate text from a prompt. Multiple concurrent calls are batched by vLLM."""
    sampling_params = SamplingParams(
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        min_tokens=5,
        ignore_eos=False,
    )

    request_id = random_uuid()
    results_generator = async_engine.generate(request.prompt, sampling_params, request_id)

    final_output = None
    async for request_output in results_generator:
        final_output = request_output

    if final_output and final_output.outputs:
        response_text = final_output.outputs[0].text
    else:
        response_text = ""

    return {"prompt": request.prompt, "response": response_text}


@app.post("/v1/chat/completions")
async def chat_completion(request: ChatRequest):
    """OpenAI-compatible chat completions endpoint."""
    user_msg = [m["content"] for m in request.messages if m["role"] == "user"]
    if not user_msg:
        raise HTTPException(400, "No user message")

    prompt = user_msg[-1]

    sampling_params = SamplingParams(
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
    )

    request_id = random_uuid()
    results_generator = async_engine.generate(prompt, sampling_params, request_id)

    final_output = None
    async for request_output in results_generator:
        final_output = request_output

    response_text = final_output.outputs[0].text if final_output and final_output.outputs else ""

    return {
        "id": request_id,
        "object": "chat.completion",
        "model": MODEL_NAME,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": response_text},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": len(prompt.split()),
            "completion_tokens": len(response_text.split()),
            "total_tokens": len(prompt.split()) + len(response_text.split()),
        },
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="vLLM Inference Server")
    parser.add_argument("command", choices=["server"], help="Run server")
    parser.add_argument("--model", type=str, default="phi2",
                        help=f"Model to load: {', '.join(AVAILABLE_MODELS.keys())}")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gpu-memory", type=float, default=0.85,
                        help="GPU memory utilization (0.0-1.0). Default: 0.85")
    parser.add_argument("--max-model-len", type=int, default=None,
                        help="Maximum model context length. Default: use model's native limit")

    args = parser.parse_args()

    if args.model in AVAILABLE_MODELS:
        MODEL_NAME = AVAILABLE_MODELS[args.model]
    else:
        MODEL_NAME = args.model

    GPU_MEMORY_UTIL = args.gpu_memory
    MAX_MODEL_LEN = args.max_model_len

    print(f"Starting vLLM server: {args.model} ({MODEL_NAME})")
    print(f"Endpoint: {args.host}:{args.port}")
    print(f"GPU Memory: {GPU_MEMORY_UTIL*100:.0f}%")
    if MAX_MODEL_LEN:
        print(f"Max model len: {MAX_MODEL_LEN}")

    # Register signal handlers so SIGTERM triggers graceful uvicorn shutdown
    # (uvicorn handles this internally, but we ensure the lifespan teardown runs)
    def _handle_signal(signum, frame):
        print(f"\nReceived signal {signum}, shutting down gracefully...")
        sys.exit(0)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    uvicorn.run(app, host=args.host, port=args.port)
