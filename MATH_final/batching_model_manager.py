"""
vLLM model manager — manages a single vLLM OpenAI-compatible API server per model.

Handles server lifecycle (start/stop/swap), GPU memory cleanup between models,
token counting via HuggingFace tokenizers, and request retry logic.
"""

import requests
import subprocess
import signal
import time
import threading
import sys
import os
import gc
from typing import Dict, Any
from collections import deque
from transformers import AutoTokenizer
import numpy as np

# vLLM server configuration
VLLM_PORT = 8000
VLLM_BASE_URL = f"http://localhost:{VLLM_PORT}"
VLLM_STARTUP_TIMEOUT = 600  # seconds to wait for server to start (8B models need more time)
VLLM_HEALTH_CHECK_INTERVAL = 5  # seconds between health checks

# Available models with context window sizes
MODELS = {
    "phi2": {
        "model_id": "microsoft/phi-2",
        "description": "Microsoft Phi-2 (2.7B) - Strong math + reasoning, fast",
        "context_window": 2048
    },
    "deepseek-r1-distill": {
        "model_id": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "description": "DeepSeek-R1-Distill-Llama-8B - 8B parameters, reasoning-focused",
        "context_window": 8192
    },
    "metamath-mistral": {
        "model_id": "meta-math/MetaMath-Mistral-7B",
        "description": "MetaMath-Mistral-7B - Fine-tuned on MetaMathQA, GSM8K 77.7% accuracy",
        "context_window": 8192
    },
    "mistral-7b-quantized": {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.1",
        "description": "Mistral-7B - High-quality reasoning",
        "context_window": 8192
    }
}

# ========== VLLM SERVER MANAGEMENT ==========

_vllm_process = None
_current_model_id = None
_tokenizer_cache = {}


def _force_free_gpu_memory():
    """Aggressively free ALL GPU memory before starting a new vLLM server.
    
    Kills any process holding GPU memory (orphaned vLLM, stale CUDA contexts),
    then polls until at least 80% of VRAM is free (up to 60s).
    """
    import torch

    # 1. Kill any orphaned vLLM / CUDA processes still holding memory
    _kill_orphaned_vllm_processes()

    # 2. Kill anything on the vLLM port
    try:
        subprocess.run(["fuser", "-k", f"{VLLM_PORT}/tcp"],
                        timeout=10, capture_output=True)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # 3. Kill any other python processes holding GPU (except ourselves)
    my_pid = os.getpid()
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            try:
                pid = int(line)
                if pid != my_pid:
                    os.kill(pid, signal.SIGKILL)
                    print(f"  Killed GPU-holding process {pid}")
            except (ValueError, ProcessLookupError, PermissionError):
                pass
    except Exception:
        pass

    # 4. Release our own CUDA context and poll until driver reclaims memory
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
        gc.collect()

        total = torch.cuda.mem_get_info()[1]
        threshold = total * 0.80  # need 80% free

        for attempt in range(30):  # up to 60s
            free, total = torch.cuda.mem_get_info()
            if free >= threshold:
                print(f"  GPU memory ready: {free/1024**3:.1f} GiB free / {total/1024**3:.1f} GiB total")
                return
            time.sleep(2)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        free, total = torch.cuda.mem_get_info()
        print(f"  ⚠️ GPU memory not fully released after 60s: {free/1024**3:.1f}/{total/1024**3:.1f} GiB free")


def _start_vllm_server(model_id: str, max_model_len: int = 8192):
    """Start vLLM server natively on Linux for the given model."""
    global _vllm_process, _current_model_id

    # Stop existing server if running
    _stop_vllm_server()

    # Aggressively free GPU memory before starting
    _force_free_gpu_memory()

    print(f"Starting vLLM server for {model_id}...")

    # Use 'spawn' for multiprocessing to avoid fork issues with CUDA
    env = os.environ.copy()
    env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    python_path = sys.executable

    cmd = [
        python_path, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_id,
        "--port", str(VLLM_PORT),
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", "0.85",
        "--dtype", "auto",
        "--trust-remote-code",
        "--disable-log-requests",
        "--enforce-eager",  # skip torch.compile — avoids Triton libcuda.so linker issues
    ]

    # Capture stderr to a log file so startup failures are debuggable
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"vllm_server_{model_id.replace('/', '_')}.log")
    log_file = open(log_path, "w")

    _vllm_process = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=log_file,
        env=env,
        start_new_session=True,  # new process group for clean shutdown
    )

    # Wait for server to be ready
    start_time = time.time()
    while time.time() - start_time < VLLM_STARTUP_TIMEOUT:
        try:
            resp = requests.get(f"{VLLM_BASE_URL}/health", timeout=2)
            if resp.status_code == 200:
                _current_model_id = model_id
                elapsed = time.time() - start_time
                print(f"vLLM server ready for {model_id} (started in {elapsed:.1f}s)")
                log_file.close()
                return True
        except requests.ConnectionError:
            pass

        # Check if process died
        if _vllm_process.poll() is not None:
            log_file.close()
            # Read the log to show why it failed
            try:
                with open(log_path, "r") as f:
                    log_tail = f.read()[-2000:]  # last 2000 chars
                print(f"vLLM server process exited with code {_vllm_process.returncode}")
                print(f"vLLM log ({log_path}):\n{log_tail}")
            except Exception:
                print(f"vLLM server process exited with code {_vllm_process.returncode}")
            finally:
                _vllm_process = None
            return False

        time.sleep(VLLM_HEALTH_CHECK_INTERVAL)

    log_file.close()
    print(f"vLLM server startup timed out after {VLLM_STARTUP_TIMEOUT}s")
    _stop_vllm_server()
    return False


def _stop_vllm_server():
    """Stop the running vLLM server, kill orphans, and free GPU memory."""
    global _vllm_process, _current_model_id

    print(f"Stopping vLLM server (model: {_current_model_id})...")

    # 1. Kill tracked process group
    if _vllm_process is not None:
        pid = _vllm_process.pid
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            pass

        try:
            _vllm_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                _vllm_process.kill()
            _vllm_process.wait(timeout=5)
        finally:
            _vllm_process = None

    # 2. Kill any orphaned vLLM / EngineCore processes
    _kill_orphaned_vllm_processes()

    # 3. Free GPU memory
    _cleanup_gpu_memory()

    _current_model_id = None
    time.sleep(2)  # Wait for port release
    print("vLLM server stopped and GPU memory released.")


def _kill_orphaned_vllm_processes():
    """Find and kill any lingering vLLM/EngineCore processes."""
    try:
        result = subprocess.run(
            ['pgrep', '-f', 'VLLM::EngineCore|vllm.entrypoints'],
            capture_output=True, text=True,
        )
        if result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid_str in pids:
                try:
                    pid = int(pid_str.strip())
                    os.kill(pid, signal.SIGKILL)
                    print(f"  Killed orphaned process {pid}")
                except (ProcessLookupError, PermissionError, ValueError):
                    pass
            time.sleep(1)
    except FileNotFoundError:
        pass  # pgrep not available

    # Also kill anything on the port
    try:
        subprocess.run(
            ["fuser", "-k", f"{VLLM_PORT}/tcp"],
            timeout=10, capture_output=True,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass


def _cleanup_gpu_memory():
    """Force-release GPU memory and wait until it's actually free.

    After killing vLLM child processes, the GPU driver may take a few
    seconds to reclaim their memory.  This method polls until at least
    80% of total VRAM is free (or 30s timeout), so the next model
    launch won't hit an OOM error.
    """
    try:
        import torch
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
            gc.collect()

            # Wait for the driver to actually release memory
            total = torch.cuda.mem_get_info()[1]
            threshold = total * 0.80  # need 80% free
            for attempt in range(15):  # up to 30s
                free, total = torch.cuda.mem_get_info()
                if free >= threshold:
                    print(f"  GPU memory: {free/1024**3:.1f} GiB free / {total/1024**3:.1f} GiB total")
                    return
                # Driver hasn't reclaimed yet — wait and retry
                time.sleep(2)
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

            # Timed out — report what we have
            free, total = torch.cuda.mem_get_info()
            print(f"  GPU memory (slow release): {free/1024**3:.1f} GiB free / {total/1024**3:.1f} GiB total")
    except Exception as e:
        print(f"  GPU cleanup warning: {e}")


def _ensure_server_running(model_id: str, context_window: int = 8192):
    """Ensure vLLM server is running with the correct model."""
    global _current_model_id

    # Check if ANY server is already running on the port
    try:
        resp = requests.get(f"{VLLM_BASE_URL}/v1/models", timeout=3)
        if resp.status_code == 200:
            models_data = resp.json()
            running_model = models_data["data"][0]["id"] if models_data.get("data") else None

            if running_model == model_id:
                # Correct model already running
                _current_model_id = model_id
                return True
            else:
                # Wrong model running - need to kill it and start the right one
                print(f"vLLM server has {running_model}, need {model_id} - restarting...")
                _stop_vllm_server()
    except requests.ConnectionError:
        pass  # No server running, will start fresh

    # Need to start
    return _start_vllm_server(model_id, context_window)


# ========== MODEL MANAGER ==========

class VLLMModelManager:
    """Model manager that communicates with vLLM via OpenAI-compatible API."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model_info = MODELS[model_name]
        self.model_id = self.model_info["model_id"]
        self.context_window = self.model_info.get("context_window", 8192)

        # Stats tracking
        self.stats = {
            'total_requests': 0,
            'total_tokens': 0,
            'total_time': 0,
            'latencies': deque(maxlen=100)
        }
        self._stats_lock = threading.Lock()

        # Load tokenizer for token counting (lightweight, runs on CPU)
        self.tokenizer = None
        if model_name not in _tokenizer_cache:
            try:
                _tokenizer_cache[model_name] = AutoTokenizer.from_pretrained(
                    self.model_id, trust_remote_code=True
                )
            except Exception as e:
                print(f"Warning: Could not load tokenizer for {model_name}: {e}")
        self.tokenizer = _tokenizer_cache.get(model_name)

        # Ensure vLLM server is running for this model
        if not _ensure_server_running(self.model_id, self.context_window):
            raise RuntimeError(f"Failed to start vLLM server for {model_name}")

        # Warm up: send a short request to trigger CUDA kernel compilation
        try:
            print(f"Warming up vLLM for {model_name}...")
            resp = requests.post(
                f"{VLLM_BASE_URL}/v1/completions",
                json={"model": self.model_id, "prompt": "Q: What is 2+2?\nA:", "max_tokens": 10, "temperature": 0.1},
                timeout=30
            )
            if resp.status_code == 200:
                print(f"Warmup complete for {model_name}")
            else:
                print(f"Warmup returned {resp.status_code}, continuing anyway")
        except Exception as e:
            print(f"Warmup failed: {e}, continuing anyway")

    def generate(self, text: str, max_tokens: int = 256,
                 temperature: float = 0.7, top_p: float = 1.0) -> str:
        """Generate response via vLLM API."""
        response, _ = self.generate_with_token_info(text, max_tokens, temperature, top_p)
        return response

    def generate_with_token_info(self, text: str, max_tokens: int = 256,
                                  temperature: float = 0.7, top_p: float = 1.0) -> tuple:
        """Generate response with token count via vLLM API."""
        start_time = time.time()

        try:
            # Window safety: cap max_tokens so prompt + generation fits
            input_token_count = self.count_tokens(text) if self.tokenizer else len(text.split())
            safe_max_tokens = min(max_tokens, self.context_window - input_token_count - 10)
            if safe_max_tokens < 50:
                safe_max_tokens = 50

            # Call vLLM OpenAI-compatible API with retry
            resp = None
            for attempt in range(3):
                try:
                    resp = requests.post(
                        f"{VLLM_BASE_URL}/v1/completions",
                        json={
                            "model": self.model_id,
                            "prompt": text,
                            "max_tokens": safe_max_tokens,
                            "temperature": temperature,
                            "top_p": top_p,
                            "repetition_penalty": 1.1,
                        },
                        timeout=120
                    )
                    break  # Success
                except (requests.ConnectionError, requests.Timeout) as e:
                    if attempt < 2:
                        print(f"  vLLM retry {attempt+1}/3: {e}")
                        time.sleep(3)
                    else:
                        raise

            if resp is None:
                return "Error: All retries failed", 0

            if resp.status_code != 200:
                error_msg = f"vLLM API error {resp.status_code}: {resp.text[:200]}"
                print(f"Error: {error_msg}")
                return error_msg, 0

            result = resp.json()

            # Extract response text and token count
            response_text = result["choices"][0]["text"]
            completion_tokens = result["usage"]["completion_tokens"]

            # Update stats
            end_time = time.time()
            latency = end_time - start_time
            self._update_stats(response_text, latency, completion_tokens)

            return response_text, completion_tokens

        except requests.Timeout:
            print(f"vLLM request timed out for {self.model_name}")
            return "Error: Request timed out", 0
        except requests.ConnectionError:
            print(f"vLLM server not reachable for {self.model_name}")
            return "Error: Server not reachable", 0
        except Exception as e:
            print(f"Error generating with {self.model_name}: {e}")
            return f"Error: {str(e)}", 0

    def count_tokens(self, text: str) -> int:
        """Count tokens using the model's tokenizer."""
        if self.tokenizer:
            import logging
            logger = logging.getLogger("transformers.tokenization_utils_base")
            prev_level = logger.level
            logger.setLevel(logging.ERROR)
            try:
                return len(self.tokenizer.encode(text, add_special_tokens=False))
            finally:
                logger.setLevel(prev_level)
        else:
            return int(len(text.split()) * 1.33)

    def get_tokenizer(self):
        """Return the tokenizer."""
        return self.tokenizer

    def _update_stats(self, response: str, latency: float, token_count: int):
        """Update performance stats."""
        with self._stats_lock:
            self.stats['total_requests'] += 1
            self.stats['total_tokens'] += token_count
            self.stats['total_time'] += latency
            self.stats['latencies'].append(latency)

    def get_stats(self):
        """Get performance statistics."""
        with self._stats_lock:
            if not self.stats['latencies']:
                return self.stats.copy()

            latencies = list(self.stats['latencies'])
            total_time = self.stats['total_time'] or 1

            return {
                'total_requests': self.stats['total_requests'],
                'total_tokens': self.stats['total_tokens'],
                'avg_latency': np.mean(latencies),
                'p95_latency': np.percentile(latencies, 95),
                'p99_latency': np.percentile(latencies, 99),
                'tokens_per_second': self.stats['total_tokens'] / total_time,
                'requests_per_second': self.stats['total_requests'] / total_time,
            }

    def reset_stats(self):
        """Reset performance stats."""
        with self._stats_lock:
            self.stats = {
                'total_requests': 0,
                'total_tokens': 0,
                'total_time': 0,
                'latencies': deque(maxlen=100)
            }


# ========== MODULE-LEVEL API (same interface as before) ==========

_model_cache = {}
_cache_lock = threading.Lock()


def get_model_manager(model_name: str) -> VLLMModelManager:
    """Get or create a model manager for the given model."""
    with _cache_lock:
        if model_name not in _model_cache:
            print(f"Creating vLLM model manager for {model_name}")
            _model_cache[model_name] = VLLMModelManager(model_name)
        return _model_cache[model_name]


def get_model_info(model_name: str) -> dict:
    """Get model configuration."""
    return MODELS[model_name]


def get_available_models() -> list:
    """Get list of available model names."""
    return list(MODELS.keys())


def clear_cache():
    """Stop vLLM server, free GPU memory, and clear model manager cache."""
    global _model_cache
    with _cache_lock:
        _model_cache.clear()
    _stop_vllm_server()
    print("Model cache cleared, vLLM server stopped, GPU memory released.")


def count_tokens_for_model(model_name: str, text: str) -> int:
    """Count tokens for a specific model without creating a full manager."""
    if model_name in _tokenizer_cache:
        return len(_tokenizer_cache[model_name].encode(text, add_special_tokens=False))
    try:
        model_id = MODELS[model_name]["model_id"]
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        _tokenizer_cache[model_name] = tok
        return len(tok.encode(text, add_special_tokens=False))
    except Exception:
        return int(len(text.split()) * 1.33)
