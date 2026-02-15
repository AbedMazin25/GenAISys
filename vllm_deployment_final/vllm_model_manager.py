"""
vLLM Model Manager - Auto-starts and manages vLLM server lifecycle.

Provides:
  - Automatic server start/stop via context manager
  - Synchronous and asynchronous generation
  - Batch generation (sends all prompts concurrently; vLLM batches on GPU)

Usage:
    with VLLMModelManager("llama2-7b", port=8000) as llm:
        response = llm.generate("What is Python?")
        batch = llm.generate_batch(["Q1", "Q2", "Q3"])
"""

import subprocess
import signal
import sys
import time
import requests
import atexit
import os
import asyncio
import aiohttp
import gc
from typing import Dict, Any, List
from collections import deque
from dotenv import load_dotenv

# Load environment variables from .env file (project root)
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env'))

# HuggingFace token (set in .env file, see example.env)
HF_ACCESS_TOKEN = os.environ.get('HF_TOKEN', '')

# Available models
# max_model_len caps vLLM's context to fit in GPU KV cache.
# Models with huge native defaults (deepseek 65k, llama3.1 128k) MUST be capped.
MODELS = {
    "llama2-7b": {
        "model_id": "meta-llama/Llama-2-7b-chat-hf",
        "description": "Meta's Llama 2 (7B)",
        "max_model_len": 4096,
    },
    "deepseek-7b": {
        "model_id": "deepseek-ai/deepseek-coder-7b-instruct",
        "description": "DeepSeek Coder (7B)",
        "max_model_len": 8192,    # native 65536 — far too large for single GPU
    },
    "mistral-7b": {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.1",
        "description": "Mistral AI (7B)",
        "max_model_len": 8192,
    },
    "phi2": {
        "model_id": "microsoft/phi-2",
        "description": "Microsoft Phi-2",
        "max_model_len": 2048,    # native limit is 2048
    },
    "nvidia/llama3-1": {
        "model_id": "nvidia/Llama-3.1-Nemotron-Nano-8B-v1",
        "description": "Nvidia Nemotron Nano 8B",
        "max_model_len": 8192,    # native 131072 — far too large for single GPU
    },
}


class VLLMModelManager:
    """
    Manages a vLLM server process on Linux and exposes synchronous/async
    generation methods.  Use as a context manager for automatic cleanup.
    """

    def __init__(self, model_name: str, port: int = 8000,
                 auto_start: bool = True, gpu_memory_utilization: float = 0.85):
        self.model_name = model_name
        self.model_info = MODELS.get(model_name, {"model_id": model_name, "description": model_name})
        self.port = port
        self.gpu_memory_utilization = gpu_memory_utilization
        self.server_url = f"http://localhost:{port}"
        self.server_process = None

        self.stats = {
            'total_requests': 0,
            'total_tokens': 0,
            'total_time': 0,
            'latencies': deque(maxlen=1000),
        }

        if auto_start:
            self._start_server()
            atexit.register(self._stop_server)

    # ── Server lifecycle ───────────────────────────────────────

    def _start_server(self):
        """Start vLLM server natively on Linux."""
        print(f"\n{'='*60}")
        print(f"Starting vLLM Server: {self.model_name}  (port {self.port})")
        print(f"{'='*60}\n")

        # Wait for GPU memory to be stable before launching
        self._wait_for_gpu_stable()

        print(f"GPU Memory utilization: {self.gpu_memory_utilization*100:.0f}%")

        vllm_dir = os.path.dirname(os.path.abspath(__file__))
        python_path = sys.executable

        # Log server output to a temp file so crashes are diagnosable
        log_path = os.path.join(vllm_dir, f'.vllm_server_{self.model_name.replace("/", "_")}.log')
        self._server_log_file = open(log_path, 'w')

        max_model_len = self.model_info.get("max_model_len", 8192)
        cmd = [
            python_path, os.path.join(vllm_dir, 'vllm_server.py'), 'server',
            '--model', self.model_name,
            '--port', str(self.port),
            '--gpu-memory', str(self.gpu_memory_utilization),
            '--max-model-len', str(max_model_len),
        ]

        self.server_process = subprocess.Popen(
            cmd,
            stdout=self._server_log_file,
            stderr=self._server_log_file,
            stdin=subprocess.DEVNULL,
            start_new_session=True,  # creates a new process group for clean shutdown
        )

        print(f"Server process started (PID: {self.server_process.pid})")

        if self._wait_for_server():
            print(f"Server ready at {self.server_url}\n")
        else:
            # Dump last lines of the server log so we can see WHY it failed
            self._dump_server_log(log_path)
            self._stop_server()
            raise RuntimeError("Failed to start vLLM server")

    @staticmethod
    def _wait_for_gpu_stable(timeout: int = 120):
        """Wait for GPU memory to be stable and mostly free before launching vLLM.

        Thunder Compute's CUDA driver can take 10-60s to reclaim memory from
        killed processes.  If vLLM starts while memory is fluctuating, its
        memory profiler hits an assertion error.  This method blocks until:
          1. Free memory is >= 90% of total, AND
          2. Two consecutive readings are within 0.5 GiB (stable)
        """
        try:
            import torch
            if not torch.cuda.is_available():
                return

            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

            prev_free = 0
            start = time.time()
            while time.time() - start < timeout:
                free, total = torch.cuda.mem_get_info()
                free_frac = free / total
                delta = abs(free - prev_free)
                stable = (delta < 0.5 * 1024**3) and (prev_free > 0)

                if free_frac >= 0.90 and stable:
                    print(f"  GPU ready: {free/1024**3:.1f} GiB free / {total/1024**3:.1f} GiB total")
                    return

                elapsed = int(time.time() - start)
                if elapsed > 0:
                    print(f"  Waiting for GPU memory to release... ({free/1024**3:.1f}/{total/1024**3:.1f} GiB free, {elapsed}s)", flush=True)

                prev_free = free
                time.sleep(3)
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

            # Timed out
            free, total = torch.cuda.mem_get_info()
            print(f"  GPU wait timed out after {timeout}s: {free/1024**3:.1f}/{total/1024**3:.1f} GiB free — launching anyway")
        except Exception as e:
            print(f"  GPU check warning: {e}")

    @staticmethod
    def _dump_server_log(log_path, n=30):
        """Print the last N lines of the server log for debugging."""
        try:
            if os.path.exists(log_path):
                with open(log_path, 'r') as f:
                    lines = f.readlines()
                tail = lines[-n:] if len(lines) > n else lines
                print(f"\n--- Server log tail ({log_path}) ---")
                for line in tail:
                    print(f"  {line.rstrip()}")
                print("--- end ---\n")
        except Exception as e:
            print(f"  Could not read server log: {e}")

    def _wait_for_server(self, timeout: int = 300) -> bool:
        """Poll health endpoint until server is ready."""
        start = time.time()
        last_print = 0
        while time.time() - start < timeout:
            try:
                r = requests.get(f"{self.server_url}/", timeout=5)
                if r.status_code == 200:
                    print(flush=True)  # newline after progress dots
                    return True
            except requests.exceptions.RequestException:
                pass
            if self.server_process.poll() is not None:
                print("\nServer process exited unexpectedly")
                return False
            elapsed = int(time.time() - start)
            # Print a dot every 10s to show progress (pipe-friendly, no \r)
            if elapsed - last_print >= 10:
                print(f"  Waiting for server... ({elapsed}s)", flush=True)
                last_print = elapsed
            time.sleep(2)
        return False

    def _stop_server(self):
        """Terminate the vLLM server process and all its children, then free GPU memory."""
        if self.server_process:
            pid = self.server_process.pid
            print(f"Stopping vLLM server (PID {pid})...")

            # 1. Kill the entire process group to catch EngineCore children
            try:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError, OSError):
                pass

            # 2. Wait for main process, escalate to SIGKILL if needed
            try:
                self.server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    self.server_process.kill()
                self.server_process.wait(timeout=5)
            finally:
                self.server_process = None

            # 3. Close server log file handle
            if hasattr(self, '_server_log_file') and self._server_log_file and not self._server_log_file.closed:
                self._server_log_file.close()

            # 4. Hunt and kill any orphaned vLLM/EngineCore processes
            self._kill_orphaned_vllm_processes()

            # 5. Force GPU memory cleanup
            self._cleanup_gpu_memory()

            print("Server stopped and GPU memory released.")

    @staticmethod
    def _kill_orphaned_vllm_processes():
        """Find and kill any lingering VLLM::EngineCore or related processes."""
        try:
            result = subprocess.run(
                ['pgrep', '-f', 'VLLM::EngineCore|vllm_server\\.py'],
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

    @staticmethod
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

    # ── Synchronous generation ─────────────────────────────────

    def generate(self, text: str, max_tokens: int = 256, temperature: float = 0.7) -> str:
        """Generate a single response synchronously."""
        start_time = time.time()
        try:
            response = requests.post(
                f"{self.server_url}/generate",
                json={"prompt": text, "max_tokens": max_tokens, "temperature": temperature},
                timeout=60,
            )
            elapsed = time.time() - start_time
            if response.status_code == 200:
                result = response.json()
                generated = result.get('response', '')
                self._update_stats(generated, elapsed)
                return generated
            return f"Error: Server returned {response.status_code}"
        except Exception as e:
            return f"Error: {e}"

    # ── Asynchronous generation ────────────────────────────────

    async def generate_async(self, session: aiohttp.ClientSession, text: str,
                             max_tokens: int = 256, temperature: float = 0.7):
        """Send one async request to vLLM. Returns (latency, response_text, error)."""
        start_time = time.time()
        try:
            async with session.post(
                f"{self.server_url}/generate",
                json={"prompt": text, "max_tokens": max_tokens, "temperature": temperature},
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                latency = time.time() - start_time
                if resp.status == 200:
                    data = await resp.json()
                    return latency, data.get('response', ''), None
                return latency, '', f"HTTP {resp.status}"
        except Exception as e:
            return time.time() - start_time, '', str(e)

    async def generate_batch_async(self, prompts: List[str],
                                   max_tokens: int = 256, temperature: float = 0.7):
        """
        Send all prompts concurrently.  vLLM batches them on the GPU.
        Returns list of (latency, response_text, error) tuples.
        """
        async with aiohttp.ClientSession() as session:
            tasks = [self.generate_async(session, p, max_tokens, temperature) for p in prompts]
            return await asyncio.gather(*tasks)

    def generate_batch(self, prompts: List[str],
                       max_tokens: int = 256, temperature: float = 0.7) -> List[str]:
        """Synchronous wrapper for batch generation."""
        results = asyncio.run(self.generate_batch_async(prompts, max_tokens, temperature))
        responses = []
        for latency, text, error in results:
            if error:
                responses.append(f"Error: {error}")
            else:
                self._update_stats(text, latency)
                responses.append(text)
        return responses

    # ── Stats ──────────────────────────────────────────────────

    def count_tokens(self, text: str) -> int:
        """Approximate token count."""
        return int(len(text.split()) * 1.3) if text else 0

    def _update_stats(self, text: str, latency: float):
        self.stats['total_requests'] += 1
        self.stats['total_tokens'] += self.count_tokens(text)
        self.stats['total_time'] += latency
        self.stats['latencies'].append(latency)

    def get_stats(self) -> Dict[str, Any]:
        if self.stats['total_requests'] == 0:
            return self.stats
        return {
            **self.stats,
            'avg_latency': self.stats['total_time'] / self.stats['total_requests'],
            'tokens_per_second': (self.stats['total_tokens'] / self.stats['total_time']
                                  if self.stats['total_time'] > 0 else 0),
        }

    # ── Context manager ────────────────────────────────────────

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._stop_server()


if __name__ == "__main__":
    print("Testing VLLMModelManager...")
    with VLLMModelManager("phi2", port=8000) as mgr:
        print(mgr.generate("Hello, what is Python?", max_tokens=50))
        print(f"Stats: {mgr.get_stats()}")
    print("Done.")
