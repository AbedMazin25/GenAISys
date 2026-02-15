"""
vLLM RAG Benchmark - Compare LangChain vs LlamaIndex with vLLM Batching

This benchmark ACTUALLY uses each framework's RAG pipeline:
  - LangChain: document loading, chunking, Chroma retrieval, prompt building
  - LlamaIndex: document loading, node parsing, Chroma retrieval, prompt building

For each (model, framework, concurrency) combination:
  1. N questions are sent to the RAG retriever (parallel threads)
  2. Retrieved contexts are merged into RAG prompts
  3. All N prompts are fired at vLLM simultaneously (async HTTP)
  4. vLLM batches them on the GPU for optimal throughput
  5. System metrics (CPU, GPU, memory, power) are sampled throughout

Usage:
    python vllm_benchmark.py                    # full benchmark
    python vllm_benchmark.py --debug            # quick test (1 model, fewer levels)
    python vllm_benchmark.py --model llama2-7b  # single model
"""

import time
import json
import gc
import psutil
import torch
import os
import subprocess
import threading
import asyncio
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Dict, Any
import sys

sys.path.insert(0, os.path.dirname(__file__))
import vllm_model_manager as model_manager

# Try to load HuggingFace dataset for questions
try:
    from datasets import load_dataset
    DATASET_AVAILABLE = True
except ImportError:
    DATASET_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════
#  System Monitoring Helpers
# ═══════════════════════════════════════════════════════════════

def get_cpu_utilization():
    return psutil.cpu_percent(interval=0.1)

def get_memory_utilization():
    return psutil.virtual_memory().used / (1024 * 1024)

def _nvidia_smi_query(query_str: str):
    """Helper: run nvidia-smi and return first line as float."""
    if not torch.cuda.is_available():
        return None
    try:
        result = subprocess.run(
            ['nvidia-smi', f'--query-gpu={query_str}', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True,
        )
        return float(result.stdout.strip().split('\n')[0].split(',')[0].strip())
    except Exception:
        return None

def get_gpu_utilization():
    return _nvidia_smi_query('utilization.gpu')

def get_gpu_memory_used():
    return _nvidia_smi_query('memory.used')

def get_gpu_power():
    return _nvidia_smi_query('power.draw')


# ═══════════════════════════════════════════════════════════════
#  Test Questions
# ═══════════════════════════════════════════════════════════════

def load_test_questions(num_questions: int = 96) -> List[str]:
    """Load test questions from dataset or use fallback set."""
    if DATASET_AVAILABLE:
        try:
            dataset = load_dataset("glaiveai/glaive-code-assistant", split="train")
            questions = [item['question'] for i, item in enumerate(dataset) if i < num_questions]
            print(f"Loaded {len(questions)} questions from dataset")
            return questions
        except Exception:
            pass

    fallback = [
        "What is a list in Python?",
        "How do I create a dictionary?",
        "Explain Python generators",
        "What are decorators?",
        "How does list comprehension work?",
        "What is the difference between append and extend?",
        "Explain the with statement",
        "What are lambda functions?",
        "How do you handle exceptions?",
        "What is the difference between == and is?",
        "How do I read a file in Python?",
        "What is a tuple in Python?",
        "Explain Python inheritance",
        "What are context managers?",
        "How do async functions work in Python?",
    ]
    return (fallback * (num_questions // len(fallback) + 1))[:num_questions]


# ═══════════════════════════════════════════════════════════════
#  Benchmark Result Container
# ═══════════════════════════════════════════════════════════════

class BenchmarkResult:
    """Thread-safe container for benchmark measurements."""

    def __init__(self):
        self.latencies: List[float] = []
        self.responses: List[str] = []
        self.token_counts: List[int] = []
        self.cpu_samples: List[float] = []
        self.gpu_samples: List[float] = []
        self.memory_samples: List[float] = []
        self.gpu_memory_samples: List[float] = []
        self.power_samples: List[float] = []
        self.start_time = None
        self.end_time = None
        self._lock = threading.Lock()

    def add_result(self, latency: float, response: str, tokens: int = None):
        with self._lock:
            self.latencies.append(latency)
            self.responses.append(response)
            self.token_counts.append(tokens if tokens else len(response.split()))

    def add_system_sample(self, cpu, gpu, memory, gpu_memory, power):
        with self._lock:
            if cpu is not None:
                self.cpu_samples.append(cpu)
            if gpu is not None:
                self.gpu_samples.append(gpu)
            if memory is not None:
                self.memory_samples.append(memory)
            if gpu_memory is not None:
                self.gpu_memory_samples.append(gpu_memory)
            if power is not None:
                self.power_samples.append(power)

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            if not self.latencies:
                return {}
            total_time = (self.end_time - self.start_time) if self.start_time and self.end_time else 1
            total_tokens = sum(self.token_counts)
            return {
                'total_requests': len(self.latencies),
                'avg_latency': float(np.mean(self.latencies)),
                'median_latency': float(np.median(self.latencies)),
                'p95_latency': float(np.percentile(self.latencies, 95)),
                'p99_latency': float(np.percentile(self.latencies, 99)),
                'min_latency': float(np.min(self.latencies)),
                'max_latency': float(np.max(self.latencies)),
                'requests_per_second': len(self.latencies) / total_time,
                'tokens_per_second': total_tokens / total_time,
                'total_tokens': total_tokens,
                'avg_cpu': float(np.mean(self.cpu_samples)) if self.cpu_samples else None,
                'peak_cpu': float(np.max(self.cpu_samples)) if self.cpu_samples else None,
                'avg_memory_mb': float(np.mean(self.memory_samples)) if self.memory_samples else None,
                'peak_memory_mb': float(np.max(self.memory_samples)) if self.memory_samples else None,
                'avg_gpu_util': float(np.mean(self.gpu_samples)) if self.gpu_samples else None,
                'peak_gpu_util': float(np.max(self.gpu_samples)) if self.gpu_samples else None,
                'avg_gpu_memory_mb': float(np.mean(self.gpu_memory_samples)) if self.gpu_memory_samples else None,
                'peak_gpu_memory_mb': float(np.max(self.gpu_memory_samples)) if self.gpu_memory_samples else None,
                'avg_power': float(np.mean(self.power_samples)) if self.power_samples else None,
                'peak_power': float(np.max(self.power_samples)) if self.power_samples else None,
            }


# ═══════════════════════════════════════════════════════════════
#  Main Benchmark Class
# ═══════════════════════════════════════════════════════════════

class VLLMBenchmark:
    """
    Benchmark that ACTUALLY uses LangChain / LlamaIndex RAG pipelines
    with vLLM batched inference.
    """

    def __init__(self, docs_path: str, debug_mode: bool = False,
                 gpu_memory_utilization: float = 0.85):
        self.docs_path = docs_path
        self.debug_mode = debug_mode
        self.gpu_memory_utilization = gpu_memory_utilization
        self.port = 8000
        self.monitoring_active = False

        if debug_mode:
            self.models = [list(model_manager.MODELS.keys())[0]]
            self.frameworks = ['langchain', 'llamaindex']
            self.concurrency_levels = [1, 8, 48, 96]
            self.questions = load_test_questions(96)
            print("DEBUG MODE: 1 model x 2 frameworks x 4 concurrency levels")
        else:
            self.models = list(model_manager.MODELS.keys())
            self.frameworks = ['langchain', 'llamaindex']
            self.concurrency_levels = [1, 2, 4, 8, 16, 24, 48, 72, 96]
            self.questions = load_test_questions(96)

    # ── GPU cleanup ─────────────────────────────────────────────

    @staticmethod
    def _cleanup_gpu():
        """Force GPU memory cleanup between model runs, waiting for driver release."""
        print("  Cleaning up GPU memory...")
        try:
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                torch.cuda.synchronize()
                gc.collect()

                # Wait for the driver to actually release child process memory
                total = torch.cuda.mem_get_info()[1]
                threshold = total * 0.80
                for _ in range(15):  # up to 30s
                    free, total = torch.cuda.mem_get_info()
                    if free >= threshold:
                        break
                    time.sleep(2)
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()

                free, total = torch.cuda.mem_get_info()
                print(f"  GPU memory: {free/1024**3:.1f} GiB free / {total/1024**3:.1f} GiB total")
        except Exception as e:
            print(f"  GPU cleanup warning: {e}")

    # ── RAG initialisation ─────────────────────────────────────

    def _init_rag(self, framework: str):
        """Initialise the appropriate RAG pipeline."""
        if framework == 'langchain':
            from rag_langchain import LangChainRAG
            return LangChainRAG(self.docs_path)
        elif framework == 'llamaindex':
            from rag_llamaindex import LlamaIndexRAG
            return LlamaIndexRAG(self.docs_path)
        else:
            raise ValueError(f"Unknown framework: {framework}")

    # ── System monitoring ──────────────────────────────────────

    def _monitor_system(self, result: BenchmarkResult):
        """Sample system metrics every second while monitoring_active."""
        while self.monitoring_active:
            result.add_system_sample(
                get_cpu_utilization(),
                get_gpu_utilization(),
                get_memory_utilization(),
                get_gpu_memory_used(),
                get_gpu_power(),
            )
            time.sleep(1.0)

    # ── Warmup ─────────────────────────────────────────────────

    async def _warmup(self, rag, llm_manager, n: int = 3):
        """Run a few RAG queries to warm up the pipeline."""
        print(f"   [Warmup] Running {n} warmup RAG queries...")
        warmup_qs = ["What is a list?", "How do I use print?", "What is a function?"][:n]
        await rag.batch_query_async(warmup_qs, llm_manager, max_tokens=32)
        print(f"   [Warmup] Done")
        await asyncio.sleep(0.5)

    # ── Single concurrency test ────────────────────────────────

    def test_concurrency(self, rag, llm_manager, model: str, framework: str,
                         concurrency: int, skip_warmup: bool = False) -> Dict:
        """
        Test a specific concurrency level using the REAL RAG pipeline:
          1. Pick N questions
          2. RAG retrieves context for all N (parallel threads)
          3. Build N augmented prompts
          4. Send all N to vLLM via llm_manager (async HTTP)
          5. Collect latencies and responses
        """
        print(f"\n   [{framework}] {model} @ concurrency={concurrency}")

        if not skip_warmup:
            asyncio.run(self._warmup(rag, llm_manager))

        questions = self.questions[:concurrency]

        # Result tracker
        result = BenchmarkResult()
        result.start_time = time.time()

        # Start system monitoring
        self.monitoring_active = True
        monitor = threading.Thread(target=self._monitor_system, args=(result,), daemon=True)
        monitor.start()

        # Run the batched RAG pipeline — uses llm_manager for HTTP requests
        try:
            raw = asyncio.run(
                rag.batch_query_async(questions, llm_manager, max_tokens=256, temperature=0.7)
            )
            for latency, response, error in raw:
                if not error:
                    result.add_result(latency, response)
        except Exception as e:
            print(f"      Error: {e}")

        # Stop monitoring
        self.monitoring_active = False
        monitor.join(timeout=2)
        result.end_time = time.time()

        stats = result.get_stats()

        if stats:
            print(f"      Completed {stats['total_requests']} requests")
            print(f"      Avg latency: {stats['avg_latency']:.2f}s  |  "
                  f"RPS: {stats['requests_per_second']:.2f}  |  "
                  f"Tokens/s: {stats['tokens_per_second']:.1f}")
            if stats.get('avg_gpu_util'):
                print(f"      GPU: {stats['avg_gpu_util']:.1f}% avg  |  "
                      f"Power: {stats.get('avg_power', 0):.0f}W avg")
        return stats

    # ── Full benchmark suite ───────────────────────────────────

    def run_benchmark_suite(self) -> List[Dict]:
        """Run the complete benchmark across all models, frameworks, and concurrency levels."""
        print("=" * 80)
        print("vLLM RAG Benchmark Suite")
        print("=" * 80)
        n_tests = len(self.models) * len(self.frameworks) * len(self.concurrency_levels)
        print(f"Models:      {self.models}")
        print(f"Frameworks:  {self.frameworks}")
        print(f"Concurrency: {self.concurrency_levels}")
        print(f"Total tests: {n_tests}")
        print("=" * 80)

        all_results = []
        test_num = 0

        for model_name in self.models:
            print(f"\n{'='*80}")
            print(f"  Model: {model_name}")
            print(f"{'='*80}")

            # Start vLLM server for this model
            try:
                with model_manager.VLLMModelManager(
                    model_name, port=self.port,
                    gpu_memory_utilization=self.gpu_memory_utilization,
                ) as llm:
                    time.sleep(2)

                    for framework in self.frameworks:
                        print(f"\n  --- Framework: {framework} ---")

                        # Initialise the REAL RAG pipeline for this framework
                        rag = self._init_rag(framework)

                        # Warmup once per (model, framework)
                        asyncio.run(self._warmup(rag, llm, n=3))
                        time.sleep(1)

                        for conc in self.concurrency_levels:
                            test_num += 1
                            print(f"\n  [{test_num}/{n_tests}]", end='')

                            try:
                                stats = self.test_concurrency(
                                    rag, llm, model_name, framework, conc, skip_warmup=True,
                                )
                                all_results.append({
                                    'model': model_name,
                                    'framework': framework,
                                    'concurrency': conc,
                                    'stats': stats,
                                    'timestamp': datetime.now().isoformat(),
                                })
                                time.sleep(1)
                            except Exception as e:
                                print(f"      Test failed: {e}")

                        # Release RAG resources before switching framework
                        rag.cleanup()

                    print(f"\n  All tests for {model_name} complete.")

                print(f"  Server stopped for {model_name}.")

                # Force GPU memory cleanup between models
                self._cleanup_gpu()
                time.sleep(2)

            except Exception as e:
                print(f"  Failed to test {model_name}: {e}")
                # Cleanup GPU even on failure
                self._cleanup_gpu()

        # Save & chart
        self._save_results(all_results)
        self._create_charts(all_results)

        print(f"\n{'='*80}")
        print(f"Benchmark complete!  {len(all_results)} / {n_tests} tests succeeded.")
        print(f"{'='*80}")
        return all_results

    # ── Persistence ────────────────────────────────────────────

    def _save_results(self, results):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"vllm_rag_benchmark_{ts}.json"
        with open(fname, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {fname}")

    # ── Charting ───────────────────────────────────────────────

    def _create_charts(self, results):
        if not results:
            return
        try:
            frameworks = sorted(set(r['framework'] for r in results))
            models = sorted(set(r['model'] for r in results))
            conc_levels = sorted(set(r['concurrency'] for r in results))

            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            fig.suptitle('vLLM RAG Benchmark: LangChain vs LlamaIndex', fontsize=16)

            metrics = [
                ('requests_per_second', 'Requests/s',          'Throughput vs Concurrency'),
                ('avg_latency',         'Avg Latency (s)',      'Latency vs Concurrency'),
                ('tokens_per_second',   'Tokens/s',             'Token Throughput vs Concurrency'),
                ('avg_gpu_memory_mb',   'GPU Memory (MB)',      'GPU Memory vs Concurrency'),
                ('avg_gpu_util',        'GPU Util (%)',         'GPU Utilization vs Concurrency'),
                ('avg_power',           'Power (W)',            'Power vs Concurrency'),
            ]

            for idx, (metric_key, ylabel, title) in enumerate(metrics):
                ax = axes[idx // 3, idx % 3]
                for fw in frameworks:
                    fw_results = [r for r in results if r['framework'] == fw]
                    xs, ys = [], []
                    for c in conc_levels:
                        vals = [r['stats'][metric_key] for r in fw_results
                                if r['concurrency'] == c and r['stats'].get(metric_key) is not None]
                        if vals:
                            xs.append(c)
                            ys.append(np.mean(vals))
                    if xs:
                        ax.plot(xs, ys, marker='o', label=fw, linewidth=2)
                ax.set_xlabel('Concurrency')
                ax.set_ylabel(ylabel)
                ax.set_title(title)
                ax.legend()
                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"vllm_rag_benchmark_charts_{ts}.png"
            plt.savefig(fname, dpi=150, bbox_inches='tight')
            print(f"Charts saved to {fname}")
        except Exception as e:
            print(f"Could not create charts: {e}")


# ═══════════════════════════════════════════════════════════════
#  CLI Entry Point
# ═══════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="vLLM RAG Benchmark - LangChain vs LlamaIndex with real retrieval",
    )
    parser.add_argument('--debug', action='store_true', help='Quick test (1 model, 4 concurrency levels)')
    parser.add_argument('--model', type=str, default=None, help='Test a specific model only')
    parser.add_argument('--docs-path', type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'python-3.13-docs-text'),
                        help='Path to Python documentation text files')
    parser.add_argument('--gpu-memory', type=float, default=0.85, help='GPU memory utilization (0-1)')

    args = parser.parse_args()

    print("vLLM RAG Benchmark - LangChain vs LlamaIndex")
    print("Each framework ACTUALLY performs retrieval before generation.")
    print()

    benchmark = VLLMBenchmark(
        docs_path=args.docs_path,
        debug_mode=args.debug,
        gpu_memory_utilization=args.gpu_memory,
    )

    if args.model:
        if args.model in model_manager.MODELS:
            benchmark.models = [args.model]
        else:
            print(f"Unknown model: {args.model}.  Available: {list(model_manager.MODELS.keys())}")
            return

    results = benchmark.run_benchmark_suite()

    # Summary
    print("\n--- Summary ---")
    for fw in ['langchain', 'llamaindex']:
        fw_r = [r for r in results if r['framework'] == fw]
        if fw_r:
            avg_tps = np.mean([r['stats']['tokens_per_second'] for r in fw_r if r['stats']])
            avg_rps = np.mean([r['stats']['requests_per_second'] for r in fw_r if r['stats']])
            print(f"  {fw:12s}:  {avg_rps:.2f} RPS,  {avg_tps:.1f} tokens/s  (avg across all tests)")


if __name__ == "__main__":
    main()
