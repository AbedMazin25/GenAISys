import time
import json
import gc
import psutil
import torch
import os
import subprocess
import threading
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Dict, Any, Optional
import warnings
warnings.filterwarnings("ignore")

# Import math tutor and related modules
import math_tutor
import batching_model_manager as model_manager

# Import framework implementations
try:
    import math_tutor_llamaindex as llamaindex_tutor
    LLAMAINDEX_AVAILABLE = True
    print("✅ LlamaIndex framework available")
except ImportError as e:
    LLAMAINDEX_AVAILABLE = False
    print(f"⚠️ LlamaIndex not available: {e}")

try:
    import math_tutor_langchain as langchain_tutor
    LANGCHAIN_AVAILABLE = True
    print("✅ LangChain framework available")
except ImportError as e:
    LANGCHAIN_AVAILABLE = False
    print(f"⚠️ LangChain not available: {e}")

try:
    import math_tutor_langchain_reasoning as langchain_reasoning_tutor
    LANGCHAIN_REASONING_AVAILABLE = True
    print("✅ LangChain-Reasoning framework available")
except ImportError as e:
    LANGCHAIN_REASONING_AVAILABLE = False
    print(f"⚠️ LangChain-Reasoning not available: {e}")

# Import GSM8K dataset from Hugging Face
from datasets import load_dataset

# Framework configurations - THREE frameworks: LlamaIndex, LangChain, LangChain-Reasoning
FRAMEWORKS = {
    "llamaindex": {
        "name": "LlamaIndex",
        "available": LLAMAINDEX_AVAILABLE,
        "solve_function": llamaindex_tutor.solve_math_problem_llamaindex_with_token_info if LLAMAINDEX_AVAILABLE else None,
        "description": "LlamaIndex PromptTemplate + CompletionResponse + Context Manager"
    },
    "langchain": {
        "name": "LangChain",
        "available": LANGCHAIN_AVAILABLE,
        "solve_function": langchain_tutor.solve_math_problem_langchain_with_token_info if LANGCHAIN_AVAILABLE else None,
        "description": "LangChain PromptTemplate + LLM + LCEL chain + Context Manager"
    },
    "langchain_reasoning": {
        "name": "LangChain-Reasoning",
        "available": LANGCHAIN_REASONING_AVAILABLE,
        "solve_function": langchain_reasoning_tutor.solve_math_problem_langchain_reasoning_with_token_info if LANGCHAIN_REASONING_AVAILABLE else None,
        "description": "LangChain SequentialChain (4-step reasoning) + Context Manager"
    }
}

# System monitoring functions
def get_cpu_utilization():
    """Get current CPU utilization percentage"""
    return psutil.cpu_percent(interval=0.1)

def get_memory_utilization():
    """Get current memory utilization in MB"""
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

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
    """Get GPU utilization if available"""
    return _nvidia_smi_query('utilization.gpu')

def get_gpu_power():
    """Get GPU power consumption in watts"""
    return _nvidia_smi_query('power.draw')

# Load GSM8K dataset
def load_gsm8k_questions(num_questions: int = 48):
    """Load questions from the GSM8K test dataset (1,319 available).
    
    Args:
        num_questions: Number of questions to load (up to 1319).
    """
    try:
        dataset = load_dataset("gsm8k", "main", split="test")
        
        questions = []
        answers = []
        
        for i, item in enumerate(dataset):
            if i >= num_questions:
                break
            questions.append(item['question'])
            answers.append(item['answer'])
        
        print(f"✅ Loaded {len(questions)} questions from GSM8K dataset")
        return questions, answers
    
    except Exception as e:
        print(f"❌ Error loading GSM8K dataset: {e}")
        # Fallback sample questions — repeat to fill the requested count
        base_questions = [
            "A store sells pencils in packs of 12. If John buys 3 packs, how many pencils does he have?",
            "Sarah has 45 stickers. She gives 12 stickers to her friend. How many stickers does she have left?",
            "There are 24 students in a class. If they are divided into groups of 4, how many groups are there?",
            "A recipe calls for 2 cups of flour. How much flour is needed for 5 recipes?",
            "Tom has 63 marbles. He wants to divide them equally among his 7 friends. How many marbles will each friend get?",
        ]
        base_answers = [
            "#### 36",
            "#### 33", 
            "#### 6",
            "#### 10",
            "#### 9",
        ]
        # Repeat enough times to cover any requested count
        repeats = (num_questions // len(base_questions)) + 1
        questions = (base_questions * repeats)[:num_questions]
        answers = (base_answers * repeats)[:num_questions]
        print(f"⚠️ Using {len(questions)} fallback questions (GSM8K unavailable)")
        return questions, answers

class MathBenchmarkResult:
    """Container for math benchmark results with thread safety"""
    def __init__(self):
        self.latencies = []
        self.responses = []
        self.errors = []
        self.cpu_samples = []
        self.gpu_samples = []
        self.memory_samples = []
        self.power_samples = []
        self.start_time = None
        self.end_time = None
        
        # Math-specific metrics
        self.correctness_scores = []
        self.reference_answers = []
        self.questions = []
        self.models_tested = []
        self.frameworks_tested = []
        
        # Token tracking
        self.token_counts = []
        self.total_tokens = 0
        
        # Thread safety
        self._lock = threading.Lock()

    def add_result(self, latency: float, response: str, question: str = "", 
                   reference: str = "", model: str = "", framework: str = "", 
                   actual_tokens: int = None, error: str = None):
        """Add a result with thread safety"""
        with self._lock:
            self.latencies.append(latency)
            self.responses.append(response)
            self.questions.append(question)
            self.reference_answers.append(reference)
            self.models_tested.append(model)
            self.frameworks_tested.append(framework)  # Track framework
            
            # Use actual token count if provided, otherwise fall back to word count
            if actual_tokens is not None:
                token_count = actual_tokens
            else:
                token_count = len(response.split()) if not error else 0
            
            self.token_counts.append(token_count)
            self.total_tokens += token_count
            
            if error:
                self.errors.append(error)
                self.correctness_scores.append(0.0)
            else:
                # Calculate correctness using math tutor's evaluation
                if reference and response:
                    try:
                        eval_result = math_tutor.evaluate_math_answer(response, reference)
                        correctness = 1.0 if eval_result['is_correct'] else 0.0
                        self.correctness_scores.append(correctness)
                    except Exception:
                        self.correctness_scores.append(0.0)
                else:
                    self.correctness_scores.append(0.0)

    def add_system_sample(self, cpu, gpu, memory, power):
        """Add system monitoring sample with thread safety"""
        with self._lock:
            if cpu is not None:
                self.cpu_samples.append(cpu)
            if gpu is not None:
                self.gpu_samples.append(gpu)
            if memory is not None:
                self.memory_samples.append(memory)
            if power is not None:
                self.power_samples.append(power)

    def get_stats(self):
        """Calculate comprehensive statistics with thread safety"""
        with self._lock:
            if not self.latencies:
                return {}
            
            total_time = (self.end_time - self.start_time) if self.start_time and self.end_time else 1
            
            # Calculate token-based metrics
            avg_tokens_per_response = np.mean(self.token_counts) if self.token_counts else 0.0
            tokens_per_second = self.total_tokens / total_time if total_time > 0 else 0.0
            
            stats = {
                'total_requests': len(self.latencies),
                'successful_requests': len(self.latencies) - len(self.errors),
                'failed_requests': len(self.errors),
                'avg_latency': np.mean(self.latencies),
                'median_latency': np.median(self.latencies),
                'p95_latency': np.percentile(self.latencies, 95),
                'p99_latency': np.percentile(self.latencies, 99),
                'min_latency': np.min(self.latencies),
                'max_latency': np.max(self.latencies),
                'requests_per_second': len(self.latencies) / total_time,
                'avg_cpu': np.mean(self.cpu_samples) if self.cpu_samples else None,
                'peak_cpu': np.max(self.cpu_samples) if self.cpu_samples else None,
                'avg_memory_mb': np.mean(self.memory_samples) if self.memory_samples else None,
                'peak_memory_mb': np.max(self.memory_samples) if self.memory_samples else None,
                'avg_gpu': np.mean(self.gpu_samples) if self.gpu_samples else None,
                'peak_gpu': np.max(self.gpu_samples) if self.gpu_samples else None,
                'avg_power': np.mean(self.power_samples) if self.power_samples else None,
                'peak_power': np.max(self.power_samples) if self.power_samples else None,
                
                # Math-specific metrics
                'avg_correctness': np.mean(self.correctness_scores) if self.correctness_scores else 0.0,
                'total_correct': sum(self.correctness_scores) if self.correctness_scores else 0,
                'accuracy_percentage': (sum(self.correctness_scores) / len(self.correctness_scores) * 100) if self.correctness_scores else 0.0,
                'correctness_std': np.std(self.correctness_scores) if self.correctness_scores else 0.0,
                
                # Token-based metrics
                'total_tokens': self.total_tokens,
                'avg_tokens_per_response': avg_tokens_per_response,
                'tokens_per_second': tokens_per_second,
            }
            
            return stats

class MathConcurrentBenchmark:
    """Math benchmark with framework comparison support for LlamaIndex and LangChain"""
    
    def __init__(self, debug_mode=False, selected_models=None, selected_frameworks=None, 
                 selected_context_lengths=None, selected_context_strategies=None,
                 num_questions=48):
        self.monitoring_active = False
        self.available_models = model_manager.get_available_models()
        
        # Framework selection
        self.available_frameworks = {k: v for k, v in FRAMEWORKS.items() if v["available"]}
        
        # Context configuration
        self.available_context_lengths = [512, 1024, 2048, 4096]
        self.available_context_strategies = ["fixed", "adaptive"]
        
        # Debug mode configuration
        self.debug_mode = debug_mode
        if debug_mode and selected_models:
            # Filter to only selected models
            self.models_to_test = [m for m in selected_models if m in self.available_models]
            if not self.models_to_test:
                print(f"⚠️ None of the selected models {selected_models} are available!")
                print(f"Available models: {self.available_models}")
                self.models_to_test = self.available_models
        else:
            self.models_to_test = self.available_models
        
        # Framework selection
        if debug_mode and selected_frameworks:
            self.frameworks_to_test = {k: v for k, v in self.available_frameworks.items() 
                                     if k in selected_frameworks}
            if not self.frameworks_to_test:
                print(f"⚠️ None of the selected frameworks {selected_frameworks} are available!")
                print(f"Available frameworks: {list(self.available_frameworks.keys())}")
                self.frameworks_to_test = self.available_frameworks
        else:
            self.frameworks_to_test = self.available_frameworks
        
        # Context length selection
        if debug_mode and selected_context_lengths:
            self.context_lengths = [l for l in selected_context_lengths if l in self.available_context_lengths]
            if not self.context_lengths:
                print(f"⚠️ Invalid context lengths: {selected_context_lengths}")
                self.context_lengths = [1024]  # Default
        else:
            self.context_lengths = self.available_context_lengths  # Test all: 512, 1024, 2048, 4096
        
        # Context strategy selection
        if debug_mode and selected_context_strategies:
            self.context_strategies = [s for s in selected_context_strategies if s in self.available_context_strategies]
            if not self.context_strategies:
                print(f"⚠️ Invalid context strategies: {selected_context_strategies}")
                self.context_strategies = ["adaptive"]  # Default
        else:
            self.context_strategies = self.available_context_strategies  # Test both: fixed, adaptive
        
        # Load GSM8K questions
        self.num_questions = num_questions
        self.questions, self.answers = load_gsm8k_questions(num_questions)
        
        # Print debug info
        if debug_mode:
            print(f"🐛 DEBUG MODE ENABLED")
            print(f"🎯 Selected models: {self.models_to_test}")
            print(f"🎯 Selected frameworks: {list(self.frameworks_to_test.keys())}")
            print(f"📏 Context lengths: {self.context_lengths}")
            print(f"🔧 Context strategies: {self.context_strategies}")
            print(f"📚 Available models: {self.available_models}")
            print(f"📚 Available frameworks: {list(self.available_frameworks.keys())}")

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

    def monitor_system(self, result: MathBenchmarkResult):
        """Monitor system resources"""
        while self.monitoring_active:
            cpu = get_cpu_utilization()
            gpu = get_gpu_utilization()
            memory = get_memory_utilization()
            power = get_gpu_power()
            result.add_system_sample(cpu, gpu, memory, power)
            time.sleep(1.0)

    def process_worker_questions(self, questions_batch, answers_batch, model: str, framework: str, 
                                   worker_id: int, context_length: int = 1024, context_strategy: str = "adaptive"):
        """Process all questions for a single worker with framework-specific solve function"""
        results = []
        
        # Get the appropriate solve function for the framework
        framework_config = FRAMEWORKS.get(framework)
        if not framework_config or not framework_config["available"]:
            error_msg = f"Framework {framework} not available"
            for question, answer in zip(questions_batch, answers_batch):
                results.append((0.0, error_msg, 0, str(error_msg), question, answer))
            return results
        
        solve_function = framework_config["solve_function"]
        
        print(f"👷 Worker {worker_id} | {framework} | ctx={context_length} | {context_strategy}")
        
        # Pre-warm: ensure model manager + vLLM server are ready before timing questions
        try:
            model_manager.get_model_manager(model)
        except Exception as e:
            print(f"Warning: model pre-warm failed: {e}")
        
        # Process all questions - both frameworks accept the same signature
        for question, answer in zip(questions_batch, answers_batch):
            start_time = time.time()
            
            try:
                # Both frameworks accept: (question, model, use_few_shot, context_strategy, max_length)
                response, actual_tokens = solve_function(
                    question, model, True,
                    context_strategy=context_strategy,
                    max_length=context_length
                )
                
                end_time = time.time()
                latency = end_time - start_time
                results.append((latency, response, actual_tokens, None, question, answer))
                
            except Exception as e:
                end_time = time.time()
                latency = end_time - start_time
                results.append((latency, "", 0, str(e), question, answer))
        
        return results

    def concurrent_math_test(self, model: str, framework: str, num_concurrent: int,
                             context_length: int = 1024, context_strategy: str = "adaptive"):
        """Run test with framework support and context configuration.
        
        Runs questions sequentially in the main thread so that Ctrl+C
        (KeyboardInterrupt) propagates immediately.  The original
        ThreadPoolExecutor wrapper was unnecessary since concurrency is
        always 1, and it swallowed SIGINT causing Ctrl+C to hang.
        """
        framework_name = FRAMEWORKS[framework]["name"]
        print(f"\n🧮 Testing {framework_name}-{model}")
        print(f"   🎭 Framework: {framework_name}")
        print(f"   📏 Context length: {context_length} tokens")
        print(f"   🔧 Context strategy: {context_strategy}")
        
        result = MathBenchmarkResult()
        result.start_time = time.time()
        
        # Start system monitoring (daemon thread — exits automatically)
        self.monitoring_active = True
        monitor_thread = threading.Thread(target=self.monitor_system, args=(result,))
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # Run questions in the MAIN thread so KeyboardInterrupt works
        worker_results = self.process_worker_questions(
            self.questions, self.answers,
            model, framework, 0,
            context_length, context_strategy
        )
        
        for latency, response, actual_tokens, error, question, answer in worker_results:
            result.add_result(
                latency, response, question, answer, model,
                framework, actual_tokens, error
            )
        
        # Stop monitoring
        self.monitoring_active = False
        monitor_thread.join(timeout=2)
        
        result.end_time = time.time()
        
        # Display results
        stats = result.get_stats()
        print(f"✅ Completed: {stats['successful_requests']}/{stats['total_requests']} requests")
        print(f"📊 Avg latency: {stats['avg_latency']:.2f}s | RPS: {stats['requests_per_second']:.1f}")
        print(f"🎯 Correctness: {stats['total_correct']}/{stats['total_requests']} = {stats['accuracy_percentage']:.1f}%")
        print(f"📈 Tokens/sec: {stats['tokens_per_second']:.1f}")
        print(f"🔥 P95 latency: {stats['p95_latency']:.2f}s | P99: {stats['p99_latency']:.2f}s")
        
        if stats['avg_cpu']:
            print(f"💻 CPU: {stats['avg_cpu']:.1f}% avg, {stats['peak_cpu']:.1f}% peak")
        if stats['avg_memory_mb']:
            print(f"🧠 Memory: {stats['avg_memory_mb']:.0f}MB avg, {stats['peak_memory_mb']:.0f}MB peak")
        if stats['avg_gpu']:
            print(f"🎮 GPU: {stats['avg_gpu']:.1f}% avg, {stats['peak_gpu']:.1f}% peak")
        
        return result, stats

    def run_math_benchmark_suite(self):
        """Run comprehensive math benchmark suite with framework comparison and context testing"""
        print("🧮 Math Tutor Framework Comparison Benchmark Suite - GSM8K Dataset")
        print("=" * 80)
        print(f"📚 Dataset: GSM8K (top {self.num_questions} questions)")
        print(f"🤖 Available models: {', '.join(self.available_models)}")
        print(f"🎯 Models to test: {', '.join(self.models_to_test)}")
        print(f"📏 Context lengths to test: {self.context_lengths}")
        print(f"🔧 Context strategies to test: {self.context_strategies}")
        fw_list = ', '.join(f'{k} ({v["name"]})' for k, v in self.frameworks_to_test.items())
        print(f"🎭 Frameworks to test: {fw_list}")
        
        # Test only 1 concurrent request (simplified)
        concurrency_levels = [1]
        
        print(f"⚡ Concurrency: 1 parallel request")
        print(f"🔢 Total questions per test: {self.num_questions}")
        if self.debug_mode:
            print(f"🐛 DEBUG MODE: Testing only selected configurations")
        
        all_results = []
        
        # Calculate total number of tests, accounting for configs that will be
        # skipped because context_length + generation headroom exceeds model window.
        total_tests = 0
        for _ in self.frameworks_to_test:
            for m in self.models_to_test:
                m_info = model_manager.get_model_info(m)
                m_window = m_info.get("context_window", 8192)
                for cl in self.context_lengths:
                    if cl + 256 > m_window:
                        continue  # will be skipped
                    total_tests += len(self.context_strategies) * len(concurrency_levels)
        current_test = 0
        
        print(f"📊 Total test configurations: {total_tests}")
        
        # Test all combinations — KeyboardInterrupt is caught at this level so
        # partial results are always saved even if the user presses Ctrl+C.
        interrupted = False
        try:
            for framework_key, framework_config in self.frameworks_to_test.items():
                if interrupted:
                    break
                print(f"\n🎭 Starting tests for framework: {framework_config['name']}")
                
                for model in self.models_to_test:
                    if interrupted:
                        break
                    print(f"\n🚀 Starting tests for model: {model} with {framework_config['name']}")
                    
                    # Get model's context window to skip incompatible lengths
                    model_info = model_manager.get_model_info(model)
                    model_window = model_info.get("context_window", 8192)
                    
                    for context_length in self.context_lengths:
                        if interrupted:
                            break
                        # Skip if context_length + generation tokens would exceed model window
                        if context_length + 256 > model_window:
                            print(f"\n⏭️ Skipping ctx={context_length} for {model} (exceeds {model_window} window)")
                            continue
                        
                        for context_strategy in self.context_strategies:
                            if interrupted:
                                break
                            for num_concurrent in concurrency_levels:
                                current_test += 1
                                print(f"\n📋 Test {current_test}/{total_tests}: {framework_key}-{model}-ctx{context_length}-{context_strategy}")
                                
                                try:
                                    result, stats = self.concurrent_math_test(
                                        model, framework_key, num_concurrent,
                                        context_length=context_length,
                                        context_strategy=context_strategy
                                    )
                                    
                                    test_info = {
                                        'framework': framework_key,
                                        'framework_name': framework_config['name'],
                                        'model': model,
                                        'concurrent_requests': num_concurrent,
                                        'context_length': context_length,
                                        'context_strategy': context_strategy,
                                        'total_requests': self.num_questions,
                                        'stats': stats,
                                        'timestamp': datetime.now().isoformat()
                                    }
                                    
                                    all_results.append(test_info)
                                    
                                    print(f"✅ Test {current_test} complete")
                                            
                                    # Cleanup between tests (context cache only - keep vLLM server running)
                                    try:
                                        import context_manager
                                        context_manager.clear_cache()
                                    except Exception:
                                        pass
                                    
                                    time.sleep(2)
                                
                                except KeyboardInterrupt:
                                    print(f"\n\n⏹️ Ctrl+C pressed — stopping after test {current_test}")
                                    interrupted = True
                                    break
                                    
                                except Exception as e:
                                    print(f"❌ Error in test {current_test} ({framework_key}/{model}/{context_length}/{context_strategy}): {e}")
                                    import traceback
                                    traceback.print_exc()
                                    continue
                    
                    if not interrupted:
                        print(f"🎉 Completed all tests for {model} with {framework_config['name']}")
                    
                    # Stop vLLM server between models and free GPU for next model
                    try:
                        model_manager.clear_cache()
                        self._cleanup_gpu()
                        print(f"🔄 vLLM server stopped and GPU memory released, ready for next model")
                    except Exception:
                        self._cleanup_gpu()
                
                if not interrupted:
                    print(f"🎭 Completed all tests for framework: {framework_config['name']}")
        
        except KeyboardInterrupt:
            print(f"\n\n⏹️ Ctrl+C pressed — saving {len(all_results)} completed results")
            interrupted = True
        
        # ALWAYS save results (even partial) and create visualizations
        if all_results:
            self.save_results(all_results)
            self.create_charts(all_results)
        
        completed = len(all_results)
        status = "interrupted" if interrupted else "complete"
        print(f"\n🎉 Benchmark {status}! {completed}/{total_tests} tests finished.")
        if not interrupted:
            print(f"🎭 Framework comparison: {len(self.frameworks_to_test)} frameworks tested")
            print(f"📏 Context lengths tested: {self.context_lengths}")
            print(f"🔧 Context strategies tested: {self.context_strategies}")
        return all_results

    def save_results(self, results):
        """Save results to JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = "_debug" if self.debug_mode else ""
        filename = f"math_benchmark_framework_comparison{suffix}_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"💾 Results saved to {filename}")

    def create_charts(self, results):
        """Create performance charts for framework comparison"""
        try:
            plt.style.use('default')
            fig, axes = plt.subplots(3, 3, figsize=(24, 20))
            title_suffix = " (Debug Mode)" if self.debug_mode else ""
            fig.suptitle(f'Framework Comparison - GSM8K Dataset{title_suffix}', fontsize=16)
            
            # Build framework name lookup
            framework_names = {}
            for k, v in self.frameworks_to_test.items():
                framework_names[k] = v['name']
            
            # Chart 1: Framework Accuracy Comparison
            ax1 = axes[0, 0]
            fw_acc = {}
            for fw_key in self.frameworks_to_test:
                fw_results = [r for r in results if r['framework'] == fw_key]
                if fw_results:
                    fw_acc[framework_names[fw_key]] = np.mean([r['stats']['accuracy_percentage'] for r in fw_results])
            fw_colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63'][:len(fw_acc)]
            if fw_acc:
                bars = ax1.bar(fw_acc.keys(), fw_acc.values(), color=fw_colors, alpha=0.8)
                ax1.set_ylabel('Accuracy (%)')
                ax1.set_title('Framework Accuracy')
                for bar, val in zip(bars, fw_acc.values()):
                    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.1f}%', ha='center', va='bottom')
            
            # Chart 2: Framework Latency Comparison
            ax2 = axes[0, 1]
            fw_lat = {}
            for fw_key in self.frameworks_to_test:
                fw_results = [r for r in results if r['framework'] == fw_key]
                if fw_results:
                    fw_lat[framework_names[fw_key]] = np.mean([r['stats']['avg_latency'] for r in fw_results])
            if fw_lat:
                bars = ax2.bar(fw_lat.keys(), fw_lat.values(), color=fw_colors[:len(fw_lat)], alpha=0.8)
                ax2.set_ylabel('Avg Latency (s)')
                ax2.set_title('Framework Latency')
                for bar, val in zip(bars, fw_lat.values()):
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{val:.2f}s', ha='center', va='bottom')
            
            # Chart 3: Framework Throughput
            ax3 = axes[0, 2]
            fw_tps = {}
            for fw_key in self.frameworks_to_test:
                fw_results = [r for r in results if r['framework'] == fw_key]
                if fw_results:
                    fw_tps[framework_names[fw_key]] = np.mean([r['stats']['tokens_per_second'] for r in fw_results])
            if fw_tps:
                bars = ax3.bar(fw_tps.keys(), fw_tps.values(), color=fw_colors[:len(fw_tps)], alpha=0.8)
                ax3.set_ylabel('Tokens/sec')
                ax3.set_title('Framework Throughput')
                for bar, val in zip(bars, fw_tps.values()):
                    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.1f}', ha='center', va='bottom')
            
            # Chart 4: Context Strategy Comparison
            ax4 = axes[1, 0]
            strat_acc = {}
            for strategy in self.context_strategies:
                strat_results = [r for r in results if r['context_strategy'] == strategy]
                if strat_results:
                    strat_acc[strategy] = np.mean([r['stats']['accuracy_percentage'] for r in strat_results])
            if strat_acc:
                bars = ax4.bar(strat_acc.keys(), strat_acc.values(), color=['#FF9800', '#9C27B0'], alpha=0.8)
                ax4.set_ylabel('Accuracy (%)')
                ax4.set_title('Context Strategy: Fixed vs Adaptive')
                for bar, val in zip(bars, strat_acc.values()):
                    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.1f}%', ha='center', va='bottom')
            
            # Chart 5: Context Length vs Accuracy
            ax5 = axes[1, 1]
            ctx_acc = {}
            for length in sorted(self.context_lengths):
                len_results = [r for r in results if r['context_length'] == length]
                if len_results:
                    ctx_acc[f"{length}"] = np.mean([r['stats']['accuracy_percentage'] for r in len_results])
            if ctx_acc:
                colors = ['#E91E63', '#FF5722', '#FF9800', '#FFC107'][:len(ctx_acc)]
                bars = ax5.bar(ctx_acc.keys(), ctx_acc.values(), color=colors, alpha=0.8)
                ax5.set_xlabel('Context Length (tokens)')
                ax5.set_ylabel('Accuracy (%)')
                ax5.set_title('Context Length vs Accuracy')
                for bar, val in zip(bars, ctx_acc.values()):
                    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.1f}%', ha='center', va='bottom')
            
            # Chart 6: Model Performance
            ax6 = axes[1, 2]
            model_acc = {}
            for model in self.models_to_test:
                m_results = [r for r in results if r['model'] == model]
                if m_results:
                    model_acc[model] = np.mean([r['stats']['accuracy_percentage'] for r in m_results])
            if model_acc:
                bars = ax6.bar(model_acc.keys(), model_acc.values(), alpha=0.8)
                ax6.set_ylabel('Accuracy (%)')
                ax6.set_title('Model Performance')
                ax6.tick_params(axis='x', rotation=30)
                for bar, val in zip(bars, model_acc.values()):
                    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.1f}%', ha='center', va='bottom')
            
            # Chart 7: Context Length x Strategy Heatmap
            ax7 = axes[2, 0]
            heatmap_data = []
            strat_labels = self.context_strategies
            ctx_labels = [str(l) for l in sorted(self.context_lengths)]
            for strategy in strat_labels:
                row = []
                for length in sorted(self.context_lengths):
                    combo = [r for r in results if r['context_strategy'] == strategy and r['context_length'] == length]
                    row.append(np.mean([r['stats']['accuracy_percentage'] for r in combo]) if combo else 0)
                heatmap_data.append(row)
            if heatmap_data:
                im = ax7.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
                ax7.set_xticks(range(len(ctx_labels)))
                ax7.set_yticks(range(len(strat_labels)))
                ax7.set_xticklabels(ctx_labels)
                ax7.set_yticklabels(strat_labels)
                ax7.set_xlabel('Context Length')
                ax7.set_ylabel('Strategy')
                ax7.set_title('Strategy x Context Length')
                for i in range(len(strat_labels)):
                    for j in range(len(ctx_labels)):
                        ax7.text(j, i, f'{heatmap_data[i][j]:.1f}%', ha='center', va='center', fontsize=9)
                plt.colorbar(im, ax=ax7, label='Accuracy (%)')
            
            # Chart 8: Framework x Model Heatmap
            ax8 = axes[2, 1]
            fm_data = []
            fw_labels = [framework_names.get(k, k) for k in self.frameworks_to_test]
            m_labels = self.models_to_test
            for fw_key in self.frameworks_to_test:
                row = []
                for model in m_labels:
                    combo = [r for r in results if r['framework'] == fw_key and r['model'] == model]
                    row.append(np.mean([r['stats']['accuracy_percentage'] for r in combo]) if combo else 0)
                fm_data.append(row)
            if fm_data:
                im = ax8.imshow(fm_data, cmap='viridis', aspect='auto')
                ax8.set_xticks(range(len(m_labels)))
                ax8.set_yticks(range(len(fw_labels)))
                ax8.set_xticklabels(m_labels, rotation=30)
                ax8.set_yticklabels(fw_labels)
                ax8.set_xlabel('Model')
                ax8.set_ylabel('Framework')
                ax8.set_title('Framework x Model')
                for i in range(len(fw_labels)):
                    for j in range(len(m_labels)):
                        ax8.text(j, i, f'{fm_data[i][j]:.1f}%', ha='center', va='center', color='white', fontsize=8)
                plt.colorbar(im, ax=ax8, label='Accuracy (%)')
            
            # Chart 9: Overall Winner
            ax9 = axes[2, 2]
            all_configs = {}
            for r in results:
                key = f"{r['framework_name']}\n{r['context_strategy']}"
                if key not in all_configs:
                    all_configs[key] = []
                all_configs[key].append(r['stats']['accuracy_percentage'])
            if all_configs:
                labels = list(all_configs.keys())
                values = [np.mean(v) for v in all_configs.values()]
                max_idx = values.index(max(values))
                colors = ['silver'] * len(values)
                colors[max_idx] = 'gold'
                bars = ax9.bar(range(len(labels)), values, color=colors, alpha=0.8)
                ax9.set_xticks(range(len(labels)))
                ax9.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
                ax9.set_ylabel('Accuracy (%)')
                ax9.set_title('Best Configuration')
                for bar, val in zip(bars, values):
                    ax9.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.1f}%', ha='center', va='bottom', fontsize=7)
            
            plt.tight_layout()
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            suffix = "_debug" if self.debug_mode else ""
            filename = f"math_benchmark_framework_comparison_charts{suffix}_{timestamp}.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"📈 Charts saved to {filename}")
            
        except Exception as e:
            print(f"⚠️ Could not create charts: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Run the Framework Comparison benchmark with CLI arguments (non-interactive)."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Math Tutor Framework Comparison Benchmark - GSM8K Dataset"
    )
    parser.add_argument('--debug', action='store_true', help='Debug mode (use selected configs)')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                        help='Models to test (e.g. metamath-mistral phi2)')
    parser.add_argument('--frameworks', type=str, nargs='+', default=None,
                        help='Frameworks to test (e.g. llamaindex langchain langchain_reasoning)')
    parser.add_argument('--context-lengths', type=int, nargs='+', default=None,
                        help='Context lengths to test (e.g. 512 1024 2048 4096)')
    parser.add_argument('--context-strategies', type=str, nargs='+', default=None,
                        help='Context strategies (e.g. fixed adaptive)')
    parser.add_argument('--num-questions', type=int, default=12,
                        help='Number of GSM8K questions per test (default: 12)')

    args = parser.parse_args()

    print("🧮 Math Tutor Framework Comparison Benchmark - GSM8K Dataset")
    print("=" * 70)
    print("Comparing: LlamaIndex vs LangChain (both using actual framework APIs)")
    print("Context strategies: Fixed (head truncation) vs Adaptive (Math-BERT + MMR)")
    print("Context lengths: 512, 1024, 2048, 4096 tokens")
    print("Metrics: Accuracy, Latency, Throughput, GPU/CPU/Memory utilization")

    debug_mode = args.debug
    num_questions = args.num_questions
    selected_models = args.models
    selected_frameworks = args.frameworks
    selected_context_lengths = args.context_lengths
    selected_context_strategies = args.context_strategies

    print(f"\nUsing {num_questions} questions per test")
    if debug_mode:
        print(f"🐛 DEBUG MODE")
        if selected_models:
            print(f"🎯 Selected models: {selected_models}")
        if selected_frameworks:
            print(f"🎭 Selected frameworks: {selected_frameworks}")
        if selected_context_lengths:
            print(f"📏 Selected context lengths: {selected_context_lengths}")
        if selected_context_strategies:
            print(f"🔧 Selected context strategies: {selected_context_strategies}")

    try:
        benchmark = MathConcurrentBenchmark(
            debug_mode=debug_mode,
            selected_models=selected_models,
            selected_frameworks=selected_frameworks,
            selected_context_lengths=selected_context_lengths,
            selected_context_strategies=selected_context_strategies,
            num_questions=num_questions
        )
        results = benchmark.run_math_benchmark_suite()
        
        print("\n🎉 Summary:")
        # Group results by framework, model, and context config
        summary = {}
        for result in results:
            key = f"{result['framework']}-{result['model']}-ctx{result.get('context_length', 1024)}-{result.get('context_strategy', 'adaptive')}"
            if key not in summary:
                summary[key] = []
            summary[key].append(result)
        
        for key, test_results in summary.items():
            avg_correctness = np.mean([r['stats']['accuracy_percentage'] for r in test_results])
            avg_latency = np.mean([r['stats']['avg_latency'] for r in test_results])
            avg_tokens_per_sec = np.mean([r['stats']['tokens_per_second'] for r in test_results])
            print(f"  {key}: {avg_correctness:.1f}% correct, {avg_latency:.2f}s avg, {avg_tokens_per_sec:.1f} tok/s")
        
        # Context length comparison summary
        print("\n📏 Context Length Comparison:")
        context_summary = {}
        for result in results:
            ctx_len = result.get('context_length', 1024)
            if ctx_len not in context_summary:
                context_summary[ctx_len] = []
            context_summary[ctx_len].append(result)
        
        for ctx_len in sorted(context_summary.keys()):
            ctx_results = context_summary[ctx_len]
            avg_correctness = np.mean([r['stats']['accuracy_percentage'] for r in ctx_results])
            avg_latency = np.mean([r['stats']['avg_latency'] for r in ctx_results])
            print(f"  {ctx_len} tokens: {avg_correctness:.1f}% correct, {avg_latency:.2f}s avg")
        
        # Context strategy comparison summary
        print("\n🔧 Context Strategy Comparison:")
        strategy_summary = {}
        for result in results:
            strategy = result.get('context_strategy', 'adaptive')
            if strategy not in strategy_summary:
                strategy_summary[strategy] = []
            strategy_summary[strategy].append(result)
        
        for strategy, strat_results in strategy_summary.items():
            avg_correctness = np.mean([r['stats']['accuracy_percentage'] for r in strat_results])
            avg_latency = np.mean([r['stats']['avg_latency'] for r in strat_results])
            print(f"  {strategy}: {avg_correctness:.1f}% correct, {avg_latency:.2f}s avg")
        
        # Framework comparison summary
        print("\n🎭 Framework Comparison:")
        framework_summary = {}
        for result in results:
            framework = result['framework_name']
            if framework not in framework_summary:
                framework_summary[framework] = []
            framework_summary[framework].append(result)
        
        for framework, framework_results in framework_summary.items():
            avg_correctness = np.mean([r['stats']['accuracy_percentage'] for r in framework_results])
            avg_latency = np.mean([r['stats']['avg_latency'] for r in framework_results])
            avg_tokens_per_sec = np.mean([r['stats']['tokens_per_second'] for r in framework_results])
            print(f"  {framework}: {avg_correctness:.1f}% correct, {avg_latency:.2f}s avg, {avg_tokens_per_sec:.1f} tok/s")
    
    except KeyboardInterrupt:
        print("\n\n⏹️ Benchmark interrupted by user")
    except Exception as e:
        print(f"\n❌ Error running benchmark: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Always clean up vLLM server + GPU memory
        try:
            model_manager.clear_cache()
        except Exception:
            pass

if __name__ == "__main__":
    main() 