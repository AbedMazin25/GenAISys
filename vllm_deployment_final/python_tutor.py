"""
Python Tutor - Interactive RAG chatbot with vLLM backend.

Supports both LangChain and LlamaIndex for retrieval.
The vLLM server is auto-managed (starts on entry, stops on exit).

Usage:
    python python_tutor.py --framework langchain --model llama2-7b
    python python_tutor.py --framework llamaindex --model phi2
    python python_tutor.py --framework langchain --benchmark --num-requests 20
"""

import time
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(__file__))
import vllm_model_manager as model_manager


def get_docs_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python-3.13-docs-text')


def create_rag(framework: str):
    """Create the appropriate RAG pipeline."""
    docs_path = get_docs_path()
    if framework == 'langchain':
        from rag_langchain import LangChainRAG
        return LangChainRAG(docs_path)
    elif framework == 'llamaindex':
        from rag_llamaindex import LlamaIndexRAG
        return LlamaIndexRAG(docs_path)
    else:
        raise ValueError(f"Unknown framework: {framework}")


def interactive_tutor(framework: str, model_name: str, port: int = 8000):
    """Run interactive Python tutor with real RAG."""
    print(f"\n{'='*60}")
    print(f"Python Tutor ({framework} + vLLM)")
    print(f"{'='*60}")
    print(f"Framework: {framework}  |  Model: {model_name}")
    print("Type 'exit' to quit")
    print(f"{'='*60}\n")

    # Create RAG pipeline
    rag = create_rag(framework)

    try:
        with model_manager.VLLMModelManager(model_name, port=port) as llm:
            print(f"\nReady! Ask any Python question.\n")

            while True:
                try:
                    question = input("\nYour question: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nGoodbye!")
                    break

                if question.lower() in ['exit', 'quit', 'q']:
                    print("Goodbye!")
                    break

                if not question:
                    continue

                print(f"\n[{framework}] Retrieving context + generating via vLLM...")
                start = time.time()
                answer = rag.query(question, llm)
                elapsed = time.time() - start

                print(f"\nAnswer ({elapsed:.2f}s):")
                print(answer)

            # Show stats
            stats = llm.get_stats()
            print(f"\nSession Stats:")
            print(f"  Requests: {stats['total_requests']}")
            print(f"  Avg latency: {stats.get('avg_latency', 0):.2f}s")
            print(f"  Tokens/s: {stats.get('tokens_per_second', 0):.2f}")

        print("Server stopped.")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        rag.cleanup()


def benchmark_mode(framework: str, model_name: str, num_requests: int = 10, port: int = 8000):
    """Benchmark the RAG pipeline with batched requests."""
    print(f"\n{'='*60}")
    print(f"RAG Benchmark ({framework} + vLLM)  |  {num_requests} requests")
    print(f"{'='*60}\n")

    rag = create_rag(framework)

    test_questions = [
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
        "How do I read a file?",
        "What is a tuple?",
        "Explain inheritance in Python",
        "What are context managers?",
        "How do async functions work?",
    ]
    questions = (test_questions * (num_requests // len(test_questions) + 1))[:num_requests]

    try:
        with model_manager.VLLMModelManager(model_name, port=port) as llm:
            # Warmup
            print("Warming up...")
            rag.batch_query(questions[:3], llm, max_tokens=32)
            time.sleep(1)

            # Benchmark: batched RAG
            print(f"\nRunning {len(questions)} batched RAG queries...")
            start = time.time()
            results = rag.batch_query(questions, llm, max_tokens=256)
            elapsed = time.time() - start

            successes = sum(1 for _, _, err in results if not err)
            latencies = [lat for lat, _, err in results if not err]
            total_tokens = sum(len(resp.split()) for _, resp, err in results if not err)

            print(f"\nResults:")
            print(f"  Total time: {elapsed:.2f}s")
            print(f"  Successful: {successes}/{len(questions)}")
            print(f"  Avg latency/request: {sum(latencies)/len(latencies):.2f}s" if latencies else "")
            print(f"  Throughput: {successes/elapsed:.2f} req/s")
            print(f"  Tokens/s: {total_tokens/elapsed:.1f}")

        print("Server stopped.")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        rag.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Python Tutor with RAG + vLLM")
    parser.add_argument("--framework", type=str, default="langchain",
                        choices=["langchain", "llamaindex"],
                        help="RAG framework to use")
    parser.add_argument("--model", type=str, default="phi2",
                        choices=list(model_manager.MODELS.keys()),
                        help="LLM model")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--benchmark", action="store_true",
                        help="Run benchmark instead of interactive mode")
    parser.add_argument("--num-requests", type=int, default=10,
                        help="Number of requests for benchmark mode")

    args = parser.parse_args()

    if args.benchmark:
        benchmark_mode(args.framework, args.model, args.num_requests, args.port)
    else:
        interactive_tutor(args.framework, args.model, args.port)
