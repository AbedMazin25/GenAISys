"""
LangChain RAG Pipeline with vLLM Backend

Implements Retrieval-Augmented Generation using:
  - LangChain for document loading, chunking, embedding, and retrieval
  - ChromaDB as the vector store
  - vLLM (via VLLMModelManager) for LLM generation

Supports both single queries and batched queries where:
  1. Retrieval is parallelized across questions (ThreadPoolExecutor)
  2. All RAG-augmented prompts are sent to vLLM via the model manager
  3. vLLM batches them on the GPU for maximum throughput
"""

import os
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ── Robust loader to handle encoding issues ────────────────────

class RobustTextLoader(TextLoader):
    """TextLoader that gracefully handles encoding errors."""

    def __init__(self, file_path: str):
        super().__init__(file_path, encoding='utf-8')

    def load(self):
        try:
            return super().load()
        except UnicodeDecodeError:
            for enc in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    self.encoding = enc
                    return super().load()
                except UnicodeDecodeError:
                    continue
            return []
        except Exception:
            return []


# ── Prompt template ────────────────────────────────────────────

PROMPT_TEMPLATE = """You are a helpful Python tutor AI built with LangChain and vLLM.
Use the retrieved context below to answer the user's question.
If you don't know the answer, say so. Keep responses concise and helpful.

Context from Python documentation:
{context}

User question: {question}

Your helpful answer:"""


# ── LangChain RAG class ───────────────────────────────────────

class LangChainRAG:
    """
    LangChain-based RAG pipeline.

    Initialise once (loads docs & builds index), then call:
      - query(question, llm_manager)            single question
      - batch_query(questions, llm_manager)      batched retrieval + batched generation
    """

    # Default directory to persist the ChromaDB vector store
    DEFAULT_PERSIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db_langchain")

    def __init__(self, docs_path: str, embed_model_name: str = "BAAI/bge-small-en",
                 chunk_size: int = 200, chunk_overlap: int = 10, top_k: int = 5,
                 persist_dir: str = None):
        self.top_k = top_k
        self.embed_model_name = embed_model_name
        self.docs_path = docs_path
        self.persist_dir = persist_dir or self.DEFAULT_PERSIST_DIR

        print("[LangChain RAG] Initialising...")
        t0 = time.time()

        # Embedding model
        self.embed_model = HuggingFaceEmbeddings(model_name=embed_model_name)

        if self._index_exists():
            # Load existing persisted vector store
            print(f"  Loading existing vector store from {self.persist_dir}")
            self.vectorstore = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embed_model,
                collection_name="python_docs_langchain",
            )
            print(f"  Loaded {self.vectorstore._collection.count()} vectors from disk")
        else:
            # Load documents
            documents = self._load_documents(docs_path)
            print(f"  Loaded {len(documents)} documents")

            # Split into chunks
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
            )
            chunks = splitter.split_documents(documents)
            print(f"  Created {len(chunks)} chunks")

            # Build and persist vector store
            os.makedirs(self.persist_dir, exist_ok=True)
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embed_model,
                collection_name="python_docs_langchain",
                persist_directory=self.persist_dir,
            )
            print(f"  Vector store persisted to {self.persist_dir}")

        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": top_k})

        print(f"[LangChain RAG] Ready  ({time.time()-t0:.1f}s)\n")

    def _index_exists(self) -> bool:
        """Check if a persisted ChromaDB already exists on disk."""
        if not os.path.isdir(self.persist_dir):
            return False
        # ChromaDB creates a chroma.sqlite3 file in the persist directory
        return os.path.exists(os.path.join(self.persist_dir, "chroma.sqlite3"))

    # ── Document loading ───────────────────────────────────────

    @staticmethod
    def _load_documents(docs_path: str):
        """Load .txt files with robust encoding handling."""
        try:
            loader = DirectoryLoader(
                docs_path, glob="**/*.txt",
                loader_cls=RobustTextLoader,
                recursive=True, show_progress=True, use_multithreading=True,
            )
            documents = loader.load()
        except Exception as e:
            print(f"  DirectoryLoader failed ({e}), falling back to manual walk")
            documents = []
            for root, _, files in os.walk(docs_path):
                for f in files:
                    if f.endswith('.txt'):
                        try:
                            documents.extend(RobustTextLoader(os.path.join(root, f)).load())
                        except Exception:
                            continue

        return [d for d in documents if d.page_content.strip()]

    # ── Retrieval ──────────────────────────────────────────────

    def retrieve(self, question: str) -> str:
        """Retrieve relevant chunks and return formatted context string."""
        docs = self.retriever.invoke(question)
        return "\n\n---\n\n".join(d.page_content for d in docs)

    def retrieve_batch(self, questions: List[str], max_workers: int = 8) -> List[str]:
        """Retrieve contexts for many questions in parallel using threads."""
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            return list(pool.map(self.retrieve, questions))

    # ── Prompt building ────────────────────────────────────────

    @staticmethod
    def build_prompt(question: str, context: str) -> str:
        return PROMPT_TEMPLATE.format(context=context, question=question)

    def build_prompts(self, questions: List[str], contexts: List[str]) -> List[str]:
        return [self.build_prompt(q, c) for q, c in zip(questions, contexts)]

    # ── Single query (sync) ────────────────────────────────────

    def query(self, question: str, llm_manager) -> str:
        """
        Full RAG query: retrieve context -> build prompt -> generate via vLLM.
        Uses llm_manager.generate() which sends HTTP request to the vLLM server.
        """
        context = self.retrieve(question)
        prompt = self.build_prompt(question, context)
        return llm_manager.generate(prompt, max_tokens=256, temperature=0.7)

    # ── Batched query ──────────────────────────────────────────

    async def batch_query_async(self, questions: List[str], llm_manager,
                                max_tokens: int = 256, temperature: float = 0.7):
        """
        Full batched RAG pipeline:
          1. Retrieve contexts for ALL questions in parallel (threaded)
          2. Build ALL RAG prompts
          3. Send ALL prompts to vLLM at the same time via llm_manager (async HTTP)
          4. vLLM batches them on the GPU

        For concurrency=48, all 48 questions are retrieved, then all 48
        RAG-augmented prompts hit vLLM simultaneously.

        Returns list of (latency, response, error) tuples.
        """
        # Step 1: retrieve contexts for ALL questions in parallel (ThreadPoolExecutor)
        loop = asyncio.get_running_loop()
        contexts = await loop.run_in_executor(None, self.retrieve_batch, questions)

        # Step 2: build ALL RAG prompts (question + retrieved context)
        prompts = self.build_prompts(questions, contexts)

        # Step 3: fire ALL prompts to vLLM at the same time via the model manager
        # llm_manager.generate_batch_async sends N async HTTP POSTs concurrently
        # vLLM's continuous batching engine processes them together on the GPU
        return await llm_manager.generate_batch_async(prompts, max_tokens, temperature)

    def batch_query(self, questions: List[str], llm_manager,
                    max_tokens: int = 256, temperature: float = 0.7):
        """Synchronous wrapper for batch_query_async."""
        return asyncio.run(
            self.batch_query_async(questions, llm_manager,
                                   max_tokens=max_tokens, temperature=temperature)
        )

    # ── Cleanup ────────────────────────────────────────────────

    def cleanup(self):
        """Release resources (persisted data remains on disk for next run)."""
        del self.vectorstore
        del self.retriever
        print("[LangChain RAG] Cleaned up (vector store persisted on disk).")
