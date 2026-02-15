"""
LlamaIndex RAG Pipeline with vLLM Backend

Implements Retrieval-Augmented Generation using:
  - LlamaIndex for document loading, node parsing, embedding, and retrieval
  - ChromaDB (persistent) as the vector store
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

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
import chromadb


# ── Prompt template ────────────────────────────────────────────

PROMPT_TEMPLATE = """You are a helpful Python tutor AI built with LlamaIndex and vLLM.
Use the retrieved context below to answer the user's question.
If you don't know the answer, say so. Keep responses concise and helpful.

Context from Python documentation:
{context}

User question: {question}

Your helpful answer:"""


# ── LlamaIndex RAG class ──────────────────────────────────────

class LlamaIndexRAG:
    """
    LlamaIndex-based RAG pipeline.

    Initialise once (loads docs & builds index), then call:
      - query(question, llm_manager)            single question
      - batch_query(questions, llm_manager)      batched retrieval + batched generation
    """

    # Default directory to persist the ChromaDB vector store
    DEFAULT_PERSIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db_llamaindex")

    COLLECTION_NAME = "python_docs_llamaindex"

    def __init__(self, docs_path: str, embed_model_name: str = "BAAI/bge-small-en",
                 chunk_size: int = 200, chunk_overlap: int = 10, top_k: int = 5,
                 persist_dir: str = None):
        self.top_k = top_k
        self.docs_path = docs_path
        self.persist_dir = persist_dir or self.DEFAULT_PERSIST_DIR

        print("[LlamaIndex RAG] Initialising...")
        t0 = time.time()

        # Embedding model
        self.embed_model = HuggingFaceEmbedding(model_name=embed_model_name)

        # Use persistent ChromaDB client
        os.makedirs(self.persist_dir, exist_ok=True)
        chroma_client = chromadb.PersistentClient(path=self.persist_dir)

        if self._index_exists(chroma_client):
            # Load existing collection from disk
            print(f"  Loading existing vector store from {self.persist_dir}")
            chroma_collection = chroma_client.get_collection(self.COLLECTION_NAME)
            print(f"  Loaded {chroma_collection.count()} vectors from disk")
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            self.index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                embed_model=self.embed_model,
            )
        else:
            # Load documents
            documents = SimpleDirectoryReader(input_dir=docs_path, recursive=True).load_data()
            print(f"  Loaded {len(documents)} documents")

            # Parse into nodes
            splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            nodes = splitter.get_nodes_from_documents(documents)
            print(f"  Created {len(nodes)} nodes")

            # Build persistent ChromaDB vector store
            chroma_collection = chroma_client.get_or_create_collection(self.COLLECTION_NAME)
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            # Build index (embeddings are persisted to ChromaDB on disk)
            self.index = VectorStoreIndex(
                nodes=nodes,
                storage_context=storage_context,
                embed_model=self.embed_model,
            )
            print(f"  Vector store persisted to {self.persist_dir}")

        self.retriever = self.index.as_retriever(similarity_top_k=top_k)

        print(f"[LlamaIndex RAG] Ready  ({time.time()-t0:.1f}s)\n")

    def _index_exists(self, chroma_client) -> bool:
        """Check if the collection already exists and has data."""
        try:
            collection = chroma_client.get_collection(self.COLLECTION_NAME)
            return collection.count() > 0
        except Exception:
            return False

    # ── Retrieval ──────────────────────────────────────────────

    def retrieve(self, question: str) -> str:
        """Retrieve relevant nodes and return formatted context string."""
        nodes = self.retriever.retrieve(question)
        return "\n\n---\n\n".join(node.text for node in nodes)

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
        del self.index
        del self.retriever
        print("[LlamaIndex RAG] Cleaned up (vector store persisted on disk).")
