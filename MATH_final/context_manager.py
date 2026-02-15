from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import re

# Math-specialized BERT embeddings
_math_bert_model = None
_math_bert_tokenizer = None
_math_bert_loaded = False

# Precomputed example cache
_example_cache = None  # dict with 'embeddings', 'operations', 'token_counts'

# Fallback sentence transformer
try:
    from sentence_transformers import SentenceTransformer
    _sentence_model = None
    _sentence_model_loaded = False
except ImportError:
    SentenceTransformer = None
    _sentence_model = None
    _sentence_model_loaded = False

@dataclass
class ProcessedContext:
    """Result of context processing"""
    processed_text: str
    original_length: int
    processed_length: int
    truncated: bool
    truncation_method: str
    importance_score: float
    # Add debug info
    debug_info: Optional[str] = None

class ContextManager:
    """Manages context window for LLM inputs"""
    
    def __init__(self, tokenizer_name: str, strategy: str = "fixed", max_length: int = 1024):
        self.strategy = strategy
        self.max_length = max_length
        self.tokenizer_name = tokenizer_name
        self.debug = False  # Debug off by default for clean output
        
        # Load tokenizer for token counting
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        except Exception as e:
            print(f"Warning: Could not load tokenizer {tokenizer_name}: {e}")
            self.tokenizer = None
        
        # Initialize sentence transformer for novel adaptive strategy
        if strategy == "adaptive":
            self._init_sentence_transformer()
    
    def _init_sentence_transformer(self):
        """Initialize Math-BERT for math-specialized embeddings"""
        global _math_bert_model, _math_bert_tokenizer, _math_bert_loaded
        global _sentence_model, _sentence_model_loaded
        
        if _math_bert_loaded:
            return
        
        # Try to load Math-BERT first (preferred for math problems)
        # Keep on CPU to avoid competing with vLLM for GPU memory
        try:
            _math_bert_tokenizer = AutoTokenizer.from_pretrained('AnReu/math_pretrained_bert')
            _math_bert_model = AutoModel.from_pretrained('AnReu/math_pretrained_bert')
            _math_bert_model.eval()
            
            _math_bert_loaded = True
            print("[SUCCESS] Math-BERT (AnReu/math_pretrained_bert) initialized on CPU for adaptive context strategy")
            return
        except Exception as e:
            print(f"Warning: Could not load Math-BERT: {e}")
        
        # Fallback to sentence transformers
        if SentenceTransformer is None:
            print("Warning: sentence-transformers not available, using fallback method")
            _math_bert_loaded = True
            _sentence_model_loaded = True
            return
        
        try:
            _sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            _sentence_model_loaded = True
            _math_bert_loaded = True
            print("[SUCCESS] Fallback embedder (all-MiniLM-L6-v2) initialized")
        except Exception as e2:
            print(f"Warning: Could not load any embedding model: {e2}")
            _sentence_model = None
            _sentence_model_loaded = True
            _math_bert_loaded = True
    
    def _get_math_bert_embedding(self, text: str) -> np.ndarray:
        """Get embedding from Math-BERT"""
        global _math_bert_model, _math_bert_tokenizer
        
        if _math_bert_model is None:
            return None
        
        with torch.no_grad():
            inputs = _math_bert_tokenizer(
                text, 
                return_tensors='pt', 
                truncation=True, 
                max_length=512,
                padding=True
            )
            # Math-BERT runs on CPU (GPU reserved for vLLM)
            outputs = _math_bert_model(**inputs)
            # Use [CLS] token embedding
            embedding = outputs.last_hidden_state[:, 0, :].numpy()
            
        return embedding[0]
    
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings using Math-BERT or fallback"""
        global _math_bert_model, _sentence_model
        
        if _math_bert_model is not None:
            embeddings = []
            for text in texts:
                emb = self._get_math_bert_embedding(text)
                embeddings.append(emb)
            return np.array(embeddings)
        elif _sentence_model is not None:
            return _sentence_model.encode(texts)
        else:
            return None
    
    def set_debug(self, debug: bool):
        """Enable/disable debug printing"""
        self.debug = debug
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text without triggering max_length warnings.
        We're only counting here - the context manager handles truncation separately."""
        if self.tokenizer:
            import logging
            logger = logging.getLogger("transformers.tokenization_utils_base")
            prev_level = logger.level
            logger.setLevel(logging.ERROR)
            try:
                count = len(self.tokenizer.encode(text, add_special_tokens=False))
            finally:
                logger.setLevel(prev_level)
            return count
        else:
            # Fallback: approximate (1 token ≈ 0.75 words)
            return int(len(text.split()) * 1.33)
    
    def process_context(self, text: str, query: str = "") -> ProcessedContext:
        """Process context according to strategy with optional debug printing"""
        original_length = self.count_tokens(text)
        
        # Only print debug info if debug is enabled
        if self.debug:
            print(f"\n[DEBUG] CONTEXT PROCESSING DEBUG - Strategy: {self.strategy}, Max Length: {self.max_length}")
            print(f"[LENGTH] Original Length: {original_length} tokens")
            print(f"[QUERY] Query: {query[:100]}..." if len(query) > 100 else f"[QUERY] Query: {query}")
            print(f"[BEFORE] BEFORE CONTEXT (first 500 chars):")
            print("-" * 50)
            print(text[:500])
            print("-" * 50)
        
        if self.strategy == "fixed":
            result = self._process_fixed(text, original_length)
        elif self.strategy == "adaptive":
            result = self._process_adaptive_novel(text, query, original_length)
        else:
            raise ValueError(f"Unknown context strategy: {self.strategy}")
        
        # Only print debug info if debug is enabled
        if self.debug:
            print(f"[AFTER] AFTER CONTEXT (first 500 chars):")
            print("-" * 50)
            print(result.processed_text[:500])
            print("-" * 50)
            print(f"[RESULTS] Processing Results:")
            print(f"    Processed Length: {result.processed_length} tokens")
            print(f"    Truncated: {result.truncated}")
            print(f"    Method: {result.truncation_method}")
            print(f"    Importance Score: {result.importance_score:.3f}")
            if result.debug_info:
                print(f"    Debug Info: {result.debug_info}")
            print(f"[END] END CONTEXT DEBUG\n")
        
        return result
    
    def _process_fixed(self, text: str, original_length: int) -> ProcessedContext:
        """Fixed-size truncation cutting whole QA pairs from HEAD (beginning) to preserve the question at the end"""
        # REDUCED safety buffer to maximize context usage
        SAFETY_BUFFER = 3  # Minimal buffer - be more aggressive in using available context
        effective_max_length = max(50, self.max_length - SAFETY_BUFFER)
        
        if original_length <= effective_max_length:
            return ProcessedContext(
                processed_text=text,
                original_length=original_length,
                processed_length=original_length,
                truncated=False,
                truncation_method="none",
                importance_score=1.0
            )
        
        # Extract QA pairs for proper truncation
        qa_pairs = self._extract_qa_pairs_for_truncation(text)
        
        if not qa_pairs:
            # Fallback to simple truncation if no QA pairs found
            return self._simple_truncate_from_head(text, original_length, effective_max_length)
        
        # Find the final question (should be preserved)
        final_question = qa_pairs[-1] if qa_pairs else ""
        final_question_tokens = self.count_tokens(final_question)
        
        # If just the final question is too long, truncate it carefully
        if final_question_tokens >= effective_max_length:
            return self._simple_truncate_from_head(text, original_length, effective_max_length)
        
        # Select QA pairs from the end, working backwards
        selected_pairs = []
        total_tokens = final_question_tokens
        
        # Add QA pairs from the end until we hit the limit
        for qa_pair in reversed(qa_pairs[:-1]):  # Exclude the final question
            pair_tokens = self.count_tokens(qa_pair)
            if total_tokens + pair_tokens <= effective_max_length:
                selected_pairs.insert(0, qa_pair)  # Insert at beginning to maintain order
                total_tokens += pair_tokens
            # Don't break - try smaller pairs that might still fit
        
        # Add the final question
        selected_pairs.append(final_question)
        
        processed_text = "".join(selected_pairs)
        processed_length = self.count_tokens(processed_text)
        
        # Final safety check: tokenizer boundary effects can cause
        # concatenated text to have different token count than sum of parts
        while processed_length > effective_max_length and len(selected_pairs) > 1:
            selected_pairs.pop(0)  # Remove earliest example
            processed_text = "".join(selected_pairs)
            processed_length = self.count_tokens(processed_text)
        
        return ProcessedContext(
            processed_text=processed_text,
            original_length=original_length,
            processed_length=processed_length,
            truncated=True,
            truncation_method="fixed_cut_whole_qa_pairs",
            importance_score=len(selected_pairs) / len(qa_pairs) if qa_pairs else 0.5
        )
    
    def _extract_qa_pairs_for_truncation(self, text: str) -> list[str]:
        """Extract QA pairs as complete blocks for truncation"""
        qa_pairs = []
        
        # Look for Q: ... A: ... patterns
        parts = text.split("Q:")
        if len(parts) < 2:
            return qa_pairs
        
        for i, part in enumerate(parts[1:], 1):  # Skip first empty part
            if "A:" in part:
                qa_text = f"Q:{part}"
                
                # Find the end of this QA pair (before next Q: or end of text)
                next_q_pos = qa_text.find("\n\nQ:", qa_text.find("A:"))
                if next_q_pos != -1:
                    qa_text = qa_text[:next_q_pos + 2]  # Include the \n\n
                
                qa_pairs.append(qa_text)
        
        return qa_pairs
    
    def _simple_truncate_from_head(self, text: str, original_length: int, effective_max_length: int) -> ProcessedContext:
        """Simple token-based truncation from head as fallback"""
        if self.tokenizer:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            # Take the LAST effective_max_length-2 tokens (cut from beginning, preserve end)
            start_idx = max(0, len(tokens) - (effective_max_length - 2))
            truncated_tokens = tokens[start_idx:]
            processed_text = self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
        else:
            words = text.split()
            # Take the LAST portion of words (cut from beginning, preserve end)
            start_idx = max(0, len(words) - int(effective_max_length * 0.75))
            truncated_words = words[start_idx:]
            processed_text = " ".join(truncated_words)
        
        processed_length = self.count_tokens(processed_text)
        
        return ProcessedContext(
            processed_text=processed_text,
            original_length=original_length,
            processed_length=processed_length,
            truncated=True,
            truncation_method="fixed_cut_from_head_fallback",
            importance_score=processed_length / original_length
        )
    
    def _process_adaptive_novel(self, text: str, query: str, original_length: int) -> ProcessedContext:
        """IMPROVED: Maximize Context Usage with Math-Specific Selection"""
        # MUCH REDUCED safety buffer to maximize context usage - be more aggressive!
        SAFETY_BUFFER = 3  # Minimal buffer to maximize context usage
        effective_max_length = max(50, self.max_length - SAFETY_BUFFER)
        
        # Use the effective max length for processing
        if original_length <= effective_max_length:
            return ProcessedContext(
                processed_text=text,
                original_length=original_length,
                processed_length=original_length,
                truncated=False,
                truncation_method="none",
                importance_score=1.0
            )
        
        # IMPROVED: Math-Specific Adaptive Selection
        processed_text, method, importance, debug_info = self._math_specific_selection(text, query, effective_max_length)
        processed_length = self.count_tokens(processed_text)
        
        return ProcessedContext(
            processed_text=processed_text,
            original_length=original_length,
            processed_length=processed_length,
            truncated=True,
            truncation_method=method,
            importance_score=importance,
            debug_info=debug_info
        )
    
    def _precompute_example_cache(self, few_shot_examples: list) -> dict:
        """Precompute and cache embeddings, operations, and token counts for all examples.
        
        Saves to disk so subsequent runs skip embedding entirely.
        Only the query embedding is computed per question.
        """
        global _example_cache
        
        import os
        import hashlib
        
        # Create a cache key from example questions (order-independent)
        questions_str = "|".join(sorted(ex['question'] for ex in few_shot_examples))
        cache_hash = hashlib.md5(questions_str.encode()).hexdigest()[:12]
        cache_file = os.path.join(os.path.dirname(__file__), f"_example_cache_{cache_hash}.npz")
        
        # Try loading from disk
        if os.path.exists(cache_file):
            try:
                cached = np.load(cache_file, allow_pickle=True)
                _example_cache = {
                    'embeddings': cached['embeddings'],
                    'operations': cached['operations'].tolist(),
                    'token_counts': cached['token_counts'].tolist(),
                    'questions': cached['questions'].tolist(),
                    'hash': cache_hash
                }
                if self.debug:
                    print(f"[CACHE] Loaded precomputed embeddings from {cache_file}")
                return _example_cache
            except Exception as e:
                if self.debug:
                    print(f"[CACHE] Failed to load cache: {e}")
        
        # Compute embeddings
        example_texts = [ex['question'] for ex in few_shot_examples]
        embeddings = self._get_embeddings(example_texts)
        
        if embeddings is None:
            return None
        
        # Compute operations and token counts
        operations = [self._detect_operations(ex['question']) for ex in few_shot_examples]
        token_counts = [self.count_tokens(ex['text']) for ex in few_shot_examples]
        questions = [ex['question'] for ex in few_shot_examples]
        
        _example_cache = {
            'embeddings': embeddings,
            'operations': operations,
            'token_counts': token_counts,
            'questions': questions,
            'hash': cache_hash
        }
        
        # Save to disk
        try:
            np.savez(cache_file, 
                     embeddings=embeddings, 
                     operations=np.array(operations, dtype=object),
                     token_counts=np.array(token_counts),
                     questions=np.array(questions, dtype=object))
            print(f"[CACHE] Saved precomputed embeddings to {cache_file} ({len(few_shot_examples)} examples)")
        except Exception as e:
            if self.debug:
                print(f"[CACHE] Failed to save cache: {e}")
        
        return _example_cache

    def _math_specific_selection(self, text: str, query: str, max_length: int) -> tuple[str, str, float, str]:
        """IMPROVED: Math-BERT + Diversity + Operation Matching + Reranking
        
        Uses precomputed embeddings for few-shot examples (only query is embedded per call).
        Cache is based on the canonical example pool, NOT the parsed prompt.
        """
        global _math_bert_model, _sentence_model, _example_cache
        
        # Extract few-shot examples from prompt
        few_shot_examples = self._extract_few_shot_examples(text)
        
        if not few_shot_examples:
            return self._smart_truncate_fallback(text, query, max_length)
        
        # Check if we have any embedding model
        if _math_bert_model is None and _sentence_model is None:
            return self._smart_truncate_fallback(text, query, max_length)
        
        try:
            # Get the final question that needs to be preserved
            final_question_part = self._extract_final_question(text)
            final_question_tokens = self.count_tokens(final_question_part)
            
            # Use canonical example pool for caching (not parsed prompt which varies per question)
            # This ensures cache is computed once and reused for ALL questions
            try:
                from few_shot_examples import EXAMPLES as canonical_examples
                cache_examples = [{'question': ex['question'], 'text': f"Q: {ex['question']}\nA: {ex['chain']} The answer is {ex['answer']}.\n\n", 'answer': ex.get('chain',''), 'original_index': i} for i, ex in enumerate(canonical_examples)]
            except ImportError:
                cache_examples = few_shot_examples
            
            if _example_cache is None:
                _example_cache = self._precompute_example_cache(cache_examples)
            
            if _example_cache is None:
                return self._smart_truncate_fallback(text, query, max_length)
            
            # Only embed the query (fast - single forward pass)
            query_embedding = self._get_embeddings([query])
            if query_embedding is None:
                return self._smart_truncate_fallback(text, query, max_length)
            query_embedding = query_embedding[0]
            
            # Build lookup from cached data (question -> index in cache)
            cached_questions = list(_example_cache['questions'])
            cached_embeddings = _example_cache['embeddings']
            cached_ops = _example_cache['operations']
            cached_token_counts = _example_cache['token_counts']
            question_to_cache_idx = {q: i for i, q in enumerate(cached_questions)}
            
            # Detect query operation type
            query_ops = self._detect_operations(query)
            
            # Score examples using cached data
            scored_examples = []
            for i, example in enumerate(few_shot_examples):
                # Skip the final question to avoid duplication
                if self._is_final_question(example['question'], query):
                    continue
                
                # Look up cached embedding by question text
                cache_idx = question_to_cache_idx.get(example['question'])
                if cache_idx is None:
                    continue  # Example not in cache, skip
                
                embedding = cached_embeddings[cache_idx]
                
                # 1. Semantic similarity (cached embedding vs fresh query embedding)
                similarity = self._cosine_similarity(query_embedding, embedding)
                
                # 2. Operation type matching (cached)
                example_ops = cached_ops[cache_idx]
                op_match = self._operation_match_score(query_ops, example_ops)
                
                # 3. Complexity matching
                complexity_match = self._calculate_complexity_match(query, example['question'])
                
                # Composite score
                composite_score = (
                    similarity * 0.35 +      # Math-BERT semantic similarity
                    op_match * 0.45 +        # Operation type matching
                    complexity_match * 0.20  # Complexity matching
                )
                
                scored_examples.append({
                    'index': i,
                    'score': composite_score,
                    'example': example,
                    'embedding': embedding,
                    'ops': example_ops,
                    'token_count': cached_token_counts[cache_idx]  # Cached token count
                })
            
            # DIVERSITY-AWARE SELECTION (MMR-style)
            selected_examples = self._select_with_diversity(
                scored_examples, 
                query_embedding,
                max_length, 
                final_question_tokens,
                diversity_lambda=0.3  # Balance relevance vs diversity
            )
            
            # FINAL RERANKING for optimal ordering
            if selected_examples and len(selected_examples) > 1:
                selected_examples = self._rerank_for_output(selected_examples, query, query_ops)
            
            # Reconstruct the context
            if selected_examples:
                selected_text = "".join([ex['example']['text'] for ex in selected_examples])
                final_text = selected_text + final_question_part
                
                # Verify we're under the limit
                # Note: encoding the full string can differ from sum of parts due to
                # tokenizer boundary effects, so we allow a small overflow margin
                final_tokens = self.count_tokens(final_text)
                if final_tokens > max_length:
                    # Try removing the last (lowest-scored) example to fit
                    while selected_examples and final_tokens > max_length:
                        removed = selected_examples.pop(0)  # Remove first (least relevant after rerank)
                        selected_text = "".join([ex['example']['text'] for ex in selected_examples])
                        final_text = selected_text + final_question_part
                        final_tokens = self.count_tokens(final_text)
                        if self.debug:
                            print(f"  [FIT] Removed example to fit: {final_tokens}/{max_length}")
                    
                    if not selected_examples:
                        return self._smart_truncate_fallback(text, query, max_length)
                
                importance = len(selected_examples) / len(few_shot_examples) if few_shot_examples else 0.5
                method = "math_bert_diversity_rerank"
                debug_info = f"Math-BERT+Diversity selected {len(selected_examples)}/{len(few_shot_examples)} examples. Ops: {query_ops}. Used {final_tokens}/{max_length} tokens ({(final_tokens/max_length)*100:.1f}%)"
                
                return final_text, method, importance, debug_info
            else:
                return self._smart_truncate_fallback(text, query, max_length)
                
        except Exception as e:
            if self.debug:
                print(f"[ERROR] Math-specific selection failed: {e}")
            return self._smart_truncate_fallback(text, query, max_length)
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors"""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    def _detect_operations(self, text: str) -> set:
        """Detect mathematical operations in text"""
        text_lower = text.lower()
        ops = set()
        
        # Addition indicators
        if any(w in text_lower for w in ['add', 'plus', 'more', 'total', 'sum', 'together', 'combined', 'increase', 'gain', 'bought', 'got', 'received']):
            ops.add('addition')
        
        # Subtraction indicators  
        if any(w in text_lower for w in ['subtract', 'minus', 'less', 'left', 'remain', 'difference', 'decrease', 'lost', 'gave', 'spent', 'sold', 'ate', 'used']):
            ops.add('subtraction')
        
        # Multiplication indicators
        if any(w in text_lower for w in ['multiply', 'times', 'each', 'per', 'every', 'product', 'double', 'triple']):
            ops.add('multiplication')
        
        # Division indicators
        if any(w in text_lower for w in ['divide', 'split', 'share', 'each get', 'per person', 'average', 'half', 'quarter']):
            ops.add('division')
        
        # Multi-step indicator
        if len(ops) > 1 or any(w in text_lower for w in ['then', 'after', 'first', 'next', 'finally']):
            ops.add('multi_step')
        
        return ops if ops else {'unknown'}
    
    def _operation_match_score(self, query_ops: set, example_ops: set) -> float:
        """Score how well operations match between query and example"""
        if not query_ops or not example_ops:
            return 0.5
        
        # Exact match bonus
        intersection = query_ops.intersection(example_ops)
        union = query_ops.union(example_ops)
        
        if not union:
            return 0.5
        
        jaccard = len(intersection) / len(union)
        
        # Bonus for matching primary operation
        primary_ops = {'addition', 'subtraction', 'multiplication', 'division'}
        query_primary = query_ops.intersection(primary_ops)
        example_primary = example_ops.intersection(primary_ops)
        
        if query_primary and example_primary:
            if query_primary == example_primary:
                jaccard = min(1.0, jaccard + 0.3)  # Exact primary match bonus
            elif query_primary.intersection(example_primary):
                jaccard = min(1.0, jaccard + 0.15)  # Partial primary match bonus
        
        return jaccard
    
    def _select_with_diversity(self, scored_examples: list, query_embedding: np.ndarray,
                               max_length: int, final_question_tokens: int, 
                               diversity_lambda: float = 0.3) -> list:
        """MMR-style selection: balance relevance with diversity.
        
        Uses cached token_count from scored_examples to avoid redundant tokenizer calls.
        """
        if not scored_examples:
            return []
        
        selected = []
        used_tokens = final_question_tokens
        candidates = scored_examples.copy()
        
        while candidates:
            best_score = -float('inf')
            best_candidate = None
            best_idx = -1
            
            for idx, candidate in enumerate(candidates):
                # Use cached token count if available, else compute
                example_tokens = candidate.get('token_count', self.count_tokens(candidate['example']['text']))
                
                # Skip if doesn't fit
                if used_tokens + example_tokens > max_length:
                    continue
                
                # Relevance score (from initial scoring)
                relevance = candidate['score']
                
                # Diversity score: max similarity to already selected
                if selected:
                    max_sim_to_selected = max(
                        self._cosine_similarity(candidate['embedding'], s['embedding'])
                        for s in selected
                    )
                    diversity = 1.0 - max_sim_to_selected
                else:
                    diversity = 1.0
                
                # MMR score: balance relevance and diversity
                mmr_score = (1 - diversity_lambda) * relevance + diversity_lambda * diversity
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_candidate = candidate
                    best_idx = idx
            
            if best_candidate is None:
                break  # No more candidates fit
            
            selected.append(best_candidate)
            used_tokens += best_candidate.get('token_count', self.count_tokens(best_candidate['example']['text']))
            candidates.pop(best_idx)
            
            if self.debug:
                print(f"  [DIVERSITY] Selected: score={best_candidate['score']:.3f}, "
                      f"ops={best_candidate['ops']}, tokens_used={used_tokens}")
        
        return selected
    
    def _rerank_for_output(self, selected_examples: list, query: str, query_ops: set) -> list:
        """Rerank selected examples for optimal output ordering
        
        Strategy: Put most relevant example LAST (closest to the question being asked)
        This follows the recency bias in transformer attention.
        """
        if len(selected_examples) <= 1:
            return selected_examples
        
        # Score for final ordering
        rerank_scores = []
        for ex in selected_examples:
            # Higher score = should come later (closer to question)
            op_match = self._operation_match_score(query_ops, ex['ops'])
            complexity = self._calculate_complexity_match(query, ex['example']['question'])
            
            # Combine for rerank score
            rerank_score = op_match * 0.7 + complexity * 0.3
            rerank_scores.append((ex, rerank_score))
        
        # Sort: lowest score first, highest score last
        rerank_scores.sort(key=lambda x: x[1])
        
        if self.debug:
            print(f"  [RERANK] Order: {[f'{s:.2f}' for _, s in rerank_scores]}")
        
        return [ex for ex, _ in rerank_scores]

    def _extract_final_question(self, text: str) -> str:
        """Extract the final question that needs to be preserved"""
        if "Q:" in text:
            text_parts = text.split("Q:")
            final_question_part = text_parts[-1].strip()
            
            if "A:" in final_question_part:
                final_question_text = final_question_part.split("A:")[0].strip()
            else:
                final_question_text = final_question_part.strip()
                
            return f"Q: {final_question_text}\nA:"
        else:
            return text[-100:]  # Fallback to last 100 chars

    def _is_final_question(self, question: str, query: str) -> bool:
        """Check if this question is the final query question"""
        return question.strip().lower() == query.strip().lower()

    def _calculate_complexity_match(self, query: str, example_question: str) -> float:
        """Calculate complexity matching score between query and example.
        
        Measures similarity along two axes:
          1. Math-indicator density (keywords like 'total', 'how many', etc.)
          2. Distinct-number count (counts actual numbers, not digit characters)
        """
        math_indicators = ['calculate', 'solve', 'find', 'how many', 'what is', 'total', 'difference']
        
        query_complexity = sum(1 for indicator in math_indicators if indicator in query.lower())
        example_complexity = sum(1 for indicator in math_indicators if indicator in example_question.lower())
        
        # Count distinct numbers (not individual digit characters)
        query_numbers = len(re.findall(r'\d+\.?\d*', query))
        example_numbers = len(re.findall(r'\d+\.?\d*', example_question))
        
        # Calculate complexity similarity
        if query_complexity == 0 and example_complexity == 0:
            complexity_sim = 1.0
        else:
            complexity_sim = 1.0 - abs(query_complexity - example_complexity) / max(query_complexity + example_complexity, 1)
        
        # Calculate numerical similarity
        if query_numbers == 0 and example_numbers == 0:
            numerical_sim = 1.0
        else:
            numerical_sim = 1.0 - abs(query_numbers - example_numbers) / max(query_numbers + example_numbers, 1)
        
        return (complexity_sim + numerical_sim) / 2.0
    
    def _extract_few_shot_examples(self, text: str) -> list[dict]:
        """Extract few-shot examples from prompt text"""
        examples = []
        
        # Look for Q: ... A: ... patterns
        parts = text.split("Q:")
        if len(parts) < 2:
            return examples
        
        for i, part in enumerate(parts[1:], 1):  # Skip first empty part
            if "A:" in part:
                qa_parts = part.split("A:", 1)
                if len(qa_parts) == 2:
                    question = qa_parts[0].strip()
                    answer = qa_parts[1].strip()
                    
                    # Find end of this example (before next Q: or end of text)
                    next_q_idx = answer.find("\n\nQ:")
                    if next_q_idx != -1:
                        answer = answer[:next_q_idx].strip()
                    
                    example_text = f"Q: {question}\nA: {answer}\n\n"
                    examples.append({
                        'text': example_text,
                        'question': question,
                        'answer': answer,
                        'original_index': i
                    })
        
        return examples
    
    def _smart_truncate_fallback(self, text: str, query: str, max_length: int) -> tuple[str, str, float, str]:
        """Fallback smart truncation method - selects sentences by importance
        while preserving original order to maintain prompt coherence."""
        sentences = self._split_sentences(text)
        
        if not sentences:
            # Fallback to simple truncation from head (preserve end with question)
            if self.tokenizer:
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                start_idx = max(0, len(tokens) - (max_length - 2))
                truncated_tokens = tokens[start_idx:]
                result = self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
            else:
                words = text.split()
                start_idx = max(0, len(words) - int(max_length * 0.75))
                truncated_words = words[start_idx:]
                result = " ".join(truncated_words)
            return result, "fallback_cut_from_head", 0.5, "Used simple head-cut truncation"
        
        # Score sentences by importance (returns sorted by score descending)
        scored_sentences = self._score_sentences(sentences, query)
        
        # Select sentences that fit within the budget
        selected_indices = set()
        total_tokens = 0
        total_score = 0
        
        for sentence, score in scored_sentences:
            sentence_tokens = self.count_tokens(sentence)
            original_idx = sentences.index(sentence)
            
            if total_tokens + sentence_tokens <= max_length:
                selected_indices.add(original_idx)
                total_tokens += sentence_tokens
                total_score += score
            elif not selected_indices:  # Must include at least one sentence
                words = sentence.split()
                truncated = []
                temp_tokens = 0
                for word in reversed(words):
                    word_tokens = self.count_tokens(word)
                    if temp_tokens + word_tokens <= max_length - 10:
                        truncated.insert(0, word)
                        temp_tokens += word_tokens
                    else:
                        break
                if truncated:
                    # Use the original index so ordering works
                    selected_indices.add(original_idx)
                    sentences[original_idx] = " ".join(truncated)
                    total_score = score * 0.7
                break
        
        if not selected_indices:
            return text[:200], "emergency_fallback", 0.2, "Emergency fallback used"
        
        # Reassemble in ORIGINAL order to preserve prompt structure
        selected = [sentences[i] for i in sorted(selected_indices)]
        
        importance_ratio = total_score / len(sentences) if sentences else 0.6
        method = "smart_relaxed_selection" if importance_ratio > 0.5 else "adaptive_relaxed_truncation"
        debug_info = f"Selected {len(selected)}/{len(sentences)} sentences (order-preserved)"
        
        return " ".join(selected), method, importance_ratio, debug_info

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences"""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _score_sentences(self, sentences: list[str], query: str) -> list[tuple[str, float]]:
        """Score sentences by relevance to query"""
        query_words = set(query.lower().split()) if query else set()
        
        scored = []
        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            
            # Basic scoring
            length_score = min(1.0, len(sentence.split()) / 20.0)  # Prefer moderate length
            
            if query_words:
                overlap = len(query_words.intersection(sentence_words))
                relevance_score = overlap / max(len(query_words), 1)
            else:
                relevance_score = 0.5
            
            # Position bonus (slight preference for earlier content)
            position_score = 1.0 - (sentences.index(sentence) / len(sentences)) * 0.2
            
            final_score = (length_score * 0.3 + relevance_score * 0.5 + position_score * 0.2)
            scored.append((sentence, final_score))
        
        # Sort by score (descending)
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

# Factory functions
def create_context_manager(strategy: str, tokenizer_name: str, max_length: int = 1024) -> ContextManager:
    """Create a context manager with specified strategy"""
    return ContextManager(tokenizer_name, strategy, max_length)

# Global cache
_context_cache = {}

def get_context_manager(strategy: str, tokenizer_name: str, max_length: int = 1024) -> ContextManager:
    """Get or create a context manager - supports 512, 1024, 2048, and 4096 context lengths for comparison"""
    key = f"{strategy}_{tokenizer_name}_{max_length}"
    if key not in _context_cache:
        _context_cache[key] = create_context_manager(strategy, tokenizer_name, max_length)
    return _context_cache[key]

def clear_cache():
    """Clear context manager cache (keeps precomputed embeddings on disk)"""
    global _context_cache, _example_cache
    _context_cache.clear()
    # Note: _example_cache stays in memory intentionally - it's reusable
    # The .npz file on disk persists across runs
    print("Context manager cache cleared") 