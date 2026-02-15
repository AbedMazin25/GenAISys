"""
LlamaIndex implementation of the GSM8K math solver.

Pipeline: LlamaIndex PromptTemplate → context manager → CustomLLM.complete() → CompletionResponse.

The LLM class extends LlamaIndex's CustomLLM base class so that
llm.complete() flows through the framework's callback system (just like
LangChain's LLM._call() does in the LangChain implementation).
Token tracking flows through LlamaIndex's CompletionResponse.additional_kwargs.
"""

import time
import re
from typing import Any, Generator

import batching_model_manager as model_manager
import context_manager
from few_shot_examples import create_demo_text, ANSWER_TRIGGER

# LlamaIndex imports
from llama_index.core.llms.custom import CustomLLM
from llama_index.core.base.llms.types import (
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core.prompts import PromptTemplate as LlamaPromptTemplate
from llama_index.core.bridge.pydantic import PrivateAttr

# Answer extraction patterns
ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

N_SHOT = 76  # Use all 76 examples (~5800 tokens) to test all context lengths
COT_FLAG = True


# ========== ANSWER EXTRACTION ==========

def extract_answer_from_output(completion):
    """Extract the numerical answer from GSM8k format answer"""
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip().replace(",", "")
        return match_str
    return INVALID_ANS


def clean_answer(model_pred):
    """Extract numerical answer from model prediction.
    
    Uses the LAST "the answer is" occurrence so that reasoning-chain outputs
    (which may contain the trigger in intermediate steps) are handled correctly.
    """
    model_pred = model_pred.lower()
    preds = model_pred.split(ANSWER_TRIGGER.lower())
    answer_flag = len(preds) > 1
    
    pred = preds[-1]
    pred = pred.replace(",", "")
    pred = [s for s in re.findall(r"-?\d+\.?\d*", pred)]
    
    if len(pred) == 0:
        return INVALID_ANS
    
    pred = pred[0] if answer_flag else pred[-1]
    if pred[-1] == ".":
        pred = pred[:-1]
    return pred


def _numerically_equal(a: str, b: str) -> bool:
    """Compare two number strings numerically (e.g. '39.0' == '39')."""
    try:
        return float(a) == float(b)
    except (ValueError, OverflowError):
        return a == b


def clean_response(response: str) -> str:
    """Clean up model response - stop at first complete answer"""
    separators = ["\n\nQ:", "\nQ:", "\n\nQuestion:", "\nQuestion:"]
    for sep in separators:
        if sep in response:
            response = response.split(sep)[0]
            break
    
    lines = response.split('\n')
    clean_lines = []
    for line in lines:
        clean_lines.append(line)
        if any(trigger in line.lower() for trigger in ["the answer is", "answer is"]):
            break
    
    return '\n'.join(clean_lines).strip()


def evaluate_math_answer(model_response: str, ground_truth: str) -> dict:
    """Evaluate correctness of a math answer"""
    model_answer = clean_answer(model_response)
    gt_answer = extract_answer_from_output(ground_truth)
    is_answer_correct = (model_answer != INVALID_ANS and 
                        gt_answer != INVALID_ANS and 
                        _numerically_equal(model_answer, gt_answer))
    return {
        'model_answer': model_answer,
        'ground_truth': gt_answer,
        'is_correct': is_answer_correct,
        'model_response': model_response,
        'extractable': model_answer != INVALID_ANS,
        'gt_extractable': gt_answer != INVALID_ANS
    }


# ========== LLAMAINDEX CUSTOM LLM ==========

class LlamaIndexMathLLM(CustomLLM):
    """
    LlamaIndex CustomLLM that wraps our batching_model_manager.
    
    Extends LlamaIndex's CustomLLM base class so that llm.complete() flows
    through the framework's callback system and returns a proper
    CompletionResponse — unlike a plain wrapper class that would bypass
    the framework entirely.
    
    This is the LlamaIndex equivalent of LangChain's LLM._call() pattern
    used in math_tutor_langchain.py.
    """
    
    # Pydantic fields (CustomLLM is a Pydantic model)
    model_name: str = "metamath-mistral"
    temperature: float = 0.3
    max_tokens: int = 1024
    
    # Private attributes (not serialized by Pydantic)
    _model_mgr: Any = PrivateAttr(default=None)
    _total_tokens: int = PrivateAttr(default=0)
    
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._model_mgr = model_manager.get_model_manager(self.model_name)
        self._total_tokens = 0
    
    @property
    def metadata(self) -> LLMMetadata:
        """LlamaIndex metadata — exposes context window and model info to the framework."""
        model_info = model_manager.get_model_info(self.model_name)
        return LLMMetadata(
            context_window=model_info.get("context_window", 8192),
            num_output=self.max_tokens,
            model_name=self.model_name,
        )
    
    def _run_inference(self, prompt: str, **kwargs: Any) -> tuple:
        """Core generation logic — shared by complete() and stream_complete()."""
        max_tok = kwargs.get("max_tokens", self.max_tokens)
        temp = kwargs.get("temperature", self.temperature)
        top_p = kwargs.get("top_p", 0.9)
        
        response_text, tokens = self._model_mgr.generate_with_token_info(
            prompt, max_tokens=max_tok, temperature=temp, top_p=top_p
        )
        self._total_tokens += tokens
        return response_text, tokens
    
    @llm_completion_callback()
    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        """
        LlamaIndex complete() — called by the framework's callback system.
        
        This is the abstract method from BaseLLM that CustomLLM subclasses
        must implement.  The @llm_completion_callback decorator fires
        LlamaIndex's event hooks (CBEventType.LLM) around the call.
        """
        response_text, tokens = self._run_inference(prompt, **kwargs)
        return CompletionResponse(
            text=response_text,
            additional_kwargs={"tokens_generated": tokens}
        )
    
    @llm_completion_callback()
    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseGen:
        """Non-streaming fallback — yields the full response as a single chunk."""
        response_text, tokens = self._run_inference(prompt, **kwargs)
        yield CompletionResponse(
            text=response_text,
            delta=response_text,
            additional_kwargs={"tokens_generated": tokens}
        )
    
    def get_total_tokens(self) -> int:
        return self._total_tokens
    
    def reset_token_count(self):
        self._total_tokens = 0


# ========== MAIN SOLVE FUNCTIONS ==========

def solve_math_problem_llamaindex_with_token_info(
    question: str, 
    model_name: str = "metamath-mistral", 
    use_few_shot: bool = True,
    context_strategy: str = "fixed",
    max_length: int = 1024
) -> tuple[str, int]:
    """
    Solve a math problem using LlamaIndex framework.
    
    Pipeline:
      1. Build few-shot prompt using LlamaIndex PromptTemplate
      2. Process through context manager (fixed/adaptive)
      3. Generate via LlamaIndex CustomLLM.complete() → CompletionResponse
      4. Extract and clean answer
    
    Args:
        question: Math problem to solve
        model_name: Model to use
        use_few_shot: Whether to use few-shot examples
        context_strategy: "fixed" or "adaptive"
        max_length: Maximum context length in tokens
    
    Returns:
        Tuple of (response, actual_token_count)
    """
    try:
        # Create LlamaIndex CustomLLM (may start vLLM server - not timed)
        llm = LlamaIndexMathLLM(model_name=model_name)
        llm.reset_token_count()
        
        # Build prompt using LlamaIndex PromptTemplate
        prompt_template = LlamaPromptTemplate(
            template="{few_shot_examples}Q: {question}\nA:",
        )
        
        if use_few_shot and N_SHOT > 0:
            few_shot_text = create_demo_text(N_SHOT, COT_FLAG)
        else:
            few_shot_text = ""
        
        # Format using LlamaIndex PromptTemplate
        raw_prompt = prompt_template.format(
            few_shot_examples=few_shot_text,
            question=question
        )
        
        # Process through context manager (fixed or adaptive)
        model_info = model_manager.get_model_info(model_name)
        tokenizer_name = model_info["model_id"]
        ctx_mgr = context_manager.get_context_manager(context_strategy, tokenizer_name, max_length)
        processed = ctx_mgr.process_context(raw_prompt, question)
        
        print(f"[LlamaIndex] {model_name} | ctx={max_length} | {context_strategy} | "
              f"prompt={processed.processed_length} tokens | truncated={processed.truncated}")
        
        # Timer starts HERE - only measures inference, not setup/server start
        start_time = time.time()
        
        # Generate via LlamaIndex CustomLLM.complete() → CompletionResponse
        # This goes through LlamaIndex's @llm_completion_callback decorator,
        # firing the framework's event system (CBEventType.LLM).
        completion = llm.complete(
            processed.processed_text,
            max_tokens=1024,
            temperature=0.3,
            top_p=0.9
        )
        
        # Extract response from LlamaIndex CompletionResponse
        response = clean_response(completion.text)
        actual_tokens = llm.get_total_tokens()
        
        end_time = time.time()
        print(f"[LlamaIndex] Done in {end_time - start_time:.2f}s ({actual_tokens} tokens)")
        
        return response, actual_tokens
        
    except Exception as e:
        print(f"[LlamaIndex] Error: {e}")
        error_msg = f"Error: {str(e)}"
        return error_msg, 0


# Convenience aliases
def solve_math_problem_llamaindex(question, model_name="metamath-mistral", use_few_shot=True, 
                                   context_strategy="fixed", max_length=1024):
    response, _ = solve_math_problem_llamaindex_with_token_info(
        question, model_name, use_few_shot, context_strategy, max_length
    )
    return response


def clear_caches():
    model_manager.clear_cache()
    context_manager.clear_cache()

def get_available_models():
    return model_manager.get_available_models()
