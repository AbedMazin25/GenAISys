"""
LangChain implementation of the GSM8K math solver.

Pipeline: few-shot prompt → context manager → LangChain LCEL (PromptTemplate | LLM).
Uses a custom LLM wrapper that extends langchain_core's LLM to call vLLM.
"""

import time
import re
from typing import Any, List, Mapping, Optional

import batching_model_manager as model_manager
import context_manager
from few_shot_examples import create_demo_text, ANSWER_TRIGGER

# LangChain imports (langchain >= 1.0 moved these to langchain_core)
from langchain_core.prompts import PromptTemplate as LangChainPromptTemplate
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.runnables import RunnableLambda

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


# ========== LANGCHAIN LLM WRAPPER ==========

class LangChainMathLLM(LLM):
    """
    Custom LangChain LLM that wraps our batching_model_manager.
    
    This is the PROPER way to integrate a local HuggingFace model with LangChain.
    It implements LangChain's LLM base class so it works with PromptTemplate and LLMChain.
    """
    
    model_name: str = "metamath-mistral"
    temperature: float = 0.3
    max_tokens: int = 1024
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, '_total_tokens', 0)
    
    @property
    def _llm_type(self) -> str:
        return "langchain_math_local"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Core LangChain _call method - invoked by LLMChain."""
        try:
            model_mgr = model_manager.get_model_manager(self.model_name)
            
            response, tokens = model_mgr.generate_with_token_info(
                prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=0.9
            )
            
            current = object.__getattribute__(self, '_total_tokens')
            object.__setattr__(self, '_total_tokens', current + tokens)
            
            return response
            
        except Exception as e:
            print(f"[LangChain LLM] Error: {e}")
            return f"Error: {str(e)}"
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model_name": self.model_name, "temperature": self.temperature, "max_tokens": self.max_tokens}
    
    def get_total_tokens(self) -> int:
        return object.__getattribute__(self, '_total_tokens')
    
    def reset_token_count(self):
        object.__setattr__(self, '_total_tokens', 0)


# ========== MAIN SOLVE FUNCTIONS ==========

def solve_math_problem_langchain_with_token_info(
    question: str,
    model_name: str = "metamath-mistral",
    use_few_shot: bool = True,
    context_strategy: str = "fixed",
    max_length: int = 1024
) -> tuple[str, int]:
    """
    Solve a math problem using LangChain framework.
    
    Pipeline:
      1. Build few-shot prompt using LangChain PromptTemplate
      2. Process through context manager (fixed/adaptive)
      3. Generate via LangChain LLM (prompt | llm) pipeline
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
        # Create LangChain LLM (may start vLLM server - not timed)
        llm = LangChainMathLLM(
            model_name=model_name,
            temperature=0.3,
            max_tokens=1024
        )
        llm.reset_token_count()
        
        # Build few-shot examples
        if use_few_shot and N_SHOT > 0:
            few_shot_text = create_demo_text(N_SHOT, COT_FLAG)
        else:
            few_shot_text = ""
        
        # LangChain PromptTemplate formats the few-shot CoT prompt
        prompt_template = LangChainPromptTemplate(
            input_variables=["few_shot_examples", "question"],
            template="{few_shot_examples}Q: {question}\nA:"
        )
        
        # Context manager wrapped as a LangChain Runnable so the full pipeline
        # is: PromptTemplate → context truncation/selection → LLM
        model_info = model_manager.get_model_info(model_name)
        tokenizer_name = model_info["model_id"]
        ctx_mgr = context_manager.get_context_manager(context_strategy, tokenizer_name, max_length)
        
        # Store metadata from context processing for logging
        ctx_meta = {}
        
        def apply_context_management(prompt_value) -> str:
            """LangChain Runnable step: apply fixed/adaptive context truncation."""
            raw_text = prompt_value.to_string() if hasattr(prompt_value, 'to_string') else str(prompt_value)
            processed = ctx_mgr.process_context(raw_text, question)
            ctx_meta['length'] = processed.processed_length
            ctx_meta['truncated'] = processed.truncated
            return processed.processed_text
        
        # Timer starts HERE - only measures inference, not setup/server start
        start_time = time.time()
        
        # Full LCEL chain: PromptTemplate → ContextManager → LLM
        chain = prompt_template | RunnableLambda(apply_context_management) | llm
        result = chain.invoke({
            "few_shot_examples": few_shot_text,
            "question": question
        })
        
        print(f"[LangChain] {model_name} | ctx={max_length} | {context_strategy} | "
              f"prompt={ctx_meta.get('length', '?')} tokens | truncated={ctx_meta.get('truncated', '?')}")
        
        # Extract response - chain.invoke returns the LLM output string
        response = clean_response(result)
        actual_tokens = llm.get_total_tokens()
        
        end_time = time.time()
        print(f"[LangChain] Done in {end_time - start_time:.2f}s ({actual_tokens} tokens)")
        
        return response, actual_tokens
        
    except Exception as e:
        print(f"[LangChain] Error: {e}")
        error_msg = f"Error: {str(e)}"
        return error_msg, 0


# Convenience aliases
def solve_math_problem_langchain(question, model_name="metamath-mistral", use_few_shot=True,
                                  context_strategy="fixed", max_length=1024):
    response, _ = solve_math_problem_langchain_with_token_info(
        question, model_name, use_few_shot, context_strategy, max_length
    )
    return response


def clear_caches():
    model_manager.clear_cache()
    context_manager.clear_cache()

def get_available_models():
    return model_manager.get_available_models()
