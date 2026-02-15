"""
LangChain 4-step reasoning implementation of the GSM8K math solver.

Uses LangChain LCEL with RunnablePassthrough.assign() to build a 4-step chain:
  1. Parse  — extract key facts
  2. Plan   — outline calculation steps
  3. Execute — solve with few-shot CoT examples (context-managed)
  4. Verify  — extract final numerical answer

Each step carries forward all previous outputs via the accumulated dict.
Makes 4 LLM calls per question vs 1 for the other frameworks, trading
latency for structured reasoning.
"""

import time
import re
from typing import Any, List, Mapping, Optional

import batching_model_manager as model_manager
import context_manager
from few_shot_examples import create_demo_text, ANSWER_TRIGGER

# LangChain imports (langchain >= 1.0 moved these to langchain_core)
from langchain_core.language_models.llms import LLM
from langchain_core.prompts import PromptTemplate
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.runnables import RunnablePassthrough

# Answer extraction
ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"
N_SHOT = 76
COT_FLAG = True


# ========== ANSWER EXTRACTION ==========

def extract_answer_from_output(completion):
    match = ANS_RE.search(completion)
    if match:
        return match.group(1).strip().replace(",", "")
    return INVALID_ANS


def clean_answer(model_pred):
    """Extract numerical answer - handles multiple formats:
    - 'The answer is 18'
    - 'The answer is $18' 
    - '<<9*2=18>>18'
    - 'Final answer: Janet makes $18'
    """
    model_pred_lower = model_pred.lower()
    
    # Try "the answer is X" first
    preds = model_pred_lower.split(ANSWER_TRIGGER.lower())
    if len(preds) > 1:
        after_trigger = preds[-1].replace(",", "").replace("$", "")
        nums = re.findall(r"-?\d+\.?\d*", after_trigger)
        if nums:
            result = nums[0]
            return result[:-1] if result.endswith(".") else result
    
    # Try GSM8K format <<calculation=answer>>
    gsm_matches = re.findall(r"<<[^>]*=(-?\d+\.?\d*)>>", model_pred)
    if gsm_matches:
        result = gsm_matches[-1]
        return result[:-1] if result.endswith(".") else result
    
    # Try "final answer" patterns
    final_match = re.search(r"final\s*answer[:\s]*\$?(-?\d+\.?\d*)", model_pred_lower)
    if final_match:
        result = final_match.group(1)
        return result[:-1] if result.endswith(".") else result
    
    # Fallback: last number in the response
    all_nums = re.findall(r"-?\d+\.?\d*", model_pred.replace(",", ""))
    if all_nums:
        result = all_nums[-1]
        return result[:-1] if result.endswith(".") else result
    
    return INVALID_ANS


def _numerically_equal(a: str, b: str) -> bool:
    """Compare two number strings numerically (e.g. '39.0' == '39')."""
    try:
        return float(a) == float(b)
    except (ValueError, OverflowError):
        return a == b


def evaluate_math_answer(model_response: str, ground_truth: str) -> dict:
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

class ReasoningLLM(LLM):
    """LangChain LLM wrapper for the reasoning chain steps.
    
    Does NOT apply clean_response() - intermediate steps must not be truncated.
    Only the final output gets cleaned.
    """
    
    model_name: str = "metamath-mistral"
    temperature: float = 0.3
    max_tokens: int = 300
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, '_total_tokens', 0)
    
    @property
    def _llm_type(self) -> str:
        return "reasoning_chain_llm"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs) -> str:
        try:
            model_mgr = model_manager.get_model_manager(self.model_name)
            response, tokens = model_mgr.generate_with_token_info(
                prompt, max_tokens=self.max_tokens, temperature=self.temperature, top_p=0.9
            )
            current = object.__getattribute__(self, '_total_tokens')
            object.__setattr__(self, '_total_tokens', current + tokens)
            # NO clean_response() here - intermediate steps must stay complete
            return response
        except Exception as e:
            print(f"[ReasoningLLM] Error: {e}")
            return f"Error: {str(e)}"
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model_name": self.model_name, "temperature": self.temperature, "max_tokens": self.max_tokens}
    
    def get_total_tokens(self) -> int:
        return object.__getattribute__(self, '_total_tokens')
    
    def reset_token_count(self):
        object.__setattr__(self, '_total_tokens', 0)


# ========== SOLVE FUNCTION ==========

def solve_math_problem_langchain_reasoning_with_token_info(
    question: str,
    model_name: str = "metamath-mistral",
    use_few_shot: bool = True,
    context_strategy: str = "fixed",
    max_length: int = 1024
) -> tuple:
    """
    Solve a math problem using LangChain LCEL 4-step reasoning chain.
    
    Pipeline (LCEL with RunnablePassthrough.assign):
      1. Parse: Extract facts          (PromptTemplate | LLM)
      2. Plan: Create calculation steps (PromptTemplate | LLM)
      3. Execute: Solve with few-shot   (PromptTemplate | LLM)
      4. Verify: Extract final answer   (PromptTemplate | LLM)
    
    Each step adds its output to the running dict via RunnablePassthrough.assign(),
    so subsequent steps can reference all previous outputs.
    
    Args:
        question: Math problem to solve
        model_name: Model to use
        use_few_shot: Whether to use few-shot examples in execute step
        context_strategy: "fixed" or "adaptive" 
        max_length: Max context length for the execute step
    
    Returns:
        Tuple of (response, total_tokens)
    """
    try:
        # Create LLMs for each step (different token budgets)
        llm_parse = ReasoningLLM(model_name=model_name, temperature=0.3, max_tokens=300)
        llm_plan = ReasoningLLM(model_name=model_name, temperature=0.3, max_tokens=300)
        llm_execute = ReasoningLLM(model_name=model_name, temperature=0.3, max_tokens=600)
        llm_verify = ReasoningLLM(model_name=model_name, temperature=0.3, max_tokens=300)
        
        all_llms = [llm_parse, llm_plan, llm_execute, llm_verify]
        for llm in all_llms:
            llm.reset_token_count()
        
        # Build few-shot context for the execute step using context manager.
        # Design note: the execute step uses half the context budget because:
        #   1. The execute prompt also embeds parsed_info + plan from prior steps.
        #   2. Four sequential LLM calls each consume context window; reserving space
        #      prevents prompt overflow on smaller context lengths.
        # This intentionally gives the reasoning chain fewer few-shot examples than
        # the single-call frameworks, which is a trade-off for multi-step reasoning.
        execute_ctx_budget = max(256, max_length // 2)
        
        if use_few_shot and N_SHOT > 0:
            demo = create_demo_text(N_SHOT, COT_FLAG)
            raw_execute_prompt = f"{demo}Q: {question}\nA:"
            
            model_info = model_manager.get_model_info(model_name)
            tokenizer_name = model_info["model_id"]
            ctx_mgr = context_manager.get_context_manager(context_strategy, tokenizer_name, execute_ctx_budget)
            processed = ctx_mgr.process_context(raw_execute_prompt, question)
            
            # Extract just the few-shot part (without the final Q: ... A:)
            few_shot_context = processed.processed_text
            last_q_idx = few_shot_context.rfind(f"Q: {question[:30]}")
            if last_q_idx > 0:
                few_shot_context = few_shot_context[:last_q_idx].rstrip()
        else:
            few_shot_context = ""
        
        # Step 1: Parse — extract key facts
        parse_template = PromptTemplate(
            input_variables=["question"],
            template="Analyze this math problem and extract ALL key facts.\n\nProblem: {question}\n\nFacts:\n-"
        )
        parse_step = parse_template | llm_parse
        
        # Step 2: Plan — outline calculation steps
        plan_template = PromptTemplate(
            input_variables=["question", "parsed_info"],
            template="PROBLEM: {question}\n\nFACTS:\n{parsed_info}\n\nCreate a step-by-step calculation plan using ONLY the facts above.\n\nPlan:\nStep 1:"
        )
        plan_step = plan_template | llm_plan
        
        # Step 3: Execute — solve with few-shot examples (context-managed)
        execute_template = PromptTemplate(
            input_variables=["question", "parsed_info", "plan"],
            template=f"""{few_shot_context}

Solve this problem by following the plan EXACTLY:

PROBLEM: {{question}}

FACTS: {{parsed_info}}

PLAN: {{plan}}

Follow the plan step by step. Use ONLY the numbers from the facts.

A:"""
        )
        execute_step = execute_template | llm_execute
        
        # Step 4: Verify — extract final numerical answer
        verify_template = PromptTemplate(
            input_variables=["question", "parsed_info", "plan", "solution"],
            template="PROBLEM: {question}\n\nSOLUTION: {solution}\n\nWhat is the final numerical answer? Reply with ONLY the number, nothing else.\n\nThe answer is"
        )
        verify_step = verify_template | llm_verify
        
        # Build LCEL chain: each step adds its output to the running dict
        # via RunnablePassthrough.assign(), so later steps see earlier outputs.
        # This is the modern LCEL replacement for the deprecated SequentialChain.
        chain = (
            RunnablePassthrough.assign(parsed_info=parse_step)
            | RunnablePassthrough.assign(plan=plan_step)
            | RunnablePassthrough.assign(solution=execute_step)
            | RunnablePassthrough.assign(verification=verify_step)
        )
        
        print(f"[LC-Reasoning] {model_name} | ctx={max_length} | {context_strategy}")
        
        # Pre-warm: ensure vLLM server is ready before timing
        model_manager.get_model_manager(model_name)
        
        # Timer starts here - only measures inference
        start_time = time.time()
        
        # Run the LCEL chain — input is {"question": ...}, output accumulates all keys
        result = chain.invoke({"question": question})
        
        end_time = time.time()
        total_tokens = sum(llm.get_total_tokens() for llm in all_llms)
        
        # Build final response with "The answer is X" at the END for extraction
        final_response = (
            f"STEP 1 - FACTS:\n{result['parsed_info'].strip()}\n\n"
            f"STEP 2 - PLAN:\n{result['plan'].strip()}\n\n"
            f"STEP 3 - SOLUTION:\n{result['solution'].strip()}\n\n"
            f"STEP 4 - VERIFICATION:\n{ANSWER_TRIGGER} {result['verification'].strip()}"
        )
        
        print(f"[LC-Reasoning] Done in {end_time - start_time:.2f}s ({total_tokens} tokens)")
        
        return final_response, total_tokens
        
    except Exception as e:
        print(f"[LC-Reasoning] Error: {e}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}", 0


# Convenience aliases
def solve_math_problem_langchain_reasoning(question, model_name="metamath-mistral", 
                                            use_few_shot=True, context_strategy="fixed", max_length=1024):
    response, _ = solve_math_problem_langchain_reasoning_with_token_info(
        question, model_name, use_few_shot, context_strategy, max_length
    )
    return response


def clear_caches():
    model_manager.clear_cache()
    context_manager.clear_cache()

def get_available_models():
    return model_manager.get_available_models()
