import time
import re
import batching_model_manager as model_manager
import context_manager
from few_shot_examples import create_demo_text, ANSWER_TRIGGER

# Answer extraction patterns
ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

N_SHOT = 76  # Use all 76 examples (~5800 tokens) to test all context lengths
COT_FLAG = True

def build_prompt(input_text, n_shot=8, cot_flag=True):
    """Build optimized prompt with few-shot examples"""
    demo = create_demo_text(n_shot, cot_flag)
    input_text_prompt = demo + "Q: " + input_text + "\n" + "A:"
    return input_text_prompt

def extract_answer_from_output(completion):
    """Extract the numerical answer from GSM8k format answer"""
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS

def _numerically_equal(a: str, b: str) -> bool:
    """Compare two number strings numerically (e.g. '39.0' == '39')."""
    try:
        return float(a) == float(b)
    except (ValueError, OverflowError):
        return a == b

def is_correct(model_answer, answer):
    """Check if the model answer matches the ground truth answer"""
    gt_answer = extract_answer_from_output(answer)
    assert gt_answer != INVALID_ANS
    return _numerically_equal(model_answer, gt_answer)

def clean_answer(model_pred):
    """Improved answer extraction from model predictions.
    
    When the answer trigger ("The answer is") is found, extracts the first
    number after the LAST trigger occurrence. This correctly handles multi-step
    outputs (e.g. reasoning chains) where intermediate steps may also contain
    the trigger phrase.
    """
    model_pred = model_pred.lower()
    preds = model_pred.split(ANSWER_TRIGGER.lower())
    answer_flag = len(preds) > 1
    
    if answer_flag:
        # Pick text after the LAST answer trigger (handles reasoning chains
        # where intermediate steps may also contain "the answer is")
        pred = preds[-1]
    else:
        # No trigger found — use last segment
        pred = preds[-1]

    pred = pred.replace(",", "")
    pred = [s for s in re.findall(r"-?\d+\.?\d*", pred)]

    if len(pred) == 0:
        return INVALID_ANS

    # First number after the (last) trigger, or last number if no trigger
    pred = pred[0] if answer_flag else pred[-1]

    # Strip trailing period (e.g. "42." → "42")
    if pred[-1] == ".":
        pred = pred[:-1]

    return pred

def solve_math_problem(question: str, model_name: str = "metamath-mistral", context_strategy: str = "fixed", use_few_shot: bool = True, max_length: int = 1024) -> str:
    """
    Solve a math problem using direct LLM calls with improved prompting
    
    Args:
        question: Math problem to solve
        model_name: Model to use for generation
        context_strategy: "fixed" or "adaptive" context processing
        use_few_shot: Whether to use few-shot examples
        max_length: Maximum context length for context manager
    """
    start_time = time.time()
    
    try:
        # Get model manager
        model_mgr = model_manager.get_model_manager(model_name)
        
        # Get context manager with specified max_length
        model_info = model_manager.get_model_info(model_name)
        tokenizer_name = model_info["model_id"]
        context_mgr = context_manager.get_context_manager(context_strategy, tokenizer_name, max_length)
        
        # Create improved prompt
        if use_few_shot and N_SHOT > 0:
            raw_prompt = build_prompt(question, N_SHOT, COT_FLAG)
        else:
            # Fallback to simple prompt
            raw_prompt = f"Q: {question}\nA:"
        
        # Process context
        processed_context = context_mgr.process_context(raw_prompt, question)
        
        print(f"Solving math problem with {model_name} (prompt: {processed_context.processed_length} tokens)")
        
        # Generate response
        response = model_mgr.generate(
            processed_context.processed_text, 
            max_tokens=1024,
            temperature=0.3,
            top_p=0.9
        )
        
        # Clean up response - stop at first complete answer
        response = clean_response(response)
        
        end_time = time.time()
        print(f"Math solution generated in {end_time - start_time:.2f}s")
        
        return response
        
    except Exception as e:
        print(f"Error solving math problem: {e}")
        return f"Sorry, I encountered an error: {str(e)}"

def clean_response(response: str) -> str:
    """Clean up model response to prevent runaway generation.
    
    Stops at:
      1. A new question boundary ("Q:", "Question:")
      2. The first blank line AFTER an answer trigger ("the answer is")
      3. End of text
    """
    # Split by common separators that indicate new questions
    separators = ["\n\nQ:", "\nQ:", "\n\nQuestion:", "\nQuestion:"]
    
    for sep in separators:
        if sep in response:
            response = response.split(sep)[0]
            break
    
    # Keep lines up to (and including) the answer, then stop at next blank line
    lines = response.split('\n')
    clean_lines = []
    found_answer = False
    
    for line in lines:
        clean_lines.append(line)
        if any(trigger in line.lower() for trigger in ["the answer is", "answer is"]):
            found_answer = True
            # Don't break here — allow the rest of this line and possible
            # continuation, but stop at the next blank line.
        elif found_answer and not line.strip():
            break
    
    return '\n'.join(clean_lines).strip()

def solve_math_problem_with_token_info(question: str, model_name: str = "metamath-mistral", context_strategy: str = "fixed", use_few_shot: bool = True, max_length: int = 1024) -> tuple[str, int]:
    """
    Solve a math problem and return both response and actual token count
    
    Args:
        question: Math problem to solve
        model_name: Model to use for generation
        context_strategy: "fixed" or "adaptive" context processing
        use_few_shot: Whether to use few-shot examples
        max_length: Maximum context length for context manager
    
    Returns:
        Tuple of (response, actual_token_count)
    """
    start_time = time.time()
    
    try:
        # Get model manager
        model_mgr = model_manager.get_model_manager(model_name)
        
        # Get context manager with specified max_length
        model_info = model_manager.get_model_info(model_name)
        tokenizer_name = model_info["model_id"]
        context_mgr = context_manager.get_context_manager(context_strategy, tokenizer_name, max_length)
        
        # Create improved prompt
        if use_few_shot and N_SHOT > 0:
            raw_prompt = build_prompt(question, N_SHOT, COT_FLAG)
        else:
            # Fallback to simple prompt
            raw_prompt = f"Q: {question}\nA:"
        
        # Process context
        processed_context = context_mgr.process_context(raw_prompt, question)
        
        print(f"Solving math problem with {model_name} (prompt: {processed_context.processed_length} tokens)")
        
        # Generate response with token info
        response, actual_tokens = model_mgr.generate_with_token_info(
            processed_context.processed_text, 
            max_tokens=1024,
            temperature=0.3,
            top_p=0.9
        )
        
        # Clean up response - stop at first complete answer
        response = clean_response(response)
        
        end_time = time.time()
        print(f"Math solution generated in {end_time - start_time:.2f}s ({actual_tokens} tokens)")
        
        return response, actual_tokens
        
    except Exception as e:
        print(f"Error solving math problem: {e}")
        error_msg = f"Sorry, I encountered an error: {str(e)}"
        try:
            model_mgr = model_manager.get_model_manager(model_name)
            error_tokens = model_mgr.count_tokens(error_msg)
        except Exception:
            error_tokens = len(error_msg.split())
        return error_msg, error_tokens

def evaluate_math_answer(model_response: str, ground_truth: str) -> dict:
    """
    Evaluate the correctness of a math answer
    
    Args:
        model_response: The model's response to the math problem
        ground_truth: The ground truth answer in GSM8k format
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Extract numerical answer from model response using improved method
    model_answer = clean_answer(model_response)
    
    # Extract ground truth answer
    gt_answer = extract_answer_from_output(ground_truth)
    
    # Check correctness (numerically — "39.0" == "39")
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

def get_model_stats(model_name: str):
    """Get performance statistics for a model"""
    try:
        model_mgr = model_manager.get_model_manager(model_name)
        return model_mgr.get_stats()
    except Exception:
        return {"error": "Model not loaded"}

def reset_model_stats(model_name: str):
    """Reset performance statistics for a model"""
    try:
        model_mgr = model_manager.get_model_manager(model_name)
        model_mgr.reset_stats()
        return {"status": "Stats reset"}
    except Exception:
        return {"error": "Model not loaded"}

def clear_caches():
    """Clear all caches"""
    model_manager.clear_cache()
    context_manager.clear_cache()

def get_available_models():
    """Get list of available models"""
    return model_manager.get_available_models()

# Main interactive loop
if __name__ == "__main__":
    print("🔢 Math Tutor AI - Enhanced Direct LLM Math Reasoning")
    print("Available models:", get_available_models())
    print("Available context strategies: fixed, adaptive")
    print(f"Using {N_SHOT}-shot prompting with Chain-of-Thought reasoning")
    print("Type 'exit' to quit\n")
    
    while True:
        user_input = input("Enter a math problem: ").strip()
        
        if user_input.lower() in ["exit", "quit", "q"]:
            print("Goodbye! 🔢")
            clear_caches()
            break
        
        if not user_input:
            continue
            
        # Use default model and strategy
        response = solve_math_problem(user_input, "mistral-7b-quantized", "fixed")
        print(f"\n🤖 {response}\n")
        
        # Extract and show the numerical answer using improved method
        numerical_answer = clean_answer(response)
        if numerical_answer != INVALID_ANS:
            print(f"📊 Extracted Answer: {numerical_answer}")
        else:
            print("⚠️ Could not extract numerical answer")
        
        # Show performance stats
        stats = get_model_stats("mistral-7b-quantized")
        if "avg_latency" in stats:
            print(f"📊 Avg latency: {stats['avg_latency']:.2f}s | "
                  f"Total requests: {stats['total_requests']} | "
                  f"Tokens/sec: {stats.get('tokens_per_second', 0):.1f}") 