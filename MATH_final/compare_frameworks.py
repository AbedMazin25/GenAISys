"""Quick side-by-side demo of all three framework implementations on a single problem."""

import time
import sys
import traceback

# Import all three versions
import math_tutor as original_tutor
import math_tutor_llamaindex as llamaindex_tutor
import math_tutor_langchain as langchain_tutor

def test_framework(framework_name, solve_function, question, model_name="phi2"):
    """Test a specific framework implementation"""
    print(f"\n{'='*60}")
    print(f"🧪 Testing {framework_name}")
    print(f"{'='*60}")
    
    try:
        start_time = time.time()
        # Use keyword arguments — parameter ordering differs between frameworks
        response = solve_function(
            question, model_name, 
            context_strategy="fixed", use_few_shot=True, max_length=1024
        )
        end_time = time.time()
        
        print(f"✅ {framework_name} Response:")
        print(f"   {response}")
        print(f"⏱️  Time taken: {end_time - start_time:.2f}s")
        
        # Extract numerical answer
        if framework_name == "Original (Direct LLM)":
            numerical_answer = original_tutor.clean_answer(response)
        elif framework_name == "LlamaIndex":
            numerical_answer = llamaindex_tutor.clean_answer(response)
        else:  # LangChain
            numerical_answer = langchain_tutor.clean_answer(response)
        
        print(f"🔢 Extracted Answer: {numerical_answer}")
        return True, numerical_answer, end_time - start_time
        
    except Exception as e:
        print(f"❌ {framework_name} Error: {str(e)}")
        print(f"Stack trace: {traceback.format_exc()}")
        return False, None, None

def main():
    """Main comparison function"""
    print("🔢 Math Tutor Framework Comparison")
    print("=" * 80)
    
    # Test question
    test_question = "Sarah has 45 stickers. She gives 12 stickers to her friend and buys 8 more stickers. How many stickers does she have now?"
    expected_answer = "41"  # 45 - 12 + 8 = 41
    
    print(f"📝 Test Question: {test_question}")
    print(f"🎯 Expected Answer: {expected_answer}")
    
    # Test configurations
    frameworks = [
        ("Original (Direct LLM)", original_tutor.solve_math_problem),
        ("LlamaIndex", llamaindex_tutor.solve_math_problem_llamaindex),
        ("LangChain", langchain_tutor.solve_math_problem_langchain)
    ]
    
    results = {}
    
    # Test each framework
    for framework_name, solve_function in frameworks:
        success, answer, duration = test_framework(framework_name, solve_function, test_question)
        try:
            correct = answer is not None and float(answer) == float(expected_answer)
        except (ValueError, TypeError):
            correct = answer == expected_answer if answer else False
        results[framework_name] = {
            'success': success,
            'answer': answer,
            'duration': duration,
            'correct': correct
        }
    
    # Summary
    print(f"\n{'='*80}")
    print("📊 COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    print(f"{'Framework':<25} {'Success':<10} {'Answer':<10} {'Correct':<10} {'Time (s)':<10}")
    print("-" * 80)
    
    for framework_name, result in results.items():
        success_icon = "✅" if result['success'] else "❌"
        correct_icon = "✅" if result['correct'] else "❌"
        answer_str = str(result['answer']) if result['answer'] else "N/A"
        duration_str = f"{result['duration']:.2f}" if result['duration'] else "N/A"
        
        print(f"{framework_name:<25} {success_icon:<10} {answer_str:<10} {correct_icon:<10} {duration_str:<10}")
    
    # Framework-specific features
    print(f"\n{'='*80}")
    print("🔍 FRAMEWORK FEATURES")
    print(f"{'='*80}")
    
    print("\n🔧 Original (Direct LLM):")
    print("   ✓ Direct HuggingFace transformers integration")
    print("   ✓ Custom batching model manager")
    print("   ✓ Optimized for performance")
    print("   ✓ Memory-efficient quantization")
    
    print("\n🦙 LlamaIndex:")
    print("   ✓ High-level abstraction for LLM apps")
    print("   ✓ Built-in token counting and callbacks")
    print("   ✓ Chat message interface")
    print("   ✓ Easy integration with vector stores")
    print("   ✓ Structured data handling")
    
    print("\n🦜 LangChain:")
    print("   ✓ Comprehensive LLM application framework")
    print("   ✓ Chain-based architecture")
    print("   ✓ Custom output parsers")
    print("   ✓ Rich ecosystem of integrations")
    print("   ✓ Memory and conversation management")
    
    print(f"\n{'='*80}")
    print("🎯 RECOMMENDATIONS")
    print(f"{'='*80}")
    
    print("\n📋 Choose based on your needs:")
    print("   🔧 Original: Maximum performance, custom control")
    print("   🦙 LlamaIndex: RAG applications, structured data")
    print("   🦜 LangChain: Complex workflows, multi-step reasoning")

def interactive_demo():
    """Interactive demo where user can test all frameworks with custom questions"""
    print("\n" + "="*80)
    print("🎮 INTERACTIVE DEMO")
    print("="*80)
    print("Enter math problems to test all three frameworks simultaneously!")
    print("Type 'exit' to quit\n")
    
    while True:
        user_question = input("Enter a math problem: ").strip()
        
        if user_question.lower() in ['exit', 'quit', 'q']:
            print("Goodbye! 👋")
            break
        
        if not user_question:
            continue
        
        print(f"\n🔍 Testing question: {user_question}")
        
        # Test all frameworks
        frameworks = [
            ("Original", original_tutor.solve_math_problem),
            ("LlamaIndex", llamaindex_tutor.solve_math_problem_llamaindex),
            ("LangChain", langchain_tutor.solve_math_problem_langchain)
        ]
        
        for framework_name, solve_function in frameworks:
            print(f"\n--- {framework_name} ---")
            try:
                start_time = time.time()
                response = solve_function(
                    user_question, "phi2",
                    context_strategy="fixed", use_few_shot=True, max_length=1024
                )
                end_time = time.time()
                print(f"Response: {response}")
                print(f"Time: {end_time - start_time:.2f}s")
            except Exception as e:
                print(f"Error: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_demo()
    else:
        main()
        
        # Ask if user wants to try interactive demo
        print(f"\n{'='*80}")
        response = input("Would you like to try the interactive demo? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            interactive_demo() 