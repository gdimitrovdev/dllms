from data import load_canitedit_sample
from utils import extract_python_code
from permutations import generate_positional_prompts, generate_shuffled_prompts
from pipeline import generate_ar, generate_dllm
from evaluation import calculate_ast_deviation
import torch

def main():
    print("=== Starting RQ2 Order Invariance Experiment ===")

    # 1. Data Loading
    before_code, instruction, constraints = load_canitedit_sample(index=0)
    print(f"\n[Data] Original Instruction: {instruction}")
    
    # 2. Prompt Generation
    prefix_prompt, suffix_prompt = generate_positional_prompts(before_code, instruction)
    prompt_ab, prompt_ba = generate_shuffled_prompts(before_code, constraints)
    
    # 3. Inference Pipeline
    
    print("\n[Inference] Running Autoregressive Baseline (Qwen2.5)...")
    raw_ar_prefix = generate_ar(prefix_prompt)
    raw_ar_suffix = generate_ar(suffix_prompt)
    
    print("[Inference] Running Diffusion Baseline (LLaDA)...")
    raw_dllm_prefix = generate_dllm(prefix_prompt)
    raw_dllm_suffix = generate_dllm(suffix_prompt)
    
    ar_prefix_code = extract_python_code(raw_ar_prefix)
    ar_suffix_code = extract_python_code(raw_ar_suffix)
    dllm_prefix_code = extract_python_code(raw_dllm_prefix)
    dllm_suffix_code = extract_python_code(raw_dllm_suffix)

    # 4. Evaluation and Metrics
    print("\n=== Results: Positional Shift (Prefix vs Suffix) ===")
    try:
        ar_score = calculate_ast_deviation(ar_prefix_code, ar_suffix_code)
        dllm_score = calculate_ast_deviation(dllm_prefix_code, dllm_suffix_code)
        bias_gap = dllm_score - ar_score
        
        print(f"AR Model (Qwen) Order-Invariance Score:   {ar_score:.4f} (1.0 = perfectly invariant)")
        print(f"dLLM (LLaDA) Order-Invariance Score:      {dllm_score:.4f} (1.0 = perfectly invariant)")
        print(f"Recency Bias Gap (dLLM vs AR):            {bias_gap:.4f}")
        
    except Exception as e:
        print(f"\n[Error] AST Parsing failed. The model likely generated invalid Python syntax. Details: {e}")

if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    main()
