from data import load_all_canitedit_samples
from utils import extract_python_code
from permutations import generate_positional_prompts
from pipeline import generate_ar, generate_dllm, unload_ar, unload_dllm
from evaluation import calculate_ast_deviation
import torch
import os
import json

def main():
    NUM_SAMPLES = 100
    print(f"=== Starting RQ2 Order Invariance Experiment on {NUM_SAMPLES} samples ===")

    # 1. Data Loading
    samples = load_all_canitedit_samples(limit=NUM_SAMPLES)
    print(f"Loaded {len(samples)} samples for evaluation.")
    
    # 2. Prompt Generation
    prompts = []
    for sample in samples:
        prefix_prompt, suffix_prompt = generate_positional_prompts(sample["before"], sample["instruction"])
        prompts.append({
            "index": sample["index"],
            "prefix_prompt": prefix_prompt,
            "suffix_prompt": suffix_prompt
        })
        
    # 3. Inference Pipeline (Grouped to avoid model loading/unloading overhead)
    
    # Setup cache for Autoregressive Baseline (Qwen2.5)
    ar_results_file = "ar_results.json"
    ar_results_cache = {}
    if os.path.exists(ar_results_file):
        try:
            with open(ar_results_file, "r") as f:
                ar_results_cache = json.load(f)
            print(f"Loaded cached AR results from {ar_results_file} (delete this file to force rerun).")
        except Exception as e:
            print(f"Failed to read cache file {ar_results_file}: {e}")

    # Check if we need to run AR model inference
    ar_needed = False
    for p in prompts:
        idx_str = str(p["index"])
        if idx_str not in ar_results_cache or not ar_results_cache[idx_str].get("ar_prefix") or not ar_results_cache[idx_str].get("ar_suffix"):
            ar_needed = True
            break

    ar_results = []
    if ar_needed:
        print("\n[Inference] Running Autoregressive Baseline (Qwen2.5) on all samples...")
        for idx, p in enumerate(prompts):
            idx_str = str(p["index"])
            # Use cached results if available
            if idx_str in ar_results_cache and ar_results_cache[idx_str].get("ar_prefix") and ar_results_cache[idx_str].get("ar_suffix"):
                ar_results.append({
                    "index": p["index"],
                    "ar_prefix": ar_results_cache[idx_str]["ar_prefix"],
                    "ar_suffix": ar_results_cache[idx_str]["ar_suffix"]
                })
                continue
                
            print(f"[{idx+1}/{len(prompts)}] Generating AR for sample {p['index']}...")
            try:
                raw_ar_prefix = generate_ar(p["prefix_prompt"])
                raw_ar_suffix = generate_ar(p["suffix_prompt"])
                ar_results.append({
                    "index": p["index"],
                    "ar_prefix": raw_ar_prefix,
                    "ar_suffix": raw_ar_suffix
                })
                ar_results_cache[idx_str] = {
                    "ar_prefix": raw_ar_prefix,
                    "ar_suffix": raw_ar_suffix
                }
            except Exception as e:
                print(f"Error during AR generation for sample {p['index']}: {e}")
                ar_results.append({
                    "index": p["index"],
                    "ar_prefix": "",
                    "ar_suffix": ""
                })
                
        # Save cache file after running AR
        try:
            with open(ar_results_file, "w") as f:
                json.dump(ar_results_cache, f, indent=4)
            print(f"Saved AR results to cache: {ar_results_file}")
        except Exception as e:
            print(f"Failed to write cache file {ar_results_file}: {e}")
            
        # Explicitly unload AR model to free VRAM before loading LLaDA
        unload_ar()
    else:
        print("All AR results loaded from cache. Skipping AR model execution.")
        for p in prompts:
            idx_str = str(p["index"])
            ar_results.append({
                "index": p["index"],
                "ar_prefix": ar_results_cache[idx_str]["ar_prefix"],
                "ar_suffix": ar_results_cache[idx_str]["ar_suffix"]
            })
    
    # Setup cache for Diffusion Baseline (LLaDA)
    dllm_results_file = "dllm_results.json"
    dllm_results_cache = {}
    if os.path.exists(dllm_results_file):
        try:
            with open(dllm_results_file, "r") as f:
                dllm_results_cache = json.load(f)
            print(f"Loaded cached LLaDA results from {dllm_results_file} (delete this file to force rerun).")
        except Exception as e:
            print(f"Failed to read cache file {dllm_results_file}: {e}")

    # Check if we need to run LLaDA model inference
    dllm_needed = False
    for p in prompts:
        idx_str = str(p["index"])
        if idx_str not in dllm_results_cache or not dllm_results_cache[idx_str].get("dllm_prefix") or not dllm_results_cache[idx_str].get("dllm_suffix"):
            dllm_needed = True
            break

    dllm_results = []
    if dllm_needed:
        print("\n[Inference] Running Diffusion Baseline (LLaDA) on all samples...")
        for idx, p in enumerate(prompts):
            idx_str = str(p["index"])
            # Use cached results if available
            if idx_str in dllm_results_cache and dllm_results_cache[idx_str].get("dllm_prefix") and dllm_results_cache[idx_str].get("dllm_suffix"):
                dllm_results.append({
                    "index": p["index"],
                    "dllm_prefix": dllm_results_cache[idx_str]["dllm_prefix"],
                    "dllm_suffix": dllm_results_cache[idx_str]["dllm_suffix"]
                })
                continue
                
            print(f"[{idx+1}/{len(prompts)}] Generating LLaDA for sample {p['index']}...")
            try:
                raw_dllm_prefix = generate_dllm(p["prefix_prompt"])
                raw_dllm_suffix = generate_dllm(p["suffix_prompt"])
                dllm_results.append({
                    "index": p["index"],
                    "dllm_prefix": raw_dllm_prefix,
                    "dllm_suffix": raw_dllm_suffix
                })
                dllm_results_cache[idx_str] = {
                    "dllm_prefix": raw_dllm_prefix,
                    "dllm_suffix": raw_dllm_suffix
                }
            except Exception as e:
                print(f"Error during LLaDA generation for sample {p['index']}: {e}")
                dllm_results.append({
                    "index": p["index"],
                    "dllm_prefix": "",
                    "dllm_suffix": ""
                })
                
        # Save cache file after running LLaDA
        try:
            with open(dllm_results_file, "w") as f:
                json.dump(dllm_results_cache, f, indent=4)
            print(f"Saved LLaDA results to cache: {dllm_results_file}")
        except Exception as e:
            print(f"Failed to write cache file {dllm_results_file}: {e}")
            
        # Explicitly unload LLaDA model to clean VRAM
        unload_dllm()
    else:
        print("All LLaDA results loaded from cache. Skipping LLaDA model execution.")
        for p in prompts:
            idx_str = str(p["index"])
            dllm_results.append({
                "index": p["index"],
                "dllm_prefix": dllm_results_cache[idx_str]["dllm_prefix"],
                "dllm_suffix": dllm_results_cache[idx_str]["dllm_suffix"]
            })
    
    # 4. Evaluation and Metrics
    print("\n=== Results: Positional Shift (Prefix vs Suffix) ===")
    ar_scores = []
    dllm_scores = []
    
    for i, sample in enumerate(samples):
        ar_res = ar_results[i]
        dllm_res = dllm_results[i]
        
        # Skip if generation failed
        if not ar_res["ar_prefix"] or not ar_res["ar_suffix"] or not dllm_res["dllm_prefix"] or not dllm_res["dllm_suffix"]:
            print(f"Sample {sample['index']} | Skipped due to generation failure.")
            continue
            
        ar_prefix_code = extract_python_code(ar_res["ar_prefix"])
        ar_suffix_code = extract_python_code(ar_res["ar_suffix"])
        dllm_prefix_code = extract_python_code(dllm_res["dllm_prefix"])
        dllm_suffix_code = extract_python_code(dllm_res["dllm_suffix"])
        
        try:
            ar_score = calculate_ast_deviation(ar_prefix_code, ar_suffix_code)
            dllm_score = calculate_ast_deviation(dllm_prefix_code, dllm_suffix_code)
            ar_scores.append(ar_score)
            dllm_scores.append(dllm_score)
            print(f"Sample {sample['index']} | AR Score: {ar_score:.4f} | LLaDA Score: {dllm_score:.4f}")
        except Exception as e:
            print(f"Sample {sample['index']} | AST Parsing failed: {e}")
            
    if ar_scores and dllm_scores:
        avg_ar = sum(ar_scores) / len(ar_scores)
        avg_dllm = sum(dllm_scores) / len(dllm_scores)
        bias_gap = avg_dllm - avg_ar
        print(f"\n=== Final Aggregated Results ({len(ar_scores)} successful parse samples) ===")
        print(f"Average AR Model (Qwen) Order-Invariance Score:   {avg_ar:.4f} (1.0 = perfectly invariant)")
        print(f"Average dLLM (LLaDA) Order-Invariance Score:      {avg_dllm:.4f} (1.0 = perfectly invariant)")
        print(f"Average Recency Bias Gap (dLLM vs AR):            {bias_gap:.4f}")
    else:
        print("\nNo samples were successfully parsed.")

if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    main()
