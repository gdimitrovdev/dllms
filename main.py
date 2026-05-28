from data import load_all_canitedit_samples
from utils import extract_python_code, is_degenerate_output
from permutations import generate_positional_prompts
from pipeline import generate_ar, generate_dllm, unload_ar, unload_dllm
from evaluation import calculate_ast_deviation, is_valid_python
import torch
import os
import json


CACHE_VERSION = "v2"
GOLD_CORRECTNESS_THRESHOLD = 0.95


def initialize_metrics():
    return {
        "generation_count": 0,
        "valid_count": 0,
        "degenerate_count": 0,
        "gold_scores": [],
        "correct_count": 0,
        "raw_invariance_scores": [],
        "filtered_invariance_scores": [],
        "correct_filtered_invariance_scores": [],
        "pair_count": 0,
        "filtered_pair_count": 0,
        "correct_pair_count": 0,
    }


def evaluate_generation(raw_text, gold_code):
    code = extract_python_code(raw_text)
    degenerate = is_degenerate_output(raw_text)
    valid = bool(code) and is_valid_python(code)

    gold_score = None
    correct = False
    if valid and gold_code:
        gold_score = calculate_ast_deviation(code, gold_code)
        correct = gold_score >= GOLD_CORRECTNESS_THRESHOLD

    return {
        "code": code,
        "valid": valid,
        "degenerate": degenerate,
        "gold_score": gold_score,
        "correct": correct,
    }


def record_generation(metrics, generation):
    metrics["generation_count"] += 1
    if generation["valid"]:
        metrics["valid_count"] += 1
    if generation["degenerate"]:
        metrics["degenerate_count"] += 1
    if generation["gold_score"] is not None:
        metrics["gold_scores"].append(generation["gold_score"])
    if generation["correct"]:
        metrics["correct_count"] += 1


def format_score(score):
    return f"{score:.4f}" if score is not None else "n/a"


def summarize_model(name, metrics, total_samples):
    generation_count = metrics["generation_count"] or 1
    valid_rate = metrics["valid_count"] / generation_count
    degeneration_rate = metrics["degenerate_count"] / generation_count
    avg_gold_score = sum(metrics["gold_scores"]) / len(metrics["gold_scores"]) if metrics["gold_scores"] else None
    correct_rate = metrics["correct_count"] / generation_count
    raw_avg = sum(metrics["raw_invariance_scores"]) / len(metrics["raw_invariance_scores"]) if metrics["raw_invariance_scores"] else None
    filtered_avg = sum(metrics["filtered_invariance_scores"]) / len(metrics["filtered_invariance_scores"]) if metrics["filtered_invariance_scores"] else None
    correct_filtered_avg = sum(metrics["correct_filtered_invariance_scores"]) / len(metrics["correct_filtered_invariance_scores"]) if metrics["correct_filtered_invariance_scores"] else None

    print(f"\n{name} summary:")
    print(f"  Syntax-valid generation rate:              {valid_rate:.4f} ({metrics['valid_count']}/{generation_count})")
    print(f"  Degeneration rate:                         {degeneration_rate:.4f} ({metrics['degenerate_count']}/{generation_count})")
    print(f"  Average gold AST similarity:               {format_score(avg_gold_score)}")
    print(f"  Gold-correct generation rate (>= {GOLD_CORRECTNESS_THRESHOLD:.2f}): {correct_rate:.4f} ({metrics['correct_count']}/{generation_count})")
    print(f"  Raw order-invariance score:                {format_score(raw_avg)} ({len(metrics['raw_invariance_scores'])}/{total_samples} pairs)")
    print(f"  Filtered order-invariance score:           {format_score(filtered_avg)} ({metrics['filtered_pair_count']}/{total_samples} valid/non-degenerate pairs)")
    print(f"  Gold-conditioned order-invariance score:   {format_score(correct_filtered_avg)} ({metrics['correct_pair_count']}/{total_samples} valid/correct pairs)")

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
    ar_results_file = f"ar_results_{CACHE_VERSION}.json"
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
    dllm_results_file = f"dllm_results_{CACHE_VERSION}.json"
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
    ar_metrics = initialize_metrics()
    dllm_metrics = initialize_metrics()
    
    for i, sample in enumerate(samples):
        ar_res = ar_results[i]
        dllm_res = dllm_results[i]

        ar_prefix = evaluate_generation(ar_res["ar_prefix"], sample.get("after", ""))
        ar_suffix = evaluate_generation(ar_res["ar_suffix"], sample.get("after", ""))
        dllm_prefix = evaluate_generation(dllm_res["dllm_prefix"], sample.get("after", ""))
        dllm_suffix = evaluate_generation(dllm_res["dllm_suffix"], sample.get("after", ""))

        for generation in (ar_prefix, ar_suffix):
            record_generation(ar_metrics, generation)

        for generation in (dllm_prefix, dllm_suffix):
            record_generation(dllm_metrics, generation)

        ar_metrics["pair_count"] += 1
        dllm_metrics["pair_count"] += 1

        ar_raw_score = None
        if ar_prefix["code"] and ar_suffix["code"]:
            ar_raw_score = calculate_ast_deviation(ar_prefix["code"], ar_suffix["code"])
            ar_metrics["raw_invariance_scores"].append(ar_raw_score)

        if ar_prefix["valid"] and ar_suffix["valid"] and not ar_prefix["degenerate"] and not ar_suffix["degenerate"]:
            ar_filtered_score = calculate_ast_deviation(ar_prefix["code"], ar_suffix["code"])
            ar_metrics["filtered_invariance_scores"].append(ar_filtered_score)
            ar_metrics["filtered_pair_count"] += 1

            if ar_prefix["correct"] and ar_suffix["correct"]:
                ar_metrics["correct_filtered_invariance_scores"].append(ar_filtered_score)
                ar_metrics["correct_pair_count"] += 1

        dllm_raw_score = None
        if dllm_prefix["code"] and dllm_suffix["code"]:
            dllm_raw_score = calculate_ast_deviation(dllm_prefix["code"], dllm_suffix["code"])
            dllm_metrics["raw_invariance_scores"].append(dllm_raw_score)

        if dllm_prefix["valid"] and dllm_suffix["valid"] and not dllm_prefix["degenerate"] and not dllm_suffix["degenerate"]:
            dllm_filtered_score = calculate_ast_deviation(dllm_prefix["code"], dllm_suffix["code"])
            dllm_metrics["filtered_invariance_scores"].append(dllm_filtered_score)
            dllm_metrics["filtered_pair_count"] += 1

            if dllm_prefix["correct"] and dllm_suffix["correct"]:
                dllm_metrics["correct_filtered_invariance_scores"].append(dllm_filtered_score)
                dllm_metrics["correct_pair_count"] += 1

        print(
            f"Sample {sample['index']} | "
            f"AR raw: {format_score(ar_raw_score)} | "
            f"LLaDA raw: {format_score(dllm_raw_score)}"
        )

    print(f"\n=== Final Aggregated Results ({len(samples)} total samples) ===")
    summarize_model("AR Model (Qwen)", ar_metrics, len(samples))
    summarize_model("dLLM (LLaDA)", dllm_metrics, len(samples))

    filtered_bias_gap = None
    ar_filtered_scores = ar_metrics["filtered_invariance_scores"]
    dllm_filtered_scores = dllm_metrics["filtered_invariance_scores"]
    if ar_filtered_scores and dllm_filtered_scores:
        filtered_bias_gap = (sum(dllm_filtered_scores) / len(dllm_filtered_scores)) - (sum(ar_filtered_scores) / len(ar_filtered_scores))

    correct_bias_gap = None
    ar_correct_scores = ar_metrics["correct_filtered_invariance_scores"]
    dllm_correct_scores = dllm_metrics["correct_filtered_invariance_scores"]
    if ar_correct_scores and dllm_correct_scores:
        correct_bias_gap = (sum(dllm_correct_scores) / len(dllm_correct_scores)) - (sum(ar_correct_scores) / len(ar_correct_scores))

    print(f"\nFiltered Recency Bias Gap (dLLM vs AR):          {format_score(filtered_bias_gap)}")
    print(f"Gold-Conditioned Bias Gap (dLLM vs AR):          {format_score(correct_bias_gap)}")

if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    main()
