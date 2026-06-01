import json
import os
import time

import torch

from data import load_all_canitedit_samples
from evaluation import calculate_ast_deviation, is_valid_python
from permutations import generate_positional_prompts
from pipeline import MODEL_GROUPS, MODEL_REGISTRY, generate_batch_with_model, generate_with_model, get_model_batch_size, unload_all_models
from utils import extract_python_code, is_degenerate_output


CACHE_VERSION = "v3_multi_model"
GOLD_CORRECTNESS_THRESHOLD = 0.95
CACHE_DIR = "cache_artifacts"


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


def format_duration(seconds):
    minutes, remaining_seconds = divmod(seconds, 60)
    hours, minutes = divmod(int(minutes), 60)

    if hours > 0:
        return f"{hours:d}h {minutes:02d}m {remaining_seconds:05.2f}s"
    if minutes > 0:
        return f"{minutes:d}m {remaining_seconds:05.2f}s"
    return f"{remaining_seconds:.2f}s"


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


def merge_metrics(metrics_list):
    merged = initialize_metrics()

    for metrics in metrics_list:
        merged["generation_count"] += metrics["generation_count"]
        merged["valid_count"] += metrics["valid_count"]
        merged["degenerate_count"] += metrics["degenerate_count"]
        merged["correct_count"] += metrics["correct_count"]
        merged["pair_count"] += metrics["pair_count"]
        merged["filtered_pair_count"] += metrics["filtered_pair_count"]
        merged["correct_pair_count"] += metrics["correct_pair_count"]
        merged["gold_scores"].extend(metrics["gold_scores"])
        merged["raw_invariance_scores"].extend(metrics["raw_invariance_scores"])
        merged["filtered_invariance_scores"].extend(metrics["filtered_invariance_scores"])
        merged["correct_filtered_invariance_scores"].extend(metrics["correct_filtered_invariance_scores"])

    return merged


def ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)


def cache_file_for_model(model_key):
    ensure_cache_dir()
    return os.path.join(CACHE_DIR, f"{model_key}_results_{CACHE_VERSION}.json")


def evaluation_cache_file_for_model(model_key):
    ensure_cache_dir()
    return os.path.join(CACHE_DIR, f"{model_key}_metrics_{CACHE_VERSION}.json")


def load_results_cache(model_key):
    results_file = cache_file_for_model(model_key)
    cache = {}

    if os.path.exists(results_file):
        try:
            with open(results_file, "r") as handle:
                cache = json.load(handle)
            print(f"Loaded cached results for {MODEL_REGISTRY[model_key]['label']} from {results_file}.")
        except Exception as error:
            print(f"Failed to read cache file {results_file}: {error}")

    return results_file, cache


def load_evaluation_cache(model_key):
    metrics_file = evaluation_cache_file_for_model(model_key)
    cache = {}

    if os.path.exists(metrics_file):
        try:
            with open(metrics_file, "r") as handle:
                cache = json.load(handle)
            print(f"Loaded cached evaluation metrics for {MODEL_REGISTRY[model_key]['label']} from {metrics_file}.")
        except Exception as error:
            print(f"Failed to read evaluation cache file {metrics_file}: {error}")

    return metrics_file, cache


def cache_entry_complete(entry):
    if not entry:
        return False
    return bool(entry.get("prefix_output")) and bool(entry.get("suffix_output"))


def metrics_cache_complete(cache, total_samples):
    required_keys = set(initialize_metrics().keys())
    return bool(cache) and required_keys.issubset(cache.keys()) and cache.get("pair_count") == total_samples


def save_evaluation_cache(model_key, metrics):
    metrics_file = evaluation_cache_file_for_model(model_key)
    try:
        with open(metrics_file, "w") as handle:
            json.dump(metrics, handle, indent=4)
        print(f"Saved evaluation metrics for {MODEL_REGISTRY[model_key]['label']} to cache: {metrics_file}")
    except Exception as error:
        print(f"Failed to write evaluation cache file {metrics_file}: {error}")


def run_model_inference(model_key, prompts):
    model_label = MODEL_REGISTRY[model_key]["label"]
    batch_size = get_model_batch_size(model_key)
    start_time = time.perf_counter()
    results_file, results_cache = load_results_cache(model_key)
    needs_inference = any(not cache_entry_complete(results_cache.get(str(prompt["index"]))) for prompt in prompts)

    results = []
    if needs_inference:
        print(f"\n[Inference] Running {model_label} on all samples with batch size {batch_size}...")
        missing_prompts = [prompt for prompt in prompts if not cache_entry_complete(results_cache.get(str(prompt["index"])))]

        for start_index in range(0, len(missing_prompts), batch_size):
            batch = missing_prompts[start_index:start_index + batch_size]
            batch_range = f"{start_index + 1}-{start_index + len(batch)}"
            batch_indices = ", ".join(str(prompt["index"]) for prompt in batch)
            print(f"[{batch_range}/{len(missing_prompts)}] Generating {model_label} for samples: {batch_indices}")

            try:
                prefix_outputs = generate_batch_with_model(
                    model_key,
                    [prompt["prefix_prompt"] for prompt in batch],
                )
                suffix_outputs = generate_batch_with_model(
                    model_key,
                    [prompt["suffix_prompt"] for prompt in batch],
                )
            except Exception as error:
                print(f"Batch generation failed for {model_label} samples {batch_indices}: {error}")
                print(f"Retrying {model_label} batch samples one-by-one.")
                prefix_outputs = []
                suffix_outputs = []
                for prompt in batch:
                    try:
                        prefix_outputs.append(generate_with_model(model_key, prompt["prefix_prompt"]))
                        suffix_outputs.append(generate_with_model(model_key, prompt["suffix_prompt"]))
                    except Exception as single_error:
                        print(f"Error during generation for {model_label} sample {prompt['index']}: {single_error}")
                        prefix_outputs.append("")
                        suffix_outputs.append("")

            for prompt, prefix_output, suffix_output in zip(batch, prefix_outputs, suffix_outputs):
                results_cache[str(prompt["index"])] = {
                    "prefix_output": prefix_output,
                    "suffix_output": suffix_output,
                }

        try:
            with open(results_file, "w") as handle:
                json.dump(results_cache, handle, indent=4)
            print(f"Saved {model_label} results to cache: {results_file}")
        except Exception as error:
            print(f"Failed to write cache file {results_file}: {error}")
    else:
        print(f"All {model_label} results loaded from cache. Skipping model execution.")

    for prompt in prompts:
        cache_key = str(prompt["index"])
        cached_result = results_cache.get(cache_key, {})
        results.append({
            "index": prompt["index"],
            "prefix_output": cached_result.get("prefix_output", ""),
            "suffix_output": cached_result.get("suffix_output", ""),
        })

    unload_all_models()
    elapsed_seconds = time.perf_counter() - start_time
    print(f"Total runtime for {model_label}: {format_duration(elapsed_seconds)}")
    return results, elapsed_seconds


def evaluate_model_results(model_key, samples, model_results):
    model_label = MODEL_REGISTRY[model_key]["label"]
    metrics = initialize_metrics()

    print(f"\n=== Results: {model_label} Prefix vs Suffix ===")
    for index, sample in enumerate(samples):
        result = model_results[index]

        prefix_eval = evaluate_generation(result["prefix_output"], sample.get("after", ""))
        suffix_eval = evaluate_generation(result["suffix_output"], sample.get("after", ""))

        for generation in (prefix_eval, suffix_eval):
            record_generation(metrics, generation)

        metrics["pair_count"] += 1

        raw_score = None
        if prefix_eval["code"] and suffix_eval["code"]:
            raw_score = calculate_ast_deviation(prefix_eval["code"], suffix_eval["code"])
            metrics["raw_invariance_scores"].append(raw_score)

        if prefix_eval["valid"] and suffix_eval["valid"] and not prefix_eval["degenerate"] and not suffix_eval["degenerate"]:
            filtered_score = calculate_ast_deviation(prefix_eval["code"], suffix_eval["code"])
            metrics["filtered_invariance_scores"].append(filtered_score)
            metrics["filtered_pair_count"] += 1

            if prefix_eval["correct"] and suffix_eval["correct"]:
                metrics["correct_filtered_invariance_scores"].append(filtered_score)
                metrics["correct_pair_count"] += 1

        print(f"Sample {sample['index']} | {model_label} raw: {format_score(raw_score)}")

    return metrics


def get_or_evaluate_model_results(model_key, samples, model_results):
    _, metrics_cache = load_evaluation_cache(model_key)
    if metrics_cache_complete(metrics_cache, len(samples)):
        print(f"Skipping evaluation for {MODEL_REGISTRY[model_key]['label']}; cached metrics are complete.")
        return metrics_cache

    metrics = evaluate_model_results(model_key, samples, model_results)
    save_evaluation_cache(model_key, metrics)
    return metrics


def average_metric(scores):
    if not scores:
        return None
    return sum(scores) / len(scores)

def main():
    num_samples = 100
    print(f"=== Starting RQ2 Order Invariance Experiment on {num_samples} samples ===")

    # 1. Data Loading
    samples = load_all_canitedit_samples(limit=num_samples)
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
        
    group_metrics = {}
    all_model_metrics = []
    model_timings = {}

    print("\n=== Model Groups ===")
    print("AR:")
    for model_key in MODEL_GROUPS["ar"]:
        print(f"  - {MODEL_REGISTRY[model_key]['label']}")
    print("Diffusion:")
    for model_key in MODEL_GROUPS["diffusion"]:
        print(f"  - {MODEL_REGISTRY[model_key]['label']}")

    for group_name, model_keys in MODEL_GROUPS.items():
        print(f"\n=== Running {group_name.upper()} models ===")
        current_group_metrics = []

        for model_key in model_keys:
            model_results, elapsed_seconds = run_model_inference(model_key, prompts)
            model_timings[model_key] = elapsed_seconds
            model_metrics = get_or_evaluate_model_results(model_key, samples, model_results)
            summarize_model(MODEL_REGISTRY[model_key]["label"], model_metrics, len(samples))
            print(f"  Total runtime:                             {format_duration(elapsed_seconds)}")
            current_group_metrics.append(model_metrics)
            all_model_metrics.append(model_metrics)

        group_metrics[group_name] = merge_metrics(current_group_metrics)
        print(f"\n=== {group_name.upper()} Group Average ({len(model_keys)} models) ===")
        summarize_model(f"{group_name.upper()} Group", group_metrics[group_name], len(samples) * len(model_keys))
        group_runtime = sum(model_timings[model_key] for model_key in model_keys)
        average_group_runtime = group_runtime / len(model_keys)
        print(f"  Total group runtime:                       {format_duration(group_runtime)}")
        print(f"  Average model runtime:                     {format_duration(average_group_runtime)}")

    overall_metrics = merge_metrics(all_model_metrics)
    print(f"\n=== All Models Combined ({len(all_model_metrics)} models) ===")
    summarize_model("All Models", overall_metrics, len(samples) * len(all_model_metrics))
    total_runtime = sum(model_timings.values())
    print(f"  Total experiment runtime:                  {format_duration(total_runtime)}")

    ar_group = group_metrics["ar"]
    diffusion_group = group_metrics["diffusion"]

    filtered_bias_gap = None
    if ar_group["filtered_invariance_scores"] and diffusion_group["filtered_invariance_scores"]:
        filtered_bias_gap = average_metric(diffusion_group["filtered_invariance_scores"]) - average_metric(ar_group["filtered_invariance_scores"])

    correct_bias_gap = None
    if ar_group["correct_filtered_invariance_scores"] and diffusion_group["correct_filtered_invariance_scores"]:
        correct_bias_gap = average_metric(diffusion_group["correct_filtered_invariance_scores"]) - average_metric(ar_group["correct_filtered_invariance_scores"])

    print("\n=== Model Runtime Summary ===")
    for group_name, model_keys in MODEL_GROUPS.items():
        print(f"{group_name.upper()}:")
        for model_key in model_keys:
            print(f"  {MODEL_REGISTRY[model_key]['label']}: {format_duration(model_timings[model_key])}")

    print(f"\nFiltered Recency Bias Gap (Diffusion vs AR groups): {format_score(filtered_bias_gap)}")
    print(f"Gold-Conditioned Bias Gap (Diffusion vs AR groups): {format_score(correct_bias_gap)}")

if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    main()
