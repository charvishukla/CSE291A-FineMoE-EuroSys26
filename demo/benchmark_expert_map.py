#
# benchmark_baseline.py
# (This is the "Naive Offload" test)
#
import warnings, sys, os, json, transformers, random as rd, time, torch, numpy as np, statistics, copy

from finemoe import MoE
from configs.common.config_common import offload_path, state_path, device_memory_ratio, device
from configs.models.config_qwen import model_path, prefetch_distance, store_capacity
from configs.datasets.config_lmsys import dataset_path, max_length, max_new_tokens, min_new_tokens

rd.seed(42)
sys.path.append("../")
warnings.filterwarnings("ignore")

def inference(model, tokenizer, prompt, max_length, max_new_tokens):
    """Runs a single inference, measuring TTFT and TPOT."""
    inputs = tokenizer(
        [prompt], truncation=True, padding="longest", max_length=max_length, return_tensors="pt"
    ).to(device)
    input_ids = inputs.input_ids
    past_key_values, next_token, all_token_ids = None, None, []
    
    with torch.no_grad():
        torch.cuda.synchronize(device) 
        start_time = time.perf_counter()
        outputs = model(input_ids=input_ids, use_cache=True, past_key_values=None)
        logits, past_key_values = outputs.logits, outputs.past_key_values
        next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
        all_token_ids.append(next_token.cpu())
        torch.cuda.synchronize(device) 
        ttft = time.perf_counter() - start_time
        
    num_decode_tokens = max_new_tokens - 1
    token_times = []
    if num_decode_tokens > 0:
        with torch.no_grad():
            for i in range(num_decode_tokens):
                torch.cuda.synchronize(device)
                token_start_time = time.perf_counter()
                outputs = model(input_ids=next_token, use_cache=True, past_key_values=past_key_values)
                logits, past_key_values = outputs.logits, outputs.past_key_values
                next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
                all_token_ids.append(next_token.cpu())
                torch.cuda.synchronize(device)
                token_times.append(time.perf_counter() - token_start_time)
    
    tpot = statistics.mean(token_times) if token_times else 0.0
    output_ids = torch.cat(all_token_ids, dim=-1)
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return ttft, tpot, output_text

# =================================================================
# === 'run_benchmark' FUNCTION (with all fixes) ===
# =================================================================
def run_benchmark(model, tokenizer, generate_config, benchmark_prompts, warmup_prompt, mode_name, max_length, max_new_tokens, output_log_file):
    """Runs a full benchmark, logs basic metrics, and saves a summary."""
    print(f"\n--- Running Benchmark for: {mode_name} ---")
    print("Performing a warm-up run...")
    warmup_inputs = tokenizer([warmup_prompt], truncation=True, padding="longest", return_tensors="pt").to(device) 
    
    with torch.no_grad():
        _ = model.generate(
            warmup_inputs.input_ids, max_new_tokens=2, min_new_tokens=1,
            attention_mask=warmup_inputs.attention_mask, **generate_config
        )
    torch.cuda.synchronize(device)
    print("Warm-up complete.")

    all_run_results = []

    print(f"Running {len(benchmark_prompts)} benchmark prompts...")
    for i, prompt in enumerate(benchmark_prompts):
        print(f"  Run {i+1}/{len(benchmark_prompts)}...")
        
        ttft, tpot, output_text = inference(
            model=model, tokenizer=tokenizer, prompt=prompt,
            max_length=max_length, max_new_tokens=max_new_tokens,
        )
        
        all_run_results.append({
            "run": i + 1, "mode": mode_name, "prompt": "...", "output_text": output_text,
            "ttft_sec": ttft, "tpot_sec": tpot
        })

    print(f"Saving detailed results to {output_log_file}...")
    with open(output_log_file, "w", encoding="utf-8") as f:
        json.dump(all_run_results, f, indent=2)

    cache_hit_rate = -1.0
    try:
        engine_handle = model.engine.archer_engine
        hit_rate_tensor = None

        if hasattr(engine_handle, "GetHitRate"):
             hit_rate_tensor = engine_handle.GetHitRate()
        elif hasattr(engine_handle, "get_hit_rate"):
            hit_rate_tensor = engine_handle.get_hit_rate()
        else:
            print("  WARNING: Could not find GetHitRate or get_hit_rate on model.engine.archer_engine.")

        if hit_rate_tensor is not None:
            cache_hit_rate = torch.mean(hit_rate_tensor.to(torch.float32)).item()
            
    except Exception as e:
        print(f"  WARNING: Could not get cache hit rate. Error: {e}")


    summary_data = {
        "mode": mode_name,
        "avg_ttft_sec": statistics.mean([r["ttft_sec"] for r in all_run_results]),
        "avg_tpot_sec": statistics.mean([r["tpot_sec"] for r in all_run_results]),
        "avg_throughput_tps": 1.0 / statistics.mean([r["tpot_sec"] for r in all_run_results]),
        "expert_cache_hit_rate": cache_hit_rate,
    }
    
    summary_filename = output_log_file.replace(".json", "_summary.json")
    print(f"Saving aggregate summary to {summary_filename}...")
    with open(summary_filename, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\n--- RESULTS ({mode_name}) ---")
    print(f"--- Average TTFT: {summary_data['avg_ttft_sec']:.4f} seconds")
    print(f"--- Average TPOT: {summary_data['avg_tpot_sec']:.4f} seconds")
    print(f"--- Average Throughput: {summary_data['avg_throughput_tps']:.2f} tokens/sec")
    print(f"--- Expert Cache Hit Rate: {summary_data['expert_cache_hit_rate']:.4f} ---")
    
    return summary_data['avg_ttft_sec'], summary_data['avg_tpot_sec']

if __name__ == "__main__":
    num_benchmark_runs = 15
    moe_name = model_path.split("/")[-1]
    dataset_name = dataset_path.split("/")[-1]

    with open(f"{state_path}/{dataset_name}~eval_prompts.json", "r") as f:
        all_prompts = [p["prompt"] for p in json.load(f)]
    sampled_prompts = rd.sample(all_prompts, num_benchmark_runs + 1)
    warmup_prompt, benchmark_prompts = sampled_prompts[0], sampled_prompts[1:]

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path, device=device, clean_up_tokenization_spaces=True,
        trust_remote_code=True, padding_side="left",
    )
    tokenizer.pad_token = tokenizer.eos_token

    generate_config = {"pad_token_id": tokenizer.pad_token_id} if "qwen" in moe_name.lower() else {}
    if not generate_config:
        raise ValueError(f"Model {moe_name} not supported")

    base_moe_config = {
        "offload_path": os.path.join(offload_path, moe_name),
        "device_memory_ratio": device_memory_ratio,
        "store_capacity": store_capacity,
        "device": device,
        "eval_batch_size": 1, 
        "eval_max_length": max_length,
        "eval_mode": "offline", 
    }
    

    baseline_config = copy.deepcopy(base_moe_config)
    baseline_config["prefetch_distance"] = 6 
    
    print(f"\nInitializing model in BASELINE (Naive Prefetch d=1) mode (prefetch_distance = 6)...")
    model_baseline = MoE(model_path, baseline_config)
    
    print("Loading expert trace...")
    model_baseline.engine.expert_tracer.expert_map_store.import_store_data(
        f"{state_path}/{moe_name}~{dataset_name}"
    )
    
    run_benchmark(
        model=model_baseline, tokenizer=tokenizer, generate_config=generate_config,
        benchmark_prompts=benchmark_prompts, warmup_prompt=warmup_prompt,
        mode_name="BASELINE (Naive Prefetch d=6)", 
        max_length=max_length,
        max_new_tokens=max_new_tokens, 
        output_log_file="baseline_benchmark_log_qwen.json"
    )
    print("\n--- Baseline benchmark complete. ---")