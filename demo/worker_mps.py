#!/usr/bin/env python3
import warnings, sys, os, json, argparse, transformers, random as rd, time, torch, numpy as np, statistics, copy

# ---- your imports (unchanged) ----
from finemoe import MoE
from configs.common.config_common import offload_path, state_path, device_memory_ratio, device
from configs.models.config_qwen import model_path, prefetch_distance, store_capacity
from configs.datasets.config_lmsys import dataset_path, max_length, max_new_tokens, min_new_tokens

# -------- CLI --------
p = argparse.ArgumentParser()
p.add_argument("--id", type=int, required=True, help="Worker ID (0..N-1)")
p.add_argument("--runs", type=int, default=10, help="How many prompts this worker should run")
p.add_argument("--out", type=str, required=True, help="Output JSON path")
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()

# Optional: keep CPU contention low per worker
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
try:
    torch.set_num_threads(1)
except Exception:
    pass

rd.seed(args.seed + args.id)
np.random.seed(args.seed + args.id)
torch.manual_seed(args.seed + args.id)

sys.path.append("../")
warnings.filterwarnings("ignore")

def inference(model, tokenizer, prompt, max_length, max_new_tokens):
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
            for _ in range(num_decode_tokens):
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

def run_benchmark(model, tokenizer, generate_config, benchmark_prompts, warmup_prompt, mode_name, max_length, max_new_tokens, output_log_file):
    print(f"\n--- [worker {args.id}] Running Benchmark for: {mode_name} ---")
    print("Performing a warm-up run...")
    warmup_inputs = tokenizer([warmup_prompt], truncation=True, padding="longest", return_tensors="pt").to(device) 
    with torch.no_grad():
        _ = model.generate(
            warmup_inputs.input_ids, max_new_tokens=2, min_new_tokens=1,
            attention_mask=warmup_inputs.attention_mask, **generate_config
        )
    torch.cuda.synchronize(device)
    print("Warm-up complete.")

    ttft_list, tpot_list, benchmark_results = [], [], []
    print(f"Running {len(benchmark_prompts)} benchmark prompts...")
    for i, prompt in enumerate(benchmark_prompts):
        print(f"  [worker {args.id}] Run {i+1}/{len(benchmark_prompts)}...")
        ttft, tpot, output_text = inference(
            model=model, tokenizer=tokenizer, prompt=prompt,
            max_length=max_length, max_new_tokens=max_new_tokens,
        )
        ttft_list.append(ttft)
        tpot_list.append(tpot)
        benchmark_results.append({
            "worker_id": args.id,
            "run": i + 1, "mode": mode_name, "prompt": prompt, "output_text": output_text,
            "ttft_sec": ttft, "tpot_sec": tpot
        })

    # Save per-worker detailed results
    with open(output_log_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2)

    avg_ttft = statistics.mean(ttft_list)
    avg_tpot = statistics.mean(tpot_list)
    avg_throughput = 1.0 / avg_tpot if avg_tpot > 0 else 0
    print(f"\n--- [worker {args.id}] RESULTS ({mode_name}) ---")
    print(f"--- Average TTFT: {avg_ttft:.4f} seconds")
    print(f"--- Average TPOT: {avg_tpot:.4f} seconds")
    print(f"--- Average Throughput: {avg_throughput:.2f} tokens/sec ---")
    return avg_ttft, avg_tpot

if __name__ == "__main__":
    # Load prompts deterministically but independently per worker
    dataset_name = dataset_path.split("/")[-1]
    moe_name = model_path.split("/")[-1]

    with open(f"{state_path}/{dataset_name}~eval_prompts.json", "r") as f:
        all_prompts = [p["prompt"] for p in json.load(f)]

    # Each worker samples its own subset (no overlap needed for MPS contention)
    num_benchmark_runs = args.runs
    sampled_prompts = rd.sample(all_prompts, num_benchmark_runs + 1)
    warmup_prompt, benchmark_prompts = sampled_prompts[0], sampled_prompts[1:]

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path, device=device, clean_up_token_spaces=True,
        trust_remote_code=True, padding_side="left",
    )
    tokenizer.pad_token = tokenizer.eos_token
    generate_config = {"pad_token_id": tokenizer.pad_token_id} if "qwen" in moe_name.lower() else {}
    if not generate_config:
        raise ValueError(f"Model {moe_name} not supported")

    base_moe_config = {
        # IMPORTANT: unique offload dir per worker to avoid file contention
        "offload_path": os.path.join(offload_path, moe_name, f"w{args.id}"),
        "device_memory_ratio": device_memory_ratio,
        "store_capacity": store_capacity,
        "device": device,
        "eval_batch_size": 1,
        "eval_max_length": max_length,
        "eval_mode": "offline",
    }
    os.makedirs(base_moe_config["offload_path"], exist_ok=True)

    finegrained_config = copy.deepcopy(base_moe_config)
    finegrained_config["prefetch_distance"] = prefetch_distance 
    
    print(f"\n[worker {args.id}] Initializing model (prefetch_distance={prefetch_distance})...")
    model_finegrained = MoE(model_path, finegrained_config)
    
    print("[worker {}] Loading expert trace...".format(args.id))
    model_finegrained.engine.expert_tracer.expert_map_store.import_store_data(
        f"{state_path}/{moe_name}~{dataset_name}"
    )

    # Run and write to a worker-specific file passed in --out
    run_benchmark(
        model=model_finegrained, tokenizer=tokenizer, generate_config=generate_config,
        benchmark_prompts=benchmark_prompts, warmup_prompt=warmup_prompt,
        mode_name="FINE-GRAINED (Prefetch)", max_length=max_length,
        max_new_tokens=max_new_tokens, output_log_file=args.out
    )
    print(f"[worker {args.id}] Done.")
