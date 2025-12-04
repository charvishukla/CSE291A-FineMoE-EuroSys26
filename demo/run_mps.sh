#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_mps.sh 1   # N=1 workers
#   ./run_mps.sh 2   # N=2 workers
#   ./run_mps.sh 4   # N=4 workers
N=${1:-1}

# --- Config ---
export CUDA_VISIBLE_DEVICES=0
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=100   # you can try 50/70/etc to shape fairness

# Optional: isolate CUDA caches/logs per worker fan-out
export CUDA_CACHE_MAXSIZE=2147483647

# Clean old outputs
rm -f mps_out_*.json

echo "[launcher] Starting CUDA MPS daemon..."
nvidia-cuda-mps-control -d

# small wait to ensure daemon is up
sleep 1

# --- Launch N concurrent workers ---
echo "[launcher] Launching $N workers..."
for i in $(seq 0 $((N-1))); do
  python -u worker_mps.py \
    --id "$i" \
    --runs 8 \
    --out "mps_out_${i}.json" \
    --seed 42 \
    > "worker_${i}.log" 2>&1 &
done

# Wait for all children
wait
echo "[launcher] Workers finished."

# --- Aggregate summary ---
python - <<'PY'
import glob, json, statistics as st
files = sorted(glob.glob("mps_out_*.json"))
ttft, tpot = [], []
for f in files:
    runs = json.load(open(f))
    ttft += [r["ttft_sec"] for r in runs]
    tpot += [r["tpot_sec"] for r in runs]

def p95(xs):
    xs = sorted(xs)
    return xs[int(0.95*(len(xs)-1))] if xs else 0.0

print("SUMMARY for N workers =", len(files))
print("TTFT_mean =", round(st.mean(ttft), 6) if ttft else 0.0, "TTFT_p95 =", round(p95(ttft), 6))
print("TPOT_mean =", round(st.mean(tpot), 6) if tpot else 0.0, "TPOT_p95 =", round(p95(tpot), 6))
PY

# --- Stop MPS ---
echo quit | nvidia-cuda-mps-control
echo "[launcher] MPS stopped."
