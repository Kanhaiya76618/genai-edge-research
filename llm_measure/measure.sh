#!/usr/bin/env bash
# Measure the serving profile of a 7B LLM with llama.cpp (Sec. V-B of the paper).
#
#   brew install llama.cpp            (or build llama.cpp from source)
#   model: mistral-7b-instruct-v0.2.Q4_K_M.gguf from huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF
#   ./measure.sh  &&  python process.py
#
# Run on an otherwise idle machine: the numbers are throughputs.
set -euo pipefail
cd "$(dirname "$0")"
M=models/mistral-7b-instruct-v0.2.Q4_K_M.gguf
[ -f "$M" ] || { echo "model not found: $M"; exit 1; }

# single-stream prefill (pp) and decode (tg) speed, 5 repetitions each
llama-bench -m "$M" -p 128,256,512,1024,2048 -n 128 -r 5 -ngl 99 -o csv > llama_bench.csv 2> llama_bench.log

# aggregate decode throughput with n concurrent sequences (continuous batching), two repetitions
for rep in 1 2; do
  llama-batched-bench -m "$M" -c 40960 -b 2048 -ub 512 -npp 256,1024 -ntg 128 -npl 1,2,4,8,16,32 -ngl 99 \
    > "batched_bench_${rep}.txt" 2> "batched_bench_${rep}.log"
done

# KV-cache size per token (printed by llama.cpp when the context is allocated)
llama-cli -m "$M" -c 4096 -n 1 -p "hi" -no-cnv -ngl 99 --temp 0 > kv_probe.txt 2> kv_probe.log || true
grep -iE "kv.*size|KV self size|llama_kv_cache" kv_probe.log | head -5 > kv_size.txt || true
echo "done"
