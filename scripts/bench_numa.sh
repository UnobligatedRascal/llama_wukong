#!/bin/bash
# llama_wukong NUMA benchmark script
# Tests: baseline (no NUMA control) vs NUMA mirror vs numactl variants
#
# Usage: ./bench_numa.sh [--model /path/to/model.gguf] [--threads N] [--prompt "text"]

set -euo pipefail

MODEL=""
THREADS=36
PROMPT="Write a Python function to compute the Fibonacci sequence using dynamic programming."
SERVER_BIN="/home/whistler/llama_wukong/build/bin/llama-server"
LOGDIR="/home/whistler/llama_wukong/bench_results"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p "$LOGDIR"

# Parse args properly
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --threads) THREADS="$2"; shift 2 ;;
        --prompt) PROMPT="$2"; shift 2 ;;
        -h|--help) echo "Usage: $0 [--model PATH] [--threads N] [--prompt TEXT]"; exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$MODEL" ]]; then
    echo "No model specified. Using HF download..."
    MODEL_FLAG="--hf-repo Qwen/Qwen2.5-0.5B-Instruct-GGUF --hf-file qwen2.5-0.5b-instruct-q4_k_m.gguf"
else
    MODEL_FLAG="-m $MODEL"
    echo "Using model: $MODEL"
fi

cleanup() {
    echo "[cleanup] Killing any remaining test servers..."
    pkill -f "llama_wukong.*--port 909[0-9]" 2>/dev/null || true
    sleep 1
}
trap cleanup EXIT

wait_ready() {
    local port=$1
    local max_wait=120
    local waited=0
    while ! ss -tlnp | grep -q ":${port} " && [[ $waited -lt $max_wait ]]; do
        sleep 1
        waited=$((waited + 1))
    done
    if [[ $waited -ge $max_wait ]]; then
        echo "[ERROR] Server on port $port did not start in time"
        return 1
    fi
    echo "[ready] Server listening on port $port after ${waited}s"
}

run_benchmark() {
    local name=$1
    local port=$2
    local cmd=$3
    local logfile="${LOGDIR}/bench_${DATE}_${name}.log"
    
    echo ""
    echo "============================================"
    echo "TEST: $name"
    echo "============================================"
    echo "Command: $cmd"
    echo "Threads: $THREADS"
    echo "Log: $logfile"
    echo ""
    
    # Kill any existing on this port
    pkill -f "--port ${port}" 2>/dev/null || true
    sleep 2
    
    # Start server
    eval "$cmd" >> "$logfile" 2>&1 &
    local server_pid=$!
    echo "[started] Server PID: $server_pid"
    
    wait_ready "$port" || return 1
    
    # Give it a moment to fully initialize
    sleep 3
    
    # Run inference benchmark with curl
    local result_file="${LOGDIR}/bench_${DATE}_${name}_result.json"
    local prompt_escaped=$(printf '%s' "$PROMPT" | python3 -c "import json,sys; print(json.dumps(sys.stdin.read()))")
    
    echo "[inference] Sending prompt..."
    curl -s --max-time 60 -X POST "http://localhost:${port}/v1/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"test\",
            \"prompt\": ${prompt_escaped},
            \"max_tokens\": 256,
            \"temperature\": 0.0,
            \"stream\": false
        }" > "$result_file" 2>&1 || echo '{"error": "curl failed"}' > "$result_file"
    
    # Parse timing from response
    local timing_info=$(python3 -c "
import json, sys
try:
    with open('$result_file') as f:
        data = json.load(f)
    if 'choices' in data and len(data['choices']) > 0:
        usage = data.get('usage', {})
        tokens = usage.get('completion_tokens', 0)
        elapsed = data.get('timings', {}).get('ttft', 'N/A')
        print(f'tokens={tokens}, ttft={elapsed}')
    else:
        print('error: no completion')
except Exception as e:
    print(f'error: {e}')
" 2>/dev/null || echo "error: extract failed")
    
    echo "[result] $timing_info"
    echo "[tokens generated preview]:"
    python3 -c "
import json
with open('$result_file') as f:
    data = json.load(f)
if 'choices' in data:
    print(data['choices'][0]['text'][:200])
" 2>/dev/null || true
    
    # Kill server
    kill "$server_pid" 2>/dev/null || true
    sleep 2
    
    echo ""
    echo "--- NUMA topology (during test) ---"
    numactl --hardware | grep -E "node [0-9] (cpus|size|free)"
    echo "---"
}

echo "=========================================="
echo "llama_wukong NUMA Benchmark Suite"
echo "Date: $(date)"
echo "=========================================="
echo ""
echo "System:"
numactl --hardware | grep -E "available|node [0-9] cpus"
echo ""
echo "numa_balancing: $(cat /proc/sys/kernel/numa_balancing)"
echo ""

# Test 1: Baseline — no NUMA control, auto-scheduling
run_benchmark "baseline" 9090 \
    "nice -n 0 $SERVER_BIN $MODEL_FLAG -t $THREADS --port 9090 -v --log-disable"

# Test 2: NUMA mirror — llama_wukong's MIRROR strategy
run_benchmark "numa_mirror" 9091 \
    "nice -n 0 $SERVER_BIN $MODEL_FLAG --numa mirror -t $THREADS --port 9091 -v --log-disable"

# Test 3: numactl bind node0 (half threads)
run_benchmark "numactl_node0" 9092 \
    "nice -n 0 numactl --cpunodebind=0 --membind=0 $SERVER_BIN $MODEL_FLAG -t 18 --port 9092 -v --log-disable"

# Test 4: numactl bind node1 (half threads)
run_benchmark "numactl_node1" 9093 \
    "nice -n 0 numactl --cpunodebind=1 --membind=1 $SERVER_BIN $MODEL_FLAG -t 18 --port 9093 -v --log-disable"

# Test 5: numactl interleave (current production behavior for llama_lazarus)
run_benchmark "numactl_interleave" 9094 \
    "nice -n 0 numactl --interleave=all $SERVER_BIN $MODEL_FLAG -t $THREADS --port 9094 -v --log-disable"

echo ""
echo "=========================================="
echo "Benchmark complete. Results in: $LOGDIR"
echo "=========================================="
ls -la "$LOGDIR"
