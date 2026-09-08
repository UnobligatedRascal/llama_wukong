#!/bin/bash
# test_rope_lut.sh - Comprehensive RoPE LUT performance test suite
# Tests: LUT on/off × 3 GPU configs × 3 slot counts × long-prompt processing
#
# Usage: bash scripts/test_rope_lut.sh [--quick] [--no-build]
#   --quick   : shorter generation (128 tokens vs 256), still uses long prompts
#   --no-build: skip rebuild, use existing build_lut_on/off binaries
#
# Output: test_logs/rope_lut_YYYYMMDD_HHMMSS/ with per-test logs and summary

set -euo pipefail

PROJ="/home/whistler/llama_wukong"
MODEL="/mnt/512gb_ssd/models/Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-MTP-Q4_K_M.gguf"
TEMPLATE="/home/whistler/models/tuvak.jinja"
API_KEY="YOUR_API_KEY_HERE"
BUILD_DIR="$PROJ/build"
LOG_DIR="$PROJ/test_logs/rope_lut_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

START_PORT=4999
PROMPT_TTF_TIMEOUT=600      # 10 min max for first token (long prompt on K80)
SERVER_START_TIMEOUT=180    # 3 min max for server startup
CONCURRENT_REQUEST_TIMEOUT=900  # 15 min for parallel completion tests

QUICK=0
NO_BUILD=0
for arg in "$@"; do
    case "$arg" in
        --quick)   QUICK=1 ;;
        --no-build) NO_BUILD=1 ;;
    esac
done

N_PREDICT=$([ "$QUICK" -eq 1 ] && echo 128 || echo 256)

# Generate a long synthetic prompt (~32K tokens worth of structured text)
generate_long_prompt() {
    local len=$1  # approximate token count target
    local prompt="SYSTEM: You are a technical expert. Provide detailed, thorough analysis.

USER: Analyze the following distributed systems architecture design document.
Identify all potential failure modes, race conditions, and scalability bottlenecks.
Provide specific recommendations for each issue found.

DOCUMENT BEGIN:

The following describes a multi-region, multi-tenant SaaS platform architecture
for real-time collaborative document editing with AI-assisted features.

Region Configuration:
- Primary: US-East (N. Virginia) with 3 availability zones
- Secondary: EU-West (Ireland) with 3 availability zones
- Tertiary: AP-Southeast (Singapore) with 2 availability zones
- Each region runs a full copy of the application stack for failover capability

Compute Layer:
- API Gateway: Kong clusters, 4 nodes per region, autoscaling to 20
- Auth Service: OAuth2/OIDC provider, 2 nodes per region, stateful with Redis backend
- Document Service: Go microservice, 8 nodes per region, handles CRUD operations
- Collaboration Service: Elixir cluster for real-time operational transformation
  - 6 nodes per region, persistent connections via WebSockets
  - OT conflict resolution with vector clocks for document versions
- AI Service: Python/FastAPI with model serving via Triton
  - Model replicas: 4 per region for Qwen3.6-27B
  - GPU nodes: NVIDIA A100-80GB, 8 per region
  - Request batching with 100ms collection window
- Notification Service: Node.js cluster, 3 nodes per region
  - Pub/sub via Kafka for event distribution
  - Rate limiting: 1000 notifications/minute per user

Data Layer:
- Primary Database: PostgreSQL 16 with Citus extension for distributed tables
  - Shard key: tenant_id (hash distribution across 32 shards)
  - Read replicas: 2 per shard, async replication with <1s lag
  - Write-ahead logging streamed to S3 for disaster recovery
- Cache Layer: Redis Cluster, 6 nodes per region
  - Document locks: 30-second TTL with automatic renewal
  - Session storage: user authentication tokens
  - Rate limiter counters with sliding window algorithm
- Message Queue: Apache Kafka, 9 brokers per region (3 racks × 3 brokers)
  - Topics: document-events (128 partitions), user-activity (64 partitions), ai-jobs (32 partitions)
  - Retention: 7 days for events, 30 days for audit logs
  - Replication factor: 3 with majority commit requirement
- Object Storage: MinIO cluster (S3-compatible), 12 nodes per region
  - Document snapshots stored with versioning enabled
  - Lifecycle policy: transition to Glacier after 90 days of inactivity

Real-Time Collaboration Protocol:
- WebSocket connections multiplexed via Nginx with sticky sessions
- Operational Transform (OT) algorithm for conflict resolution
- Document state maintained as JSON patch operations with sequence numbers
- Offline support: client-side operation queue with eventual consistency
- Presence tracking: user cursors and selections broadcast every 200ms

AI Integration:
- Auto-completion: triggered on typing pause >500ms
  - Context window: 4096 tokens from current document section
  - Response timeout: 2 seconds, silent fallback if exceeded
- Smart suggestions: background analysis every 60 seconds of inactivity
  - Uses full document context up to 32K tokens
  - Results cached for 5 minutes to avoid redundant computation
- Document summarization: on-demand feature with progress indicator
  - Async job queued to AI service
  - Result delivered via WebSocket when complete

Security Architecture:
- mTLS between all microservices using cert-manager with 24h rotation
- API keys hashed with Argon2id for tenant authentication
- Data encryption: AES-256-GCM at rest, TLS 1.3 in transit
- RBAC with 7 permission levels: owner, admin, editor, commenter, viewer, guest, public
- Audit logging for all document access and permission changes

Observability:
- Metrics: Prometheus with 15s scrape interval
  - Custom metrics for document edit latency, collaboration conflicts, AI response time
  - Alerting via PagerDuty for P0/P1 incidents
- Tracing: Jaeger with 1% sampling for production, 10% for staging
- Logging: ELK stack with structured JSON logs
  - Log rotation: 28 days hot, 1 year warm, indefinite cold storage

Scaling Requirements:
- Target: 100,000 concurrent users, 10,000 active documents
- Document edit latency: <100ms p99 for local operations
- AI suggestion latency: <2s p95
- Regional failover: <5 minutes RTO, <1 minute RPO
- Multi-region active-active for read operations, active-passive for writes

End of document.

ANALYSIS REQUESTED:
1. Map all data flows between components and identify single points of failure
2. Calculate expected resource requirements under peak load
3. Identify potential consistency issues in the multi-region deployment
4. Recommend specific configuration changes to improve reliability
5. Assess the AI service scaling strategy under heavy concurrent usage
6. Evaluate the OT-based collaboration approach for documents >100K characters
7. Propose disaster recovery runbook steps for each failure scenario
8. Calculate cost implications of running this architecture at target scale

Provide your analysis in structured sections with specific, actionable recommendations.
For each issue, include: severity (critical/high/medium/low), affected components,
root cause analysis, and recommended fix with implementation effort estimate."

    # Pad with structured technical content to approach ~32K tokens
    for i in $(seq 1 40); do
        prompt+="

ADDENDUM $i: Additional technical specifications for component behavior.

The $i-th subsystem requires the following integration parameters:
- Connection pool size: $((i * 50)) with max wait time of $((i * 10))ms
- Circuit breaker: opens after $((i * 5)) consecutive failures, half-open after $((i * 100))ms
- Retry policy: exponential backoff starting at $((i * 10))ms, max $((i * 500))ms, max 3 attempts
- Health check interval: $((i * 500))ms with $((i * 2)) consecutive failures required for unhealthy status
- Graceful shutdown timeout: $((i * 5))s to drain in-flight requests
- Thread pool configuration: $((i * 4)) core threads, $((i * 8)) max threads, $((i * 60))s keep-alive
- Request timeout hierarchy: operation=$((i * 100))ms, service=$((i * 500))ms, client=$((i * 2000))ms
- Buffer sizes: read=$((i * 8192)) bytes, write=$((i * 16384)) bytes, high-water=$((i * 65536)) bytes
- Logging levels: production=INFO, staging=DEBUG, development=TRACE
- Feature flags: canary deployment at $((i * 2))% traffic, rolling rollout at 10%/15min intervals

Performance characteristics observed in load testing phase $i:
- Throughput: $((i * 500)) requests/second per instance at p50 latency
- Memory: $((i * 512)) MB baseline + $((i * 16)) MB per 1000 concurrent connections
- CPU: $((i * 8)) cores saturated at $((i * 1000)) concurrent requests
- Network: $((i * 2)) Gbps aggregate, $((i * 50000)) connections/second rate limit
- Disk I/O: $((i * 1000)) IOPS read, $((i * 500)) IOPS write with $((i * 10))ms average latency

Error budget allocation for subsystem $i:
- Monthly SLA target: $((99 + i % 5)).$((i % 10))% availability
- Error budget: $((10000 - (99 + i % 5) * 100 - (i % 10))) minutes/month
- Budget consumption alerts at 50% (warning) and 80% (critical)
- Freeze deployments when budget exhausted for 24-hour investigation window"
    done

    prompt+="

Based on all the above specifications, provide your comprehensive analysis.
Focus on practical, implementable solutions that balance reliability with cost."

    echo "$prompt"
}

LONG_PROMPT=""

# Kill any leftover llama-server processes on our test ports
cleanup_ports() {
    local port=$1
    local pid
    pid=$(lsof -ti:"$port" 2>/dev/null || true)
    if [ -n "$pid" ]; then
        echo "  Cleaning up leftover process on port $port (pid $pid)..."
        kill "$pid" 2>/dev/null || true
        sleep 1
        kill -9 "$pid" 2>/dev/null || true
    fi
}

# Build LUT variants
build_variant() {
    local name="$1"
    local lut_val="$2"
    local build_subdir="build_lut_${name}"
    local binary="$PROJ/$build_subdir/bin/llama-server"
    
    # Skip if exists
    if [ "$NO_BUILD" -eq 1 ] && [ -f "$binary" ]; then
        echo "  Using existing $name binary: $binary"
        echo "$binary"
        return
    fi
    
    echo "  Building $name (ROPE_USE_LUT=$lut_val)..."
    mkdir -p "$PROJ/$build_subdir"
    
    cd "$PROJ"
    cmake -B "$build_subdir" \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON \
        -DGGML_SCHED_MAX_COPIES=2 \
        -DGGML_CUDA_NCCL=ON \
        -DGGML_CUDA_FA_ALL_QUANTS=ON \
        -DGGML_CUDA_FORCE_MMQ=ON \
        -DGGML_CUDA_FA=ON \
        -DGGML_CUDA_GRAPHS=OFF \
        -DCMAKE_CUDA_ARCHITECTURES="37" \
        -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.8/bin/nvcc \
        -DGGML_CUDA_PEER_MAX_BATCH_SIZE=32 \
        -DCMAKE_CUDA_HOST_COMPILER=g++-11 \
        -DCMAKE_C_COMPILER=gcc-11 \
        -DCMAKE_CXX_COMPILER=g++-11 \
        -DGGML_AVX2=ON \
        -DGGML_AVX512=OFF \
        -DGGML_FMA=ON \
        -DGGML_F16C=ON \
        -DGGML_SSE42=ON \
        -DGGML_BMI2=ON \
        -DGGML_NATIVE=OFF \
        -DGGML_OPENMP=ON \
        -DLLAMA_CURL=OFF \
        -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-rpath,/usr/local/cuda-11.8/targets/x86_64-linux/lib" \
        -DGGML_CUDA_CUBLAS=ON \
        -DCMAKE_CUDA_FLAGS="-DROPE_USE_LUT=$lut_val" \
        > "$LOG_DIR/cmake_${name}.log" 2>&1
    
    cmake --build "$build_subdir" --target llama-server -j$(nproc) \
        > "$LOG_DIR/build_${name}.log" 2>&1
    
    if [ -f "$binary" ]; then
        echo "  Built: $binary"
        echo "$binary"
    else
        echo "  ERROR: Build failed for $name"
        tail -20 "$LOG_DIR/build_${name}.log"
        exit 1
    fi
}

# Start server with timeout and health check
start_server() {
    local config_name="$1"
    local binary="$2"
    local cuda_devs="$3"
    local numa_node="$4"
    local threads="$5"
    local affinity="$6"
    local tensor_split="$7"
    local np="$8"
    local port="$9"
    local log_file="$LOG_DIR/${config_name}.log"
    
    cleanup_ports "$port"
    
    # Start server
    sudo GGML_NUMA_REPLICATE=0 GGML_CUDA_P2P=1 OMP_NUM_THREADS=$threads GOMP_CPU_AFFINITY="$affinity" \
        CUDA_VISIBLE_DEVICES="$cuda_devs" \
        -E nice -n -20 \
        numactl --cpunodebind="$numa_node" --membind="$numa_node" \
        "$binary" \
        -m "$MODEL" \
        -t "$threads" \
        -c 65536 \
        -ngl 99 \
        --port "$port" \
        --host 0.0.0.0 \
        --api-key "$API_KEY" \
        --jinja \
        --chat-template-file "$TEMPLATE" \
        --load-mode none \
        -np "$np" \
        --batch-size 4096 \
        --ubatch-size 1024 \
        --cache-type-k q4_0 \
        --cache-type-v q4_0 \
        --tensor-split "$tensor_split" \
        --kv-unified \
        --seed 1016 \
        --spec-type draft-mtp \
        --spec-draft-p-min 0.75 \
        --spec-draft-n-max 3 \
        --split-mode tensor \
        > "$log_file" 2>&1 &
    
    local server_pid=$!
    echo "$server_pid" > "$LOG_DIR/${config_name}.pid"
    
    # Health check with timeout
    local tries=0
    while [ $tries -lt "$SERVER_START_TIMEOUT" ]; do
        if curl -sf "http://127.0.0.1:$port/health" 2>/dev/null | grep -q healthy; then
            return 0
        fi
        sleep 1
        tries=$((tries + 1))
    done
    
    echo "  ERROR: Server failed to start on port $port within ${SERVER_START_TIMEOUT}s"
    cat "$log_file" | tail -40
    kill "$server_pid" 2>/dev/null || true
    return 1
}

# Stop server cleanly with force fallback
stop_server() {
    local port=$1
    local pid_file="$2"
    
    local pid=""
    if [ -f "$pid_file" ]; then
        pid=$(cat "$pid_file")
    fi
    
    # Also check via port
    local port_pid
    port_pid=$(lsof -ti:"$port" 2>/dev/null || true)
    if [ -n "$port_pid" ]; then
        pid="$port_pid"
    fi
    
    if [ -n "$pid" ]; then
        # Graceful shutdown via API first
        curl -sf -X POST "http://127.0.0.1:$port/reload" \
            -H "Authorization: Bearer $API_KEY" \
            -d '{"unload_model_only": true}' >/dev/null 2>&1 || true
        
        sleep 2
        
        # SIGTERM
        kill "$pid" 2>/dev/null || true
        sleep 3
        
        # SIGKILL if still alive
        if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null || true
            sleep 1
        fi
    fi
    
    cleanup_ports "$port"
    rm -f "$pid_file"
}

# Single-prompt test (long prompt, sequential)
test_single_prompt() {
    local config_name="$1"
    local port="$2"
    local predict="$3"
    local result_file="$LOG_DIR/${config_name}.results.json"
    
    local start_time=$(date +%s%N)
    
    local response
    response=$(curl -sf --max-time "$PROMPT_TTF_TIMEOUT" \
        -X POST "http://127.0.0.1:$port/completion" \
        -H "Authorization: Bearer $API_KEY" \
        -H "Content-Type: application/json" \
        -d "{\"prompt\": $LONG_PROMPT_JSON, \"n_predict\": $predict, \"stream\": false, \"cache_prompt\": false}" \
        2>&1) || {
        echo "  ERROR: Request failed or timed out (${PROMPT_TTF_TIMEOUT}s)"
        echo '{"status":"error","error":"request_timeout"}' > "$result_file"
        return 1
    }
    
    local end_time=$(date +%s%N)
    local duration_ms=$(( (end_time - start_time) / 1000000 ))
    
    # Extract server-side timing from response
    local prompt_tokens eval_tokens prompt_ms eval_ms total_ms
    prompt_tokens=$(echo "$response" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('timings',{}).get('prompt_n',0))" 2>/dev/null || echo 0)
    eval_tokens=$(echo "$response" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('timings',{}).get('eval_n',0))" 2>/dev/null || echo 0)
    prompt_ms=$(echo "$response" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('timings',{}).get('prompt_ms',0))" 2>/dev/null || echo 0)
    eval_ms=$(echo "$response" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('timings',{}).get('eval_ms',0))" 2>/dev/null || echo 0)
    total_ms=$(echo "$response" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('timings',{}).get('total_ms',0))" 2>/dev/null || echo 0)
    
    local prompt_tps eval_tps
    if [ "$prompt_ms" != "0" ] && [ "$prompt_tokens" != "0" ]; then
        prompt_tps=$(python3 -c "print(f'{1000.0 * $prompt_tokens / $prompt_ms:.2f}')")
    else
        prompt_tps="0.00"
    fi
    if [ "$eval_ms" != "0" ] && [ "$eval_tokens" != "0" ]; then
        eval_tps=$(python3 -c "print(f'{1000.0 * $eval_tokens / $eval_ms:.2f}')")
    else
        eval_tps="0.00"
    fi
    
    # Parse server log for detailed timing
    local log_prompt_tps log_eval_tps
    log_prompt_tps=$(grep 'prompt eval time' "$LOG_DIR/${config_name%_*}_*.log" 2>/dev/null | tail -1 || echo "")
    log_eval_tps=$(grep 'eval time' "$LOG_DIR/${config_name%_*}_*.log" 2>/dev/null | tail -1 || echo "")
    
    cat > "$result_file" << EOF
{
  "status": "ok",
  "duration_ms": $duration_ms,
  "prompt_tokens": $prompt_tokens,
  "eval_tokens": $eval_tokens,
  "prompt_ms": $prompt_ms,
  "eval_ms": $eval_ms,
  "prompt_tps": $prompt_tps,
  "eval_tps": $eval_tps,
  "server_prompt_line": $(echo "$log_prompt_tps" | python3 -c "import sys,json; print(json.dumps(sys.stdin.read().strip()))" 2>/dev/null || echo '""'),
  "server_eval_line": $(echo "$log_eval_tps" | python3 -c "import sys,json; print(json.dumps(sys.stdin.read().strip()))" 2>/dev/null || echo '""')
}
EOF
    
    echo "    Prompt: $prompt_tokens tokens in ${prompt_ms}ms = ${prompt_tps} t/s"
    echo "    Generate: $eval_tokens tokens in ${eval_ms}ms = ${eval_tps} t/s"
    echo "    Total: ${total_ms}ms"
}

# Concurrent multi-user test (-np slots, each user sends same long prompt)
test_concurrent() {
    local config_name="$1"
    local port="$2"
    local n_users="$3"
    local predict="$4"
    local result_file="$LOG_DIR/${config_name}.results.json"
    
    echo "    Sending $n_users concurrent requests..."
    
    local start_time=$(date +%s%N)
    
    # Launch all requests in parallel using background curl + file descriptors
    local tmpdir=$(mktemp -d)
    local pids=()
    
    for i in $(seq 1 "$n_users"); do
        curl -sf --max-time "$CONCURRENT_REQUEST_TIMEOUT" \
            -X POST "http://127.0.0.1:$port/completion" \
            -H "Authorization: Bearer $API_KEY" \
            -H "Content-Type: application/json" \
            -d "{\"prompt\": $LONG_PROMPT_JSON, \"n_predict\": $predict, \"stream\": false, \"cache_prompt\": false}" \
            > "$tmpdir/user_$i.json" 2>&1 &
        pids+=($!)
    done
    
    # Wait for all to complete (or timeout)
    local all_ok=1
    for i in $(seq 0 $((n_users - 1))); do
        if ! wait "${pids[$i]}" 2>/dev/null; then
            echo "    User $((i+1)): request failed or timed out"
            all_ok=0
        fi
    done
    
    local end_time=$(date +%s%N)
    local wall_ms=$(( (end_time - start_time) / 1000000 ))
    
    # Aggregate results
    local total_prompt_tokens=0 total_eval_tokens=0
    local total_prompt_ms=0 total_eval_ms=0
    local max_latency=0 min_latency=999999999
    local completed=0
    
    for i in $(seq 1 "$n_users"); do
        local f="$tmpdir/user_$i.json"
        if [ -f "$f" ] && grep -q '"timings"' "$f" 2>/dev/null; then
            local pt et pm em tm
            pt=$(python3 -c "import json; d=json.load(open('$f')); print(d.get('timings',{}).get('prompt_n',0))" 2>/dev/null || echo 0)
            et=$(python3 -c "import json; d=json.load(open('$f')); print(d.get('timings',{}).get('eval_n',0))" 2>/dev/null || echo 0)
            pm=$(python3 -c "import json; d=json.load(open('$f')); print(d.get('timings',{}).get('prompt_ms',0))" 2>/dev/null || echo 0)
            em=$(python3 -c "import json; d=json.load(open('$f')); print(d.get('timings',{}).get('eval_ms',0))" 2>/dev/null || echo 0)
            tm=$(python3 -c "import json; d=json.load(open('$f')); print(d.get('timings',{}).get('total_ms',0))" 2>/dev/null || echo 0)
            
            total_prompt_tokens=$((total_prompt_tokens + pt))
            total_eval_tokens=$((total_eval_tokens + et))
            total_prompt_ms=$((total_prompt_ms + pm))
            total_eval_ms=$((total_eval_ms + em))
            
            if [ "$tm" -gt "$max_latency" ] 2>/dev/null; then max_latency=$tm; fi
            if [ "$tm" -lt "$min_latency" ] 2>/dev/null; then min_latency=$tm; fi
            completed=$((completed + 1))
        fi
    done
    
    rm -rf "$tmpdir"
    
    local agg_prompt_tps agg_eval_tps per_user_prompt_tps per_user_eval_tps
    if [ "$total_prompt_ms" -gt 0 ]; then
        agg_prompt_tps=$(python3 -c "print(f'{1000.0 * $total_prompt_tokens / $total_prompt_ms:.2f}')")
    else
        agg_prompt_tps="0.00"
    fi
    if [ "$total_eval_ms" -gt 0 ]; then
        agg_eval_tps=$(python3 -c "print(f'{1000.0 * $total_eval_tokens / $total_eval_ms:.2f}')")
    else
        agg_eval_tps="0.00"
    fi
    if [ "$total_prompt_ms" -gt 0 ] && [ "$completed" -gt 0 ]; then
        per_user_prompt_tps=$(python3 -c "print(f'{1000.0 * $total_prompt_tokens / $total_prompt_ms / $completed:.2f}')")
    else
        per_user_prompt_tps="0.00"
    fi
    if [ "$total_eval_ms" -gt 0 ] && [ "$completed" -gt 0 ]; then
        per_user_eval_tps=$(python3 -c "print(f'{1000.0 * $total_eval_tokens / $total_eval_ms / $completed:.2f}')")
    else
        per_user_eval_tps="0.00"
    fi
    
    cat > "$result_file" << EOF
{
  "status": "$([ "$all_ok" -eq 1 ] && echo "ok" || echo "partial")",
  "n_users": $n_users,
  "completed": $completed,
  "wall_ms": $wall_ms,
  "total_prompt_tokens": $total_prompt_tokens,
  "total_eval_tokens": $total_eval_tokens,
  "total_prompt_ms": $total_prompt_ms,
  "total_eval_ms": $total_eval_ms,
  "agg_prompt_tps": $agg_prompt_tps,
  "agg_eval_tps": $agg_eval_tps,
  "per_user_prompt_tps": $per_user_prompt_tps,
  "per_user_eval_tps": $per_user_eval_tps,
  "max_latency_ms": $max_latency,
  "min_latency_ms": $min_latency
}
EOF
    
    echo "    Completed: $completed/$n_users users"
    echo "    Wall clock: ${wall_ms}ms"
    echo "    Aggregate prompt: ${agg_prompt_tps} t/s (per-user: ${per_user_prompt_tps} t/s)"
    echo "    Aggregate eval: ${agg_eval_tps} t/s (per-user: ${per_user_eval_tps} t/s)"
    echo "    Latency range: ${min_latency}ms - ${max_latency}ms"
}

# Run a full test configuration (one GPU setup, one LUT mode, one np)
run_test_config() {
    local config_name="$1"
    local binary="$2"
    local cuda_devs="$3"
    local numa_node="$4"
    local threads="$5"
    local affinity="$6"
    local tensor_split="$7"
    local np="$8"
    local port="$9"
    
    echo ""
    echo "========================================"
    echo "Test: $config_name"
    echo "  GPUs: $cuda_devs | NUMA: $numa_node | Threads: $threads | Slots: -np $np"
    echo "========================================"
    
    local server_name="${config_name}_server"
    
    # Start server
    echo "  Starting server..."
    if ! start_server "$server_name" "$binary" "$cuda_devs" "$numa_node" "$threads" "$affinity" "$tensor_split" "$np" "$port"; then
        echo "  SKIPPED due to server start failure"
        return 1
    fi
    echo "  Server running on port $port"
    
    sleep 2  # Let server settle after load
    
    # Test 1: Single long prompt
    echo "  Test: single-prompt"
    test_single_prompt "${config_name}_single" "$port" "$N_PREDICT"
    
    # Test 2: Concurrent users (only if np > 1)
    if [ "$np" -ge 2 ]; then
        echo "  Test: concurrent-${np}users"
        test_concurrent "${config_name}_concurrent_${np}u" "$port" "$np" "$N_PREDICT"
    fi
    
    # Stop server
    echo "  Stopping server..."
    stop_server "$port" "$LOG_DIR/${server_name}.pid"
    
    return 0
}

# Generate summary report
generate_summary() {
    local summary="$LOG_DIR/summary.txt"
    
    echo ""
    echo "============================================"
    echo "SUMMARY"
    echo "============================================"
    
    echo "" > "$summary"
    echo "RoPE LUT Test Summary" >> "$summary"
    echo "Date: $(date)" >> "$summary"
    echo "Model: $(basename "$MODEL")" >> "$summary"
    echo "Predict: $N_PREDICT tokens" >> "$summary"
    echo "" >> "$summary"
    
    # Single-prompt results
    echo "" >> "$summary"
    echo "=== SINGLE PROMPT RESULTS ===" >> "$summary"
    echo "" >> "$summary"
    printf "%-25s %-12s %-10s %-10s %-10s\n" "CONFIG" "PROMPT_TPS" "EVAL_TPS" "PROMPT_TOK" "DUR_MS" >> "$summary"
    printf "%-25s %-12s %-10s %-10s %-10s\n" "------" "----------" "--------" "----------" "------" >> "$summary"
    
    for f in "$LOG_DIR"/*_single.results.json; do
        [ -f "$f" ] || continue
        local name=$(basename "$f" .results.json | sed 's/_single$//')
        local status prompt_tps eval_tps ptok dur
        status=$(python3 -c "import json; print(json.load(open('$f')).get('status','err'))" 2>/dev/null)
        if [ "$status" = "ok" ]; then
            prompt_tps=$(python3 -c "import json; print(json.load(open('$f')).get('prompt_tps','0'))" 2>/dev/null)
            eval_tps=$(python3 -c "import json; print(json.load(open('$f')).get('eval_tps','0'))" 2>/dev/null)
            ptok=$(python3 -c "import json; print(json.load(open('$f')).get('prompt_tokens','0'))" 2>/dev/null)
            dur=$(python3 -c "import json; print(json.load(open('$f')).get('duration_ms','0'))" 2>/dev/null)
            printf "%-25s %-12s %-10s %-10s %-10s\n" "$name" "$prompt_tps" "$eval_tps" "$ptok" "$dur" >> "$summary"
            printf "  %-25s prompt=%-8s eval=%-8s tok=%-6s dur=%sms\n" "$name" "$prompt_tps" "$eval_tps" "$ptok" "$dur"
        else
            printf "%-25s %-12s\n" "$name" "ERROR" >> "$summary"
            printf "  %-25s ERROR\n" "$name"
        fi
    done
    
    # Concurrent results
    echo "" >> "$summary"
    echo "=== CONCURRENT USER RESULTS ===" >> "$summary"
    echo "" >> "$summary"
    printf "%-25s %-6s %-12s %-10s %-12s %-10s %-12s\n" "CONFIG" "USERS" "AGG_PR_TPS" "AGG_EV_TPS" "USR_PR_TPS" "USR_EV_TPS" "WALL_MS" >> "$summary"
    printf "%-25s %-6s %-12s %-10s %-12s %-10s %-12s\n" "------" "-----" "----------" "---------" "----------" "---------" "-------" >> "$summary"
    
    for f in "$LOG_DIR"/*concurrent*.results.json; do
        [ -f "$f" ] || continue
        local name=$(basename "$f" .results.json)
        local status users agg_pt agg_et usr_pt usr_et wall
        status=$(python3 -c "import json; print(json.load(open('$f')).get('status','err'))" 2>/dev/null)
        if [ "$status" = "ok" ]; then
            users=$(python3 -c "import json; print(json.load(open('$f')).get('n_users','0'))" 2>/dev/null)
            agg_pt=$(python3 -c "import json; print(json.load(open('$f')).get('agg_prompt_tps','0'))" 2>/dev/null)
            agg_et=$(python3 -c "import json; print(json.load(open('$f')).get('agg_eval_tps','0'))" 2>/dev/null)
            usr_pt=$(python3 -c "import json; print(json.load(open('$f')).get('per_user_prompt_tps','0'))" 2>/dev/null)
            usr_et=$(python3 -c "import json; print(json.load(open('$f')).get('per_user_eval_tps','0'))" 2>/dev/null)
            wall=$(python3 -c "import json; print(json.load(open('$f')).get('wall_ms','0'))" 2>/dev/null)
            printf "%-25s %-6s %-12s %-10s %-12s %-10s %-12s\n" "$name" "$users" "$agg_pt" "$agg_et" "$usr_pt" "$usr_et" "$wall" >> "$summary"
            printf "  %-25s users=%-3s agg_pr=%-7s usr_pr=%-7s wall=%sms\n" "$name" "$users" "$agg_pt" "$usr_pt" "$wall"
        else
            printf "%-25s %-6s\n" "$name" "ERROR" >> "$summary"
            printf "  %-25s ERROR\n" "$name"
        fi
    done
    
    # LUT comparison
    echo "" >> "$summary"
    echo "=== LUT ON vs OFF COMPARISON (SINGLE PROMPT) ===" >> "$summary"
    echo "" >> "$summary"
    
    for gpu_config in node0 node1 all8; do
        local off_file="$LOG_DIR/${gpu_config}_lut_OFF_single.results.json"
        local on_file="$LOG_DIR/${gpu_config}_lut_ON_single.results.json"
        
        if [ -f "$off_file" ] && [ -f "$on_file" ]; then
            local off_pr off_ev on_pr on_ev pr_delta ev_delta
            off_pr=$(python3 -c "import json; print(json.load(open('$off_file')).get('prompt_tps','0'))" 2>/dev/null)
            off_ev=$(python3 -c "import json; print(json.load(open('$off_file')).get('eval_tps','0'))" 2>/dev/null)
            on_pr=$(python3 -c "import json; print(json.load(open('$on_file')).get('prompt_tps','0'))" 2>/dev/null)
            on_ev=$(python3 -c "import json; print(json.load(open('$on_file')).get('eval_tps','0'))" 2>/dev/null)
            
            if [ "$off_pr" != "0" ] && [ "$on_pr" != "0" ]; then
                pr_delta=$(python3 -c "print(f'{(float(\"$on_pr\")/float(\"$off_pr\")-1)*100:.1f}%')")
                ev_delta=$(python3 -c "print(f'{(float(\"$on_ev\")/float(\"$off_ev\")-1)*100:.1f}%')")
                echo "$gpu_config: prompt_tps ${off_pr}→${on_pr} ($pr_delta), eval_tps ${off_ev}→${on_ev} ($ev_delta)" >> "$summary"
                echo "  $gpu_config: prompt ${off_pr}→${on_pr} ($pr_delta), eval ${off_ev}→${on_ev} ($ev_delta)"
            fi
        fi
    done
    
    echo ""
    echo "Full summary: $summary"
    echo "All logs: $LOG_DIR"
}

# MAIN
echo "RoPE LUT Comprehensive Test Suite"
echo "================================="
echo "Model: $MODEL"
echo "Predict: $N_PREDICT tokens per request"
echo "Logs: $LOG_DIR"
echo ""

# Generate long prompt and escape for JSON
echo "Generating long test prompt..."
LONG_PROMPT=$(generate_long_prompt 32000)
LONG_PROMPT_JSON=$(python3 -c "import sys,json; print(json.dumps(sys.stdin.read()))" <<< "$LONG_PROMPT")
prompt_tok_est=$(python3 -c "import sys; print(len(sys.stdin.read().split()))" <<< "$LONG_PROMPT")
echo "  Prompt ~${prompt_tok_est} words (~$((prompt_tok_est * 1.3)) estimated tokens)"

# Build variants
echo ""
echo "Building LUT variants..."
BINARY_OFF=$(build_variant "off" 0)
BINARY_ON=$(build_variant "on" 1)

# Test matrix
echo ""
echo "Test matrix: 2 LUT modes × 3 GPU configs × 3 slot counts = 18 test configs"
echo ""

PORT=$START_PORT
TESTS_PASSED=0
TESTS_FAILED=0

# GPU Config 1: Node 0 only (GPUs 0-3, 18 threads)
for np in 1 2 3; do
    for lut in OFF ON; do
        local_binary=$([ "$lut" = "OFF" ] && echo "$BINARY_OFF" || echo "$BINARY_ON")
        local_name="node0_lut_${lut}_np${np}"
        
        if run_test_config "$local_name" "$local_binary" "0,1,2,3" "0" "18" "0-17,36-53" "1,1,1,1" "$np" "$PORT"; then
            TESTS_PASSED=$((TESTS_PASSED + 1))
        else
            TESTS_FAILED=$((TESTS_FAILED + 1))
        fi
        PORT=$((PORT + 1))
    done
done

# GPU Config 2: Node 1 only (GPUs 4-7, 18 threads)
for np in 1 2 3; do
    for lut in OFF ON; do
        local_binary=$([ "$lut" = "OFF" ] && echo "$BINARY_OFF" || echo "$BINARY_ON")
        local_name="node1_lut_${lut}_np${np}"
        
        if run_test_config "$local_name" "$local_binary" "4,5,6,7" "1" "18" "18-35,54-71" "1,1,1,1" "$np" "$PORT"; then
            TESTS_PASSED=$((TESTS_PASSED + 1))
        else
            TESTS_FAILED=$((TESTS_FAILED + 1))
        fi
        PORT=$((PORT + 1))
    done
done

# GPU Config 3: All 8 GPUs (36 threads)
for np in 1 2 3; do
    for lut in OFF ON; do
        local_binary=$([ "$lut" = "OFF" ] && echo "$BINARY_OFF" || echo "$BINARY_ON")
        local_name="all8_lut_${lut}_np${np}"
        
        if run_test_config "$local_name" "$local_binary" "0,1,2,3,4,5,6,7" "0" "36" "0-17,36-53,18-35,54-71" "1,1,1,1,1,1,1,1" "$np" "$PORT"; then
            TESTS_PASSED=$((TESTS_PASSED + 1))
        else
            TESTS_FAILED=$((TESTS_FAILED + 1))
        fi
        PORT=$((PORT + 1))
    done
done

# Generate summary
generate_summary

echo ""
echo "Tests passed: $TESTS_PASSED, failed: $TESTS_FAILED"
echo "All logs in: $LOG_DIR"
