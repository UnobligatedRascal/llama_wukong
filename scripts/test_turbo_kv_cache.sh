#!/bin/bash
# Test turbo2_0/turbo3_0 KV cache with tensor-split mode
# Uses the existing model but with minimal context to verify no crash

set -e

LLAMA_SERVER="/home/whistler/llama_wukong/build-turbo-fix/bin/llama-server"
MODEL="/mnt/512gb_ssd/models/Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-MTP-Q4_K_M.gguf"
MMPROJ="/mnt/512gb_ssd/models/Qwen3.6-27B-mmproj-F16.gguf"
PORT=4280
TIMEOUT=120

echo "=== Testing turbo KV cache fix ==="
echo "Binary: $LLAMA_SERVER"
echo "Port: $PORT"

# Clean up any existing process on test port
pkill -f "llama-server.*--port $PORT" 2>/dev/null || true
sleep 2

test_cache_type() {
    local CACHE_K="$1"
    local CACHE_V="$2"
    local TEST_NAME="turbo K=$CACHE_K V=$CACHE_V"
    
    echo ""
    echo "=========================================="
    echo "TEST: $TEST_NAME"
    echo "=========================================="
    
    # Start server in background
    sudo GGML_CUDA_P2P=1 -E nice -n -20 numactl --interleave=all \
        $LLAMA_SERVER \
        -m "$MODEL" -t 36 -c 4096 -ngl 99 \
        --port $PORT --host 127.0.0.1 --api-key test \
        --jinja --chat-template-file /home/whistler/models/tuvak.jinja \
        --load-mode none -np 4 \
        --batch-size 512 --ubatch-size 256 \
        --cache-type-k "$CACHE_K" --cache-type-v "$CACHE_V" \
        --tensor-split 1,1,1,1,1,1,1,1 --kv-unified \
        --seed 42 --spec-type draft-mtp --spec-draft-p-min 0.75 --spec-draft-n-max 3 \
        --split-mode tensor \
        > /tmp/llama-server-test-$PORT.log 2>&1 &
    
    SERVER_PID=$!
    echo "Server PID: $SERVER_PID"
    
    # Wait for server to start
    echo "Waiting for server to start..."
    for i in $(seq 1 60); do
        if curl -s http://127.0.0.1:$PORT/health > /dev/null 2>&1; then
            echo "Server started successfully"
            break
        fi
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo "SERVER CRASHED! Checking logs..."
            tail -50 /tmp/llama-server-test-$PORT.log
            return 1
        fi
        sleep 1
    done
    
    # Check if server is still running
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "SERVER FAILED TO START! Checking logs..."
        tail -100 /tmp/llama-server-test-$PORT.log
        return 1
    fi
    
    # Send a simple completion request
    echo "Sending test completion request..."
    RESPONSE=$(curl -s --max-time 60 \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer test" \
        -d '{
            "model": "test",
            "prompt": "Hello, how are you?",
            "max_tokens": 20,
            "temperature": 0.7
        }' \
        http://127.0.0.1:$PORT/v1/completions 2>&1) || true
    
    # Check response
    if echo "$RESPONSE" | grep -q "error"; then
        echo "ERROR in response: $RESPONSE"
    else
        echo "SUCCESS: Got response (truncated): ${RESPONSE:0:200}..."
    fi
    
    # Check for assertion failures in logs
    if grep -q "GGML_ASSERT.*failed" /tmp/llama-server-test-$PORT.log; then
        echo "ASSERTION FAILURE DETECTED!"
        grep "GGML_ASSERT" /tmp/llama-server-test-$PORT.log
        kill $SERVER_PID 2>/dev/null || true
        return 1
    fi
    
    # Clean up
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    sleep 1
    
    echo "TEST PASSED: $TEST_NAME"
    return 0
}

# Test configurations
FAILED=0

# Test turbo4_0 (should work - baseline)
test_cache_type "turbo4_0" "turbo4_0" || FAILED=$((FAILED + 1))

# Test turbo3_0 (was failing before fix)
test_cache_type "turbo3_0" "turbo3_0" || FAILED=$((FAILED + 1))

# Test turbo2_0 (was failing before fix)
test_cache_type "turbo2_0" "turbo2_0" || FAILED=$((FAILED + 1))

# Test mixed types
test_cache_type "q8_0" "turbo3_0" || FAILED=$((FAILED + 1))

echo ""
echo "=========================================="
if [ $FAILED -eq 0 ]; then
    echo "ALL TESTS PASSED!"
else
    echo "$FAILED TEST(S) FAILED!"
fi
echo "=========================================="

exit $FAILED
