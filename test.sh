#!/usr/bin/env bash
# test.sh - run a few example queries against the bonsai mlx_lm server
set -euo pipefail

BASE_URL="${BONSAI_URL:-http://127.0.0.1:8430}"
PASS=0
FAIL=0

run_test() {
    local name="$1"
    local endpoint="$2"
    local payload="$3"

    printf "\n-- %s --\n" "$name"
    HTTP_CODE=$(curl -s -o /tmp/bonsai_test_resp.json -w "%{http_code}" \
        -X POST "$BASE_URL$endpoint" \
        -H "Content-Type: application/json" \
        -d "$payload")

    if [[ "$HTTP_CODE" -ge 200 && "$HTTP_CODE" -lt 300 ]]; then
        printf "  [OK] HTTP %s\n" "$HTTP_CODE"
        python3 -m json.tool /tmp/bonsai_test_resp.json 2>/dev/null | head -30
        PASS=$((PASS + 1))
    else
        printf "  [X] HTTP %s\n" "$HTTP_CODE"
        cat /tmp/bonsai_test_resp.json 2>/dev/null
        FAIL=$((FAIL + 1))
    fi
}

# -- List models (health check) -----------
printf "\n-- List models --\n"
HTTP_CODE=$(curl -s -o /tmp/bonsai_test_resp.json -w "%{http_code}" "$BASE_URL/v1/models")
if [[ "$HTTP_CODE" == "200" ]]; then
    printf "  [OK] HTTP %s\n" "$HTTP_CODE"
    python3 -m json.tool /tmp/bonsai_test_resp.json 2>/dev/null
    PASS=$((PASS + 1))
else
    printf "  [X] HTTP %s\n" "$HTTP_CODE"
    FAIL=$((FAIL + 1))
fi

# -- Chat completions ---------------------
run_test "Chat: Explain quantum computing" "/v1/chat/completions" '{
    "messages": [{"role": "user", "content": "Explain quantum computing in simple terms."}],
    "max_tokens": 128
}'

run_test "Chat: Write a haiku" "/v1/chat/completions" '{
    "messages": [{"role": "user", "content": "Write a haiku about programming."}],
    "max_tokens": 64
}'

run_test "Chat: System + user prompt" "/v1/chat/completions" '{
    "messages": [
        {"role": "system", "content": "You are a helpful assistant that responds concisely."},
        {"role": "user", "content": "What is the capital of France?"}
    ],
    "max_tokens": 64
}'

run_test "Chat: Code question" "/v1/chat/completions" '{
    "messages": [{"role": "user", "content": "Write a Python function that returns the nth Fibonacci number."}],
    "max_tokens": 128
}'

run_test "Chat: Creative" "/v1/chat/completions" '{
    "messages": [{"role": "user", "content": "Write a limerick about a program that never compiles."}],
    "max_tokens": 64
}'

# -- Streaming chat completion ------------
printf "\n-- Chat: Streaming --\n"
STREAM_OUTPUT=$(curl -sN -X POST "$BASE_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"messages": [{"role": "user", "content": "Count from 1 to 5."}], "max_tokens": 64, "stream": true}' \
    --max-time 30 2>/dev/null)

# Check we got SSE data lines
CHUNK_COUNT=$(echo "$STREAM_OUTPUT" | grep -c "^data: " || true)
HAS_DONE=$(echo "$STREAM_OUTPUT" | grep -c "data: \[DONE\]" || true)

if [[ "$CHUNK_COUNT" -gt 1 && "$HAS_DONE" -ge 1 ]]; then
    printf "  [OK] Streaming: %d chunks received, [DONE] present\n" "$CHUNK_COUNT"
    PASS=$((PASS + 1))
else
    printf "  [X] Streaming: got %d chunks, [DONE]=%d\n" "$CHUNK_COUNT" "$HAS_DONE"
    echo "$STREAM_OUTPUT" | head -5
    FAIL=$((FAIL + 1))
fi

# -- Summary ------------------------------
printf "\n==============================\n"
printf "  Results: %d passed, %d failed\n" "$PASS" "$FAIL"
printf "==============================\n"

[[ "$FAIL" -eq 0 ]]
