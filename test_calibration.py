#!/usr/bin/env python3
"""Calibration test: sends prompts to the local server via curl and records timing stats."""

import json
import subprocess
import time

SERVER = "http://127.0.0.1:8430"
ENDPOINT = f"{SERVER}/v1/chat/completions"

PROMPTS = [
    "What is 2+2? Answer with just the number.",
    "Solve: 17 × 23. Show your work step by step.",
    "If a train travels 120 km in 2 hours, what is its average speed in m/s?",
    "What is the derivative of x³ + 2x² - 5x + 3?",
    "A box has 3 red, 5 blue, and 2 green balls. What's the probability of picking blue?",
    "What is the sum of the first 100 positive integers?",
    "Convert 0.375 to a fraction in simplest form.",
    "If f(x) = 2x + 1 and g(x) = x², what is f(g(3))?",
]

results = []

for i, prompt in enumerate(PROMPTS):
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
    }

    body = json.dumps(payload)
    print(f"[{i}] Sending: {prompt[:60]}...")

    t0 = time.perf_counter()
    proc = subprocess.run(
        ["curl", "-s", "-X", "POST", ENDPOINT,
         "-H", "Content-Type: application/json",
         "-d", body],
        capture_output=True, text=True, timeout=120,
    )
    t1 = time.perf_counter()

    if proc.returncode != 0:
        print(f"    ✗ curl failed: {proc.stderr}")
        continue

    data = json.loads(proc.stdout)

    choice = data["choices"][0]
    content = choice["message"]["content"]
    usage = data.get("usage", {})
    completion_tokens = usage.get("completion_tokens", 0)

    elapsed = round(t1 - t0, 2)
    tok_s = round(completion_tokens / elapsed, 1) if elapsed > 0 else 0

    entry = {
        "id": i,
        "prompt": prompt,
        "response": content,
        "tokens": completion_tokens,
        "time_s": elapsed,
        "tok_per_sec": tok_s,
    }
    results.append(entry)
    print(f"    → {completion_tokens} tokens in {elapsed}s ({tok_s} tok/s)")

with open("calibration.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\nWrote {len(results)} entries to calibration.json")
