# MLX Metal Port of AutoKernel — Validated Implementation Plan

## Bottom line

Your hotspot intuition is **directionally correct**, but the current draft is **not fully correct as written**.

### What is correct

- **Affine quantized matmul is the Tier‑1 target.** For MLX LLM inference, the highest leverage work is still the quantized projection/MLP stack. The public API exposes this as `mlx.core.quantized_matmul(...)`, and MLX’s own decode-path discussion explicitly calls out `affine_qmv_fast` and `sdpa_vector` as already-optimized kernels whose gains are now partly limited by dispatch overhead rather than raw arithmetic throughput.
- **RMSNorm, RoPE, and SDPA are the next decode-critical paths.**
- **A dispatch file must be treated as part of the optimization surface**, not just the `.metal` and `.h` files. In MLX, the CPU/C++ dispatch logic decides whether you actually hit `qmv_fast`, `qmv_quad`, `qmm`, `qvm`, `qvm_split_k`, NAX, and the SDPA vector/full paths.

### What is incorrect / incomplete in the current draft

1. **`quantized_utils.h` is not the packing-constants file in this tree.**  
   In the uploaded source, `get_pack_factor()` and `get_bytes_per_pack()` live in `quantized.h` and `quantized_nax.h`. `quantized_utils.h` contains the GEMM-loop helpers (`gemm_loop_aligned`, `gemm_loop_unaligned`, `gemm_loop_finalize`) used by the QMM-family kernels.

2. **`quantized.cpp` must be in Tier‑1, not optional.**  
   This file decides:
   - `qmv_fast` vs `qmv`
   - `qmv_quad`
   - `qmm`
   - `qvm`
   - `qvm_split_k`
   - NAX routing for `qmm`
   If you ignore it, you can optimize a kernel that the runtime almost never dispatches.

3. **The `qmv_quad` routing condition is narrower than “K ∈ {64,128}”.**  
   In the uploaded source it is:
   - `K == 128`, or
   - `K == 64 and bits >= 2`, and
   - `bits` must be a power of two.

4. **NAX is not “preferred on capable hardware” for all QMM paths.**  
   In the uploaded source, NAX QMM dispatch is gated by:
   - NAX availability,
   - `transpose == true`,
   - `K % 64 == 0`,
   - and a dtype / TF32 condition.

5. **`qvm_split_k` is missing from your Tier‑1/Tier‑3 list.**  
   It is part of the actual non-transposed routing for large‑K cases and should be in the manifest.

6. **`sdpa_vector.h` is missing from your SDPA target set.**  
   In this tree the vector attention kernels live there; `scaled_dot_product_attention.metal` is mostly instantiation glue.

7. **`softmax.h` / `softmax.metal` are not “used inside SDPA” here.**  
   In the current MLX Metal source, the SDPA path is self-contained. The standalone softmax kernels are still important for explicit softmax ops, but they are lower ROI for your stated LLM hot path than you claimed.

8. **The bit-width set in the uploaded tree is broader than `{1,3,4,8}`.**  
   The uploaded source instantiates affine quantized kernels for **1, 2, 3, 4, 5, 6, 8** bits, each for group sizes **32, 64, 128**.

9. **The “single highest-impact function” statement is close but too narrow.**  
   `qdot()` is one of the hottest inner loops, but for the real port the highest-impact *optimization surface* is:
   - `quantized.cpp` dispatch,
   - `qdot()` / `qdot_safe()` inner loops,
   - `QuantizedBlockLoader` + GEMM loop helpers for QMM,
   - and `qvm_split_k` for non-transposed large‑K workloads.

---

## Source-validated hotspot hierarchy

This hierarchy is validated against:

- the uploaded `metal.zip` source tree,
- the uploaded `agents.md` AutoKernel instructions,
- current MLX docs,
- and current MLX repo / issue state as of 2026‑04‑06.

## Tier 1 — Must optimize first

### 1) Quantized affine matmul dispatch + kernels

**Files**
- `mlx/backend/metal/quantized.cpp`
- `mlx/backend/metal/kernels/quantized.h`
- `mlx/backend/metal/kernels/quantized.metal`
- `mlx/backend/metal/kernels/quantized_nax.h`
- `mlx/backend/metal/kernels/quantized_nax.metal`
- `mlx/backend/metal/kernels/quantized_utils.h`

**Why**
This is the dominant surface for quantized projection and MLP matmuls. The runtime dispatches:
- decode-style `qmv_fast` / `qmv` / `qmv_quad`,
- prefill-style `qmm`,
- non-transposed `qvm`,
- and large-K `qvm_split_k`.

**Functions / units to treat as hot**
- `dispatch_qmv(...)`
- `QuantizedMatmul::eval_gpu(...)`
- `qmv_fast_impl()`
- `qmv_impl()`
- `qmv_quad_impl()`
- `qvm_impl()`
- `qvm_split_k` path
- `qmm_t_impl()`
- `qmm_n_impl()`
- `qdot()`
- `qdot_safe()`
- `qouter()`
- `dequantize()`
- `load_vector()`
- `load_vector_safe()`
- `QuantizedBlockLoader`
- `gemm_loop_aligned()`
- `gemm_loop_unaligned()`
- `gemm_loop_finalize()`

**Validated dispatch facts**
- `qmv_fast` only if `N % 8 == 0 && K % 512 == 0`.
- `qmv_quad` only if `((K == 128) || (K == 64 && bits >= 2)) && is_power_of_2(bits)`.
- `qmm` is selected when `M >= vector_limit`.
- If `transpose == false` and `M < vector_limit`:
  - `qvm` is used when `K < 1024`
  - otherwise `qvm_split_k` is used.
- NAX QMM is gated, not universal.

### 2) SDPA vector/full dispatch + vector kernels

**Files**
- `mlx/backend/metal/scaled_dot_product_attention.cpp`
- `mlx/backend/metal/kernels/sdpa_vector.h`
- `mlx/backend/metal/kernels/scaled_dot_product_attention.metal`

**Why**
This is the next real decode bottleneck after quantized matmul. MLX’s own public fast path exposes this as `mlx.core.fast.scaled_dot_product_attention(...)`.

**Validated dispatch facts**
- `sdpa_vector` is used only for short-query regimes (`query_sequence_length <= 8`) and supported head dims.
- Vector supported head dims: **64, 96, 128, 256**
- Full supported head dims: **64, 80, 128**
- Vector and full dispatch predicates are distinct and must be benchmarked separately.

---

## Tier 2 — Important decode kernels

### 3) RMSNorm

**Files**
- `mlx/backend/metal/normalization.cpp`
- `mlx/backend/metal/kernels/rms_norm.metal`

**Why**
RMSNorm is frequent and cheap enough that dispatch overhead matters at decode.

**Validated dispatch fact**
- MLX switches between single-row and looped RMS kernels around `RMS_LOOPED_LIMIT`.

### 4) RoPE

**Files**
- `mlx/backend/metal/rope.cpp`
- `mlx/backend/metal/kernels/rope.metal`

**Why**
RoPE runs every layer on Q and K and has distinct inference-specific routing.

**Validated dispatch facts**
- There is a **single-token special case** (`T == 1`, contiguous, scalar offset).
- There are `with_freqs` and `large` variants.
- Non-contiguous 4D head/sequence-transposed inputs get special handling.

---

## Tier 3 — Important only if your traces confirm them

### 5) GEMV / final logits / tiny-matmul path

**Files**
- `mlx/backend/metal/matmul.cpp`
- `mlx/backend/metal/kernels/gemv.metal`

**Why**
Useful for:
- unquantized embeddings / logits heads,
- narrow/tall projection tails,
- fallback paths where `std::min(M, N) == 1`.

### 6) Gathered / masked GEMV

**Files**
- `mlx/backend/metal/kernels/gemv_masked.h`
- `mlx/backend/metal/kernels/gemv_masked.metal`

**Why**
Only if your real model path hits the gather / mask variants enough to matter.

### 7) Standalone softmax

**Files**
- `mlx/backend/metal/softmax.cpp`
- `mlx/backend/metal/kernels/softmax.h`
- `mlx/backend/metal/kernels/softmax.metal`

**Why**
Still useful for explicit softmax operators, but **not** a top decode priority for the path you described.

### 8) Hadamard

**Files**
- `mlx/backend/metal/hadamard.cpp`
- `mlx/backend/metal/kernels/hadamard.h`

**Why**
Only if your rotation/rotq pipeline really uses it in inference.

---

## Tier 4 — Optional / model-dependent

### 9) LayerNorm

**Files**
- `mlx/backend/metal/kernels/layer_norm.metal`

**Why**
Only if the target model actually uses LayerNorm instead of RMSNorm.

---

## Recommended final target manifest

This is the order I would hand to an agent.

| Priority | Target | Why now |
|---|---|---|
| P0 | `quantized_dispatch_decode` | decode throughput, `qmv_fast/qmv/qmv_quad`, dispatch correctness |
| P0 | `quantized_dispatch_prefill` | prefill throughput, `qmm_t/qmm_n`, dispatch correctness |
| P0 | `quantized_qdot_innerloop` | hottest arithmetic loop |
| P0 | `quantized_qmm_loader_pipeline` | most leverage for batched/prefill |
| P1 | `sdpa_vector_decode` | short-query decode attention |
| P1 | `sdpa_vector_2pass_long_kv` | long-context / GQA decode |
| P1 | `rms_norm_decode` | frequent, bandwidth-bound |
| P1 | `rope_single_token` | per-layer decode overhead |
| P2 | `qvm_split_k_large_k` | non-transposed large-K cases |
| P2 | `gemv_logits` | if logits / head path matters |
| P3 | `softmax_standalone` | only if trace confirms |
| P3 | `layer_norm_optional` | only if model needs it |
| P3 | `hadamard_optional` | only if rotq path needs it |

---

## Port architecture: how AutoKernel must change

The current AutoKernel workflow is built around **PyTorch profiling + extracted Triton/CUDA kernels + a single editable `kernel.py` + fixed `bench.py`**. That is explicit in the uploaded `agents.md`. MLX does not fit that model directly.

### Therefore the correct port is **not**
“replace Triton kernels with MLX kernels and keep everything else unchanged.”

### The correct port is
Add a **backend abstraction** and make MLX a **source-tree backend** with its own:
- target manifest,
- build step,
- benchmark harness,
- correctness references,
- and verification flow.

## The minimal viable MLX backend

### A. Backend abstraction
Add a backend interface with:
- `profile()`
- `enumerate_targets()`
- `build()`
- `benchmark_target()`
- `verify_model()`

For MLX, `enumerate_targets()` should come from a **hand-written manifest**, not from PyTorch extraction.

### B. Replace `kernel.py` with a target-oriented edit loop
For MLX, a “kernel” is a **set of source files**, often both:
- a C++ dispatch file, and
- one or more `.h/.metal` files.

So the editable unit becomes:
- `target.quantized_dispatch_decode`
- `target.sdpa_vector_decode`
- etc.

### C. Replace `bench.py` with `bench_mlx.py`
This benchmark must call **public MLX ops**, not private C++ symbols:
- `mlx.core.quantized_matmul`
- `mlx.core.fast.rms_norm`
- `mlx.core.fast.rope`
- `mlx.core.fast.scaled_dot_product_attention`

This gives you a stable benchmark surface while editing the MLX backend underneath.

### D. Add an MLX rebuild step
Each experiment needs:
1. patch source
2. rebuild MLX / metallib
3. run `bench_mlx.py`
4. keep or revert

### E. Keep AutoKernel’s decision logic
The 5-stage correctness loop, keep/revert policy, and Amdahl scheduling are still valuable. They just need an MLX-specific benchmark adapter.

---

## Concrete implementation plan

## Phase 0 — Lock scope and validation rules

### Deliverable
A frozen MLX target manifest.

### Acceptance criteria
- Every target maps to a public MLX benchmark surface.
- Every target maps to exact source files.
- Every target has explicit dispatch predicates.

### Notes
Do **not** start by adding generic MLX model profiling. First get a stable target-driven loop working.

---

## Phase 1 — Add an MLX backend to AutoKernel

### Work
1. Add `autokernel/backends/mlx/`
2. Define:
   - `MLXBackendAdapter`
   - `MLXTarget`
   - `MLXBuildConfig`
3. Encode the validated target manifest.

### Acceptance criteria
- `python -m autokernel_mlx.backend --list-targets` prints the manifest.
- Each target resolves to source files and benchmark entry points.

---

## Phase 2 — Build / rebuild automation

### Work
1. Add a build driver that can:
   - configure CMake if needed,
   - rebuild MLX,
   - optionally rebuild only relevant targets if your checkout supports it,
   - surface stderr/stdout cleanly.
2. Cache source hashes to avoid redundant rebuilds.

### Acceptance criteria
- One command rebuilds the edited checkout.
- Failed compile output is captured and persisted.
- No benchmark runs against stale binaries.

### Practical rule
Treat the build as part of correctness. A compile failure is a failed experiment.

---

## Phase 3 — MLX benchmark harness

### Work
Create `bench_mlx.py` that mirrors AutoKernel’s philosophy:
- smoke test
- shape sweep
- numerical stability
- determinism
- edge cases
- performance

### Public MLX benchmark surfaces
- `mlx.core.quantized_matmul(...)`
- `mlx.core.dequantize(...)` + `mlx.core.matmul(...)` for reference
- `mlx.core.fast.rms_norm(...)`
- `mlx.core.fast.rope(...)`
- `mlx.core.fast.scaled_dot_product_attention(...)`

### Acceptance criteria
- Each benchmark writes a TSV row compatible with AutoKernel’s existing reporting shape.
- Each benchmark can be run independently by target.

---

## Phase 4 — Model-shape presets for your real workloads

### Work
Create a model-shape module that can generate representative cases for:
- decode
- prefill
- long-context decode

### Required cases

#### Quantized matmul
For each model, at minimum:
- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`
- `gate_proj`
- `up_proj`
- `down_proj`

For each op:
- decode (`M=1`)
- prefill (`M in {32, 128, 512}` or real prompt lengths)

#### RMSNorm
- hidden size only
- decode and prefill rows

#### RoPE
- single-token (`T=1`)
- short prefill (`T=32`)
- long prefill (`T=512` or more)

#### SDPA
- decode attention (`Q=1`, `K/V=context`)
- short prefill
- long-context decode
- GQA / MQA settings matching the real model

### Acceptance criteria
- Shape presets are generated from config, not hand-copied constants.
- Benchmark case generation is deterministic and reproducible.

---

## Phase 5 — Agent loop integration

### Work
Replace the current Triton/CUDA assumptions with target-based MLX execution:
- `next target`
- edit files for that target
- build
- benchmark
- keep/revert
- record results

### Acceptance criteria
- Results land in `results.tsv`
- Existing analysis scripts can still consume the output
- Plateau / revert policy works unchanged

---

## Phase 6 — End-to-end MLX verification

### Work
Create `verify_mlx.py`:
- run real MLX model inference on exactly one of the frozen packs: `E4B`, `E2B`, or `Bonsai`
- compare prompt-by-prompt decoded continuation against the stored baseline `response`
- compare generated token count, elapsed time, tok/s, and memory
- emit a per-prompt replay report and an aggregate summary

### Acceptance criteria
- deterministic decode on the frozen pack
- no unacceptable response drift on any prompt id
- benchmark improvement reproduces at model level on the same pack

---

## Phase 7 — Only after that: optional fused-kernel research

The current MLX issue tracker shows a real decode bottleneck from **kernel dispatch overhead** even when kernels like `affine_qmv_fast` and `sdpa_vector` are already good. That means the long-term upside is probably **fusion across ops**, not only micro-optimizing each kernel.

But this is **Phase 7**, not Phase 1.

---


## Fixed calibration and model-replay protocol

This port will use **only three frozen replay/calibration packs**:
- `calibration_e4b.json`
- `calibration_e2b.json`
- `calibration_bonsai.json`

These packs are the authoritative evaluation inputs for the port. No larger generic eval suite, no open-ended prompt collection, and no dynamic dataset sampling is part of the initial plan.

### Rules

- Run **real model inference** against the exact target models for E4B, E2B, and Bonsai.
- Use the provided prompts exactly as written.
- Keep tokenizer, sampling mode, max token limit, KV/cache mode, and all decode settings fixed per model.
- Use **greedy decode / deterministic decode only** for validation runs.
- Record baseline output, tokens, elapsed time, and tok/s for every prompt id.
- Treat the stored `response` field as the frozen baseline continuation for regression checking.
- Do not widen the calibration protocol until the port is stable.

### What this calibration is used for

1. **Replay validation**  
   Run the real model with the exact prompt pack before and after a kernel change.

2. **Performance validation**  
   Measure per-prompt:
   - total generated tokens
   - total wall time
   - tokens/sec
   - TTFT if available
   - peak memory if available

3. **Output regression detection**  
   Compare against the stored baseline response using:
   - exact token match when possible
   - otherwise exact string/prefix match on the decoded continuation
   - token-count drift

4. **Trace capture**  
   Use these exact prompts for representative GPU trace capture after kept changes.

### Where this fits in the loop

- **Every candidate change**:
  - compile
  - run op-level microbench
  - run op-level correctness

- **Every kept candidate**:
  - run the relevant fixed model replay pack (`E4B`, `E2B`, or `Bonsai`)
  - compare against baseline prompt-by-prompt
  - record throughput deltas

- **Every promoted candidate**:
  - capture one representative Metal trace on the fixed prompt pack
  - run the full fixed pack again from a clean process

### Acceptance criteria for replay validation

A candidate is only eligible to stay kept if all of the following hold for the relevant pack:
- compile succeeds
- op-level correctness passes
- deterministic rerun passes
- no prompt regresses beyond the accepted response-match rule
- aggregate tok/s improves or a trace proves a justified tradeoff on the intended hot path

### Scope discipline

This is a **frozen calibration protocol**, not an attempt at broad model quality evaluation. Its job is to detect inference-path regressions and confirm that kernel-level wins survive inside the real model runtime.

## Validation matrix

A compact decision table for where to spend effort.

| Target | Call frequency | Cost / call | Dispatch leverage | ROI | First action |
|---|---:|---:|---:|---:|---|
| quantized dispatch + qmv/qmm/qvm | 10/10 | 10/10 | 9/10 | **10/10** | benchmark + route validation |
| qdot / qdot_safe | 10/10 | 10/10 | 2/10 | **9/10** | inner-loop microbench |
| QuantizedBlockLoader + GEMM loops | 8/10 | 9/10 | 4/10 | **9/10** | prefill benchmark |
| sdpa_vector / 2pass | 7/10 | 7/10 | 6/10 | **8/10** | decode benchmark |
| rms_norm | 9/10 | 3/10 | 8/10 | **7/10** | decode microbench |
| rope single-token | 9/10 | 2/10 | 8/10 | **7/10** | decode microbench |
| gemv/logits | 3/10 | 5/10 | 4/10 | 4/10 | only if trace confirms |
| standalone softmax | 2/10 | 3/10 | 4/10 | 3/10 | defer |
| layer_norm | model-dependent | model-dependent | model-dependent | 2/10 | defer |
| hadamard | model-dependent | model-dependent | model-dependent | 1/10 | defer |

---

## Bench matrix you should actually run

In addition to the op-level matrix below, every target family must be checked against the matching frozen replay pack:
- E4B target model -> `calibration_e4b.json`
- E2B target model -> `calibration_e2b.json`
- Bonsai target model -> `calibration_bonsai.json`


## Quantized matmul

### Decode
- `M=1`
- `N` / `K` from each projection
- `bits ∈ {1, 3, 4}` for your current stated workloads
- `group_size ∈ {32, 64, 128}` only if the actual model weights use them

### Prefill
- `M ∈ {32, 128, 512}`
- same projection set
- separate transposed and non-transposed cases

### Edge cases
- `K=64`
- `K=128`
- `K=256`
- `K=1024`
- aligned vs non-aligned `N`
- power-of-two vs non-power-of-two bit widths if your checkout uses them

## SDPA
- `Q=1`, `KV=2048`
- `Q=1`, `KV=8192`
- `Q=8`, `KV=8192`
- supported head dims separately: 64 / 96 / 128 / 256
- GQA / MQA separately

## RMSNorm
- hidden sizes from real models
- `rows ∈ {1, 32, 512}`

## RoPE
- `T=1`, scalar offset
- `T=1`, vector offsets
- `T=32`
- `T=512`
- with and without explicit freqs if your model path uses them

---

## Risks and non-negotiable checks

1. **Branch / release drift matters.**  
   MLX is moving quickly. There are recent release and issue notes around NAX, fp8/fp4, and SDPA behavior. Lock the exact MLX commit before starting the agent loop.

2. **Do not optimize against the wrong dispatch branch.**  
   Every benchmark row must record which route actually fired:
   - `qmv_fast`
   - `qmv`
   - `qmv_quad`
   - `qmm_t`
   - `qmm_n`
   - `qmm_t_nax`
   - `qmm_n_nax`
   - `qvm`
   - `qvm_split_k`
   - `sdpa_vector`
   - `sdpa_vector_2pass`
   - `sdpa_full`

3. **Do not assume softmax is part of SDPA.**  
   Treat standalone softmax separately.

4. **Do not assume 1-bit behavior is identical across regular and NAX paths.**  
   In the uploaded source the non-NAX 1-bit path uses `select(...)` patterns; the NAX path differs.

5. **Do not accept source-only wins.**  
   Every “improvement” must pass:
   - compile,
   - op-level correctness,
   - deterministic rerun,
   - end-to-end model verification.

---

## What is included in the handoff bundle

This bundle includes:

- this plan,
- a ready-to-edit MLX target manifest,
- a build driver scaffold,
- a benchmark harness scaffold using public MLX APIs,
- model-shape helpers for decode / prefill case generation,
- and an MLX backend adapter skeleton suitable for merging into AutoKernel.

The code is designed as a **continuation point** for another agent.

### Honest limitation
This bundle is **source-validated**, not runtime-validated in this container. I validated it against the uploaded MLX Metal source and current MLX docs/repo state, but I did **not** build MLX or execute Metal benchmarks here.

---

## Recommended immediate next move

Do these in order:

1. freeze the MLX commit / submodule you will target
2. drop the included `autokernel_mlx` package into the AutoKernel repo
3. wire `bench_mlx.py` into the existing results loop
4. benchmark **quantized decode** first
5. only then iterate on `qdot` / dispatch / loader changes

That is the shortest path to a real working MLX port.
