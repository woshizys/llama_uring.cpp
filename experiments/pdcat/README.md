# PDCat reproducible experiments

PDCat runs a GGUF MoE model whose routed expert bytes remain on NVMe and are
demand-paged through `InterfaceIO/io-scheduler`. DeepSeek-V2-Lite Q8_0 is the
primary model for the complete correctness and performance evaluation. Model
topology, expert tensor mapping, cache capacity, prediction transitions, and
experiment variants are data-driven so another compatible GGUF MoE can be
validated without adding model constants to the cache or scheduler core.

The semantic invariant is strict: the native router chooses the final experts.
Prediction may only move bytes earlier. A prediction miss, late prefetch, budget
rejection, checksum failure, or promotion failure must fall back to the normal
P0 demand path and must not change generated tokens.

The requirement-by-requirement delivery audit is in
`completion-audit-v03.md`; measured results and negative findings are in
`results-v03.md`.

## Repository boundary

- `llama_uring.cpp` owns GGUF/model semantics, router observation, slot-aware
  CPU/CUDA execution, the thin Rust FFI, server P4 adapter, and experiment tools.
- `/workspace/InterfaceIO/io-scheduler` is the source of truth for aligned slot
  memory, generation-safe handles, asynchronous expert loading, P0-P4 admission,
  predictor execution, and unified expert/KV I/O traces.
- `/workspace/EK-Edge` is a vLLM/EK comparison and methodology repository. It is
  not copied into the llama runtime and its historical measurements are not PDCat
  results.

## Versioned inputs

| Input | Path | Identity |
| --- | --- | --- |
| Primary GGUF | `/data/models/DeepSeek-V2-Lite-GGUF/DeepSeek-V2-Lite-Q8_0.gguf` | 16,702,520,096 bytes; SHA-256 `7b131e7fdddd10eeca4d1716832de9aaa60ff2544a94bec13cb34946bec514b0` |
| Run config | `configs/deepseek-v2-lite-q8_0.json` | topology, tensor adapter, profiles, budgets, model/predictor identity |
| Expert pack | `/data/models/DeepSeek-V2-Lite-GGUF/DeepSeek-V2-Lite-Q8_0.pdcat-experts.pack` | 15,294,529,536 bytes; SHA-256 `9a24592f0151c87d9107e996a8896359d6a6639f6fdb22ea283dab3e0ded6231` |
| Gate test | `datasets/gate_a_100.jsonl` | 100 prompts; SHA-256 `d47f5960a3519b81349a12c3e5c548e047e2719aa0e95200df2b034cf937396b` |
| Predictor split | `datasets/predictor_train_validation_100.jsonl` | 80 train + 20 validation; SHA-256 `52d76af69be0ef5dee347855d154794f8c69a489f61dd9bb9cf403be6032c4b4` |
| Secondary GGUF | `/data/models/qwen1.5moe-remoe/qwen1.5moe-q8.gguf` | 15,226,010,912 bytes; SHA-256 `4e676979a67033ccdfc0daacf893705e5b3063973b42056e48f287f04d80cb4a` |

The 100 held-out Gate prompts have zero prompt-hash overlap with predictor train
or validation. Dataset generation records the source dataset hash, deterministic
selection rank, source row, split, prompt ID, and prompt SHA-256.

## Build

The local llama build disables InterfaceIO's optional KEDGE feature because the
KEDGE SDK is not present in this workspace.

```bash
cmake --build build-expert-cache --target \
  llama-debug llama-completion llama-cli llama-server -j2
cargo test --manifest-path /workspace/InterfaceIO/io-scheduler/Cargo.toml \
  --no-default-features
cargo check --manifest-path rust/expert-cache-ffi/Cargo.toml
```

## Model inspection and configuration

Validate file size, architecture, MoE layers, expert count, router top-k, expert
axis, parts, object bytes, and all matched expert tensors without starting model
execution:

```bash
python3 tools/run_pdcat_experiment.py \
  --config experiments/pdcat/configs/deepseek-v2-lite-q8_0.json \
  --profile cache_mixed --dry-run
```

Add `--verify-model-hash` to publication setup checks. It streams the complete
GGUF and rejects a hash mismatch. Normal runs still perform topology and file-size
checks. Cache profiles also validate the expert-pack manifest, its binding to the
model hash and size, alignment, object count, pack size, and recorded pack hash.
Add `--verify-expert-pack-hash` when the full 15.3 GB pack should be re-hashed.

The DeepSeek config exposes these principal profiles:

- `native_cpu`: mmap CPU execution without ExpertManager.
- `cache_cpu`: CPU MoE execution from demand-loaded slots.
- `native_mixed`: CUDA non-expert graph with native CPU MoE tensors.
- `cache_mixed`: mapped host-slot direct use by the CUDA MoE kernel.
- `cache_mixed_promote`: copy active slot bytes to CUDA residency before the same
  MoE kernel; this is the strict same-kernel numerical oracle.
- `cache_mixed_prefill_stream`: topology-driven P1 next-layer staging with two
  layer-sized cache buffers.
- `cache_mixed_predict_history`: model-generic history prediction plus P2 prefetch.
- `cache_mixed_predict_mlp`: prompt-isolated safetensors MLP plus P2 prefetch.

Cross-backend CPU vs CUDA comparisons are reported separately because floating
point accumulation may differ. The strict Gate uses promote vs mapped so a failure
isolates slot addressing/delivery rather than CPU/CUDA numerical variation.

## Expert sidecar pack

The pack stores each logical `(layer, expert)` as a contiguous 4 KiB-aligned
object in configured part order. The manifest records source and packed offsets,
valid/padded lengths, per-part SHA-256, per-object SHA-256, whole-pack SHA-256,
topology, adapter, source model identity, and source-byte verification status.

Create a new pack only at a new explicit destination; the tool refuses to
overwrite an existing pack:

```bash
python3 tools/pack_pdcat_experts.py \
  --config experiments/pdcat/configs/deepseek-v2-lite-q8_0.json \
  --pack /data/models/DeepSeek-V2-Lite-GGUF/DeepSeek-V2-Lite-Q8_0.pdcat-v03.pack \
  --manifest /data/models/DeepSeek-V2-Lite-GGUF/DeepSeek-V2-Lite-Q8_0.pdcat-v03.pack.json \
  --alignment 4096 --verify-model-hash
```

Verify an existing pack and compare every part with the original GGUF:

```bash
python3 tools/pack_pdcat_experts.py --verify-only \
  --manifest /data/models/DeepSeek-V2-Lite-GGUF/DeepSeek-V2-Lite-Q8_0.pdcat-experts.pack.json \
  --verify-model-hash
```

## Gate A correctness

`llama-debug` saves every generated-token logits row, generated token ID, native
router selection/score, and slot owner/generation event. The suite checks dataset
identity, prompt completeness, greedy tokens, logits layout, all finite/non-finite
values, logits top-1, router IDs, router scores, and malformed/duplicate contexts.

```bash
python3 tools/run_pdcat_gate_a.py \
  --config experiments/pdcat/configs/deepseek-v2-lite-q8_0.json \
  --dataset experiments/pdcat/datasets/gate_a_100.jsonl \
  --run-id gate-a-strict-100-v03 \
  --n-predict 2 --expected-prompts 100
```

The completed strict run at
`runs/gate-a-strict-100-v03-20260811` passed 100/100 prompts: 200 generated
tokens, 20,480,000 logits values, 5,200 route contexts, and 16,200 router scores
were bitwise identical between promote and mapped delivery. Max/mean absolute
logits error and RMSE were all zero. Runtime lifecycle changes made after that run
were followed by a strict four-prompt regression of the final binaries at
`runs/gate-a-final-4-v03-20260812`: all 819,200 logits, 208 route contexts, and
648 router scores were bitwise identical, with no generated-token mismatch.

## Prompt-isolated route predictor

Collect native decode routes for all 80 train and 20 validation prompts. This
collection is for predictor labels, not timing conclusions:

```bash
python3 tools/run_pdcat_gate_a.py \
  --dataset experiments/pdcat/datasets/predictor_train_validation_100.jsonl \
  --binary build-expert-cache/bin/llama-debug \
  --run-id predictor-routes-final-v03-20260812 --profile cache_mixed \
  --no-compare --n-predict 8 --expected-prompts 100
```

Train one deterministic candidate. `--test-trace` is mandatory: the manifest is
publication-valid only after calculating metrics from the independent 100-prompt
Gate trace as well as the 80/20 split.

```bash
python3 tools/train_pdcat_predictor.py \
  --trace experiments/pdcat/runs/predictor-routes-final-v03-20260812/cache_mixed/moe_gate.jsonl \
  --test-trace experiments/pdcat/runs/gate-a-strict-100-v03-20260811/cache_mixed/moe_gate.jsonl \
  --hidden-dim 128 --epochs 20 \
  --output-dir experiments/pdcat/runs/predictor-h128 --overwrite
```

Hyperparameters are selected only from aggregate validation recall/waste. The
held-out metrics are reported after selection and are not used to choose the
candidate. The installed bundle consists of `manifest.json` and
`predictor.safetensors` under `predictor/deepseek-v2-lite/`.

One train prompt generated EOS immediately, so it has one generated token and
zero decode transitions. The trainer validates this against the per-prompt
results and records the exclusion; the effective traces contain 79 train,
20 validation, and 100 held-out prompts. The two-token held-out Gate trace
scores source layers 1-25. It cannot score the cyclic layer-26 to next-token
layer-1 transition without a third generated token, so the manifest explicitly
records 25/26 held-out transition coverage.

At load time the Rust FFI requires `PDCAT_MODEL_ID`, `PDCAT_MODEL_SHA256`, and
`PDCAT_MODEL_EXPERT_COUNT` to match the predictor manifest. Layer transitions,
input/output dimensions, hidden width, number of experts, and cyclic last-layer
target are also read from the manifest. Missing/mismatched predictors fail before
inference; runtime prediction failures fall back to demand loading.

## One-run and performance matrix

Run one monitored profile:

```bash
python3 tools/run_pdcat_experiment.py --profile cache_mixed \
  --run-id pdcat-demand-smoke --n-predict 16 --monitor
```

Run selected or all versioned matrix cases:

```bash
python3 tools/run_pdcat_matrix.py \
  --matrix experiments/pdcat/matrices/deepseek-v2-lite-v03.json \
  --case demand-cap64-qd6 --case prefill-stream-cap128 \
  --case history-top2 --case mlp-top2
```

The matrix defines one warmup and three measured repetitions for normal cases;
very slow native baselines explicitly override that count. It covers capacities
64/96/128, P1 prefill staging, history top-1/2/4, MLP top-2, mapped direct-use vs
promotion, QD 1/2/6, and speculative byte budgets. `matrix-summary.json` excludes
warmups and aggregates elapsed time, load time, prefill/decode tokens/s, cache and
prediction hit rates, deadline recall, read amplification, wrong-prefetch bytes,
predictor latency, RSS, and temperature.
Every numerical distribution includes mean, median, min, and max so
storage/lifecycle tails remain visible.

The local v0.3 measurements, exact run directories, negative ablations, and
limitations are reported in `results-v03.md`. In particular, capacity 96
reproduces an approximately 30 s slot/CUDA-handle tail and is not used for a
positive performance claim.

`resource_samples.jsonl` records process RSS/peak RSS, CPU ticks/utilization,
process I/O counters, available memory, thermal zones, NVMe temperatures, and any
visible hwmon/INA power rails. `tegrastats` is also captured when available. The
current container exposes temperatures but neither `tegrastats` nor power sysfs;
therefore absence of a power field is recorded as unavailable, never treated as
zero watts.

## P4 KV write interference

The server experiment populates slot 0, warms the measured slot-1 prompt once,
erases its KV state, brackets an identical slot-1 completion with two no-write
baselines, and concurrently saves slot 0 through the shared P4 scheduler during
the middle completion:

```bash
python3 tools/run_pdcat_kv_interference.py \
  --run-id deepseek-v2-lite-v03-kv-p4-interference \
  --n-predict 16 --populate-repeat 24
```

It preserves server stdout/stderr, HTTP responses/timings, P4 save timing and
bytes, slot image, unified `expert_io`/`kv_io` trace, resource samples, and the
middle-run HTTP, prompt, and decode slowdown relative to the bracketing mean.
P4 work is admitted only
when P0-P3 are empty and is chunked so new foreground work stops further P4
admission; an already submitted storage request cannot be preempted.

## Trace and artifact contract

Raw runs are written under `runs/` and intentionally ignored by Git. Each generic
run records a config snapshot, full inspected model manifest, command argv,
configured environment, three-repository commit/dirty snapshot, model/pack
verification, stdout/stderr, traces, timing, resource summary, status, and errors.

Unified trace events include `expert_io`, `expert_use`, and `kv_io`. Depending on
event type they record run/request/token/layer/phase, logical expert/object,
physical slot/generation, request class P0-P4, operation, submit/ready/use time,
blocked time, predicted deadline/probability, ready-before-deadline, cache hit,
prediction hit, direct-use/promotion, useful/issued bytes, QD/device QD/in-flight
bytes, durability, and success/error. `tools/summarize_pdcat_io.py` emits mean,
p50, p95, p99, max, amplification, actual-use hit metrics, wrong-prefetch bytes,
deadline recall, and deduplicated per-invocation predictor latency.

## Adding another GGUF MoE

Copy the primary config and change only model-specific fields:

1. file path, size, SHA-256, architecture, quantization, and `primary_model`;
2. `expected_topology`, derived from GGUF metadata;
3. adapter regex named groups `layer`/`part`, expert axis, and part order;
4. cache capacity, runtime budget, and model-identity environment;
5. optionally an independently built expert pack and model-specific predictor.

The inspector rejects unmatched tensors, invalid axes, inconsistent expert counts,
non-divisible expert strides, duplicate part roles, and configured topology
mismatches before execution. The scheduler itself does not assume DeepSeek's 27
blocks, 26 MoE layers, 64 experts, top-6, or adjacent MoE layers. The secondary
`configs/qwen1.5-moe-a2.7b-q8.json` configuration validates a different
24-layer, 60-expert, top-4 topology. Cache-CPU and mapped-CUDA smokes cover every
MoE layer, and native prefixed tensor names are traced correctly.
DeepSeek-V2-Lite remains the only model required to complete the full paper
matrix in this version.
