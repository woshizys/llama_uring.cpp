# PDCat v0.3 experiment results

This document reports the local Jetson Orin NX 16 GB results produced on
2026-08-11/12 before the source commit. Raw run directories are ignored by Git
but retained locally under `experiments/pdcat/runs/`; every cited directory
contains the command, config snapshot, repository state, traces, timings,
resource samples, and status.

## Correctness gates

- Formal strict Gate A:
  `runs/gate-a-strict-100-v03-20260811`.
  All 100 prompts passed between promote-CUDA and mapped-CUDA: 200 generated
  tokens, 20,480,000 logits, 5,200 route contexts, and 16,200 router scores
  were bitwise identical. Max/mean logits error and RMSE were zero.
- Final post-build ABI/lifecycle regression:
  `runs/gate-a-final-4-v03-20260812`.
  All four prompts passed; 819,200 logits, 208 route contexts, and 648 router
  scores were bitwise identical.
- The CPU-native versus CUDA result is a semantic cross-backend comparison,
  not the strict numerical oracle. Greedy/top-1 results matched in the tested
  suite, while CPU/CUDA accumulation may differ below top-1.
- The aligned expert pack contains 1,664 4 KiB-aligned objects. Model, pack,
  and manifest SHA-256 identities are recorded in the run config.

## Predictor

The route corpus run is
`runs/predictor-routes-final-v03-20260812`. One train prompt generated EOS
immediately and therefore had no decode transition; the trainer records and
excludes it. The effective trace split is 79 train, 20 validation, and 100
prompt-isolated held-out test prompts with zero prompt-hash overlap.

Hidden dimensions 32, 64, and 128 were compared using validation only. Hidden
128 was selected. Its aggregate metrics are:

| Split | recall@2 | waste@2 | recall@4 | waste@4 |
| --- | ---: | ---: | ---: | ---: |
| Validation (3,594 samples) | 0.2349 | 0.2954 | 0.4080 | 0.3879 |
| Held-out test (2,500 samples) | 0.2942 | 0.1174 | 0.5387 | 0.1919 |

The held-out two-token Gate trace scores 25 of 26 source-layer transitions.
It cannot score layer 26 to the next token's layer 1 because that requires a
third generated token. This missing transition is explicit in the predictor
manifest and must not be described as full 26/26 held-out coverage.

The installed weights SHA-256 is
`e50e1e6029485ffe2022a6d40aee551c075c30761ab21f7d362e6b02a7580233`.
An online eight-token smoke traced 209 independent predictor calls. Predictor
latency was 132.7 us p50 and 161.8 us p95; 233 predictions were visible at
actual use and 185 correlated reads were wrong-prefetches.

## Performance and ablations

The principal matrix is `runs/matrix-core-v03-20260812`: one warmup and three
measurements per normal case, with warmups excluded. The original manifest is
marked failed only because all four pre-fix full-layer prefill runs reproduced
a P1-cancel/P0-promotion race. The race was fixed and its replacement matrix
`runs/matrix-prefill-fix-v03-20260812` passed 4/4. Values below are medians;
ranges remain in each `matrix-summary.json`.

| Case | Decode tok/s | Prompt tok/s | Total ms | Peak RSS |
| --- | ---: | ---: | ---: | ---: |
| Native mixed baseline (one run) | 0.70 | 0.29 | 147,237 | 12.70 GB |
| Cache CPU baseline (one run) | 0.89 | 4.96 | 25,156 | 2.42 GB |
| Mapped demand, capacity 64, QD6 | 1.16 | 4.90 | 21,140 | 1.21 GB |
| Mapped demand, capacity 128, QD6 | 1.16 | 4.89 | 21,200 | 1.80 GB |
| Promote-copy, capacity 64 | 0.97 | 4.45 | 24,656 | 1.21 GB |
| History predictor, top-1 | 1.20 | 5.05 | 20,439 | 1.21 GB |
| MLP predictor, top-2 | 1.19 | 5.04 | 20,670 | 1.21 GB |
| Full-layer P1 prefill, capacity 128 | 1.10 | 0.97 | 53,136 | 1.80 GB |

Mapped direct use is about 19.6% faster in median decode throughput than
promote-copy for this workload. Capacity 64 and 128 have indistinguishable
median decode throughput; capacity 128 consumes about 0.59 GB more peak RSS.
QD1/QD2/QD6 medians are 1.17/1.16/1.16 tok/s, so increasing device concurrency
past one did not improve this single-request workload.

History top-1 has 3.45% prediction-hit rate over all expert uses, 100% of its
admitted predictions ready before deadline, and 2.64 GB correlated wrong
prefetch bytes. MLP top-2 raises the hit rate to 12.90% with 100% median ready
recall and 3.20 GB wrong-prefetch bytes; its matrix predictor latency is
97.0 us median. The 9 MiB speculative budget keeps about 99.5% ready recall.
The 36 MiB/top-4 variant admits more work but only 1.44% is ready by deadline,
showing that larger speculative budgets can create late queueing rather than
useful overlap.

Full-next-layer P1 is functionally correct after the race fix but is not a
winning default for this topology. It has a median queue depth of 120 versus 56
for demand and issues 35.56 versus 32.25 GiB. Its median total time is 2.51x
demand-cap128. The result supports bounded/top-k staging rather than loading all
64 experts of the next layer.

Capacity 96 has a reproducible unresolved tail: four of six measurements across
the original and diagnostic rerun contain an approximately 30 s wait for a
reusable slot. The storage read completes successfully and other capacities do
not show the same behavior. Evidence points to long-lived CUDA/graph handle
pinning at this intermediate capacity. Capacity-96 numbers are retained as a
negative diagnostic and excluded from positive performance claims.

## P4 KV interference

The fair, warmed formal run is
`runs/p4-kv-interference-warm-formal-v03-20260812`. Slot 0 contains 578 tokens
and is saved as 17,987,716 bytes in five chunks, followed by `fdatasync` and
directory `fsync`. All seven P4 events succeeded.

The identical slot-1 prompt is warmed once, erased, then measured before,
during, and after the P4 save:

| Metric | Slowdown ratio |
| --- | ---: |
| HTTP elapsed | 1.0029x |
| Server prompt time | 1.0005x |
| Server decode time | 0.9935x |

No degradation beyond run-to-run noise is detectable for this 18 MB write. The
result demonstrates correct bounded coexistence, not a claim for arbitrary KV
sizes or multi-request saturation.

## Secondary-model portability

`configs/qwen1.5-moe-a2.7b-q8.json` binds the local Qwen1.5-MoE A2.7B Q8
model by SHA-256. The generic inspector discovered a different topology:
24 MoE layers (0-23), 60 experts per layer, and top-4 routing. Cache-CPU and
mapped-CUDA real-inference smokes both completed and covered all 24 layers.
A native-mixed trace produced 48 route events across all layers with zero
tensor-name parsing warnings. Qwen is a portability smoke, not part of the full
DeepSeek performance matrix.

## Measurement limitations

- This container exposes thermal sensors but neither `tegrastats` nor hwmon
  power rails. Temperature and RSS are measured; power is unavailable, not 0 W.
- The run has one very slow native baseline and one cache-CPU baseline because
  they are expensive. Normal cache cases use three measurements.
- Several cases contain isolated approximately 30 s slot/CUDA lifecycle tails.
  Median, min, and max are all preserved; no outlier is silently deleted.
- Results are local working-tree artifacts until the reviewed source is
  committed. Re-run the publication matrix from the final commit before using
  these values in a paper table.
