# PDCat v0.3 completion audit

This audit maps the v0.3 objective to current source or raw-run evidence. Raw
run directories are intentionally ignored by Git and retained locally under
`experiments/pdcat/runs/`; a result is not considered proven merely because
the source contains an intended implementation.

## Delivery requirements

| Requirement | Status | Authoritative evidence |
| --- | --- | --- |
| Repository map and operating constraints | Complete | `/workspace/agent.md`, this README, and repository Git state |
| Reproducible environment/model/config/runner | Complete | `environment.json`, `environment.md`, config/schema files, `pdcat_model.py`, `run_pdcat_experiment.py`, and `run_pdcat_matrix.py` |
| DeepSeek-V2-Lite as primary, generic MoE topology | Complete | `configs/deepseek-v2-lite-q8_0.json`; topology and tensor adapter are validated from GGUF metadata rather than scheduler constants |
| Correct mapped mixed-GPU expert execution | Complete | `runs/gate-a-strict-100-v03-20260811/comparison-strict-cache_mixed_promote__vs__cache_mixed.json`: 100 prompts, 20,480,000 bitwise-identical logits, 5,200 route contexts, zero failures |
| Final-binary correctness regression | Complete | `runs/gate-a-final-4-v03-20260812/comparison-strict-cache_mixed_promote__vs__cache_mixed.json`: 4 prompts, 819,200 identical logits, zero token/router mismatch |
| Aligned expert sidecar and model match | Complete | packer/verifier plus the model, pack, and manifest SHA-256 identities in the primary config; 1,664 aligned objects |
| Generation-safe slot lifecycle | Complete | InterfaceIO memory-pool state machine, leases, ticket/FFI generation, CUDA completion release, stale-generation tests, and long route/reuse traces |
| Prefill streaming and progressive handoff | Complete with negative performance result | `cache_mixed_prefill_stream`, zero-copy staging-to-persistent handoff, and `runs/matrix-prefill-fix-v03-20260812`; full-next-layer P1 is 2.51x demand total time in this workload |
| Decode prediction/prefetch/direct-use/promotion/fallback | Complete | history and low-rank MLP predictors, mapped/promote profiles, actual-use accounting, model-bound predictor manifest, strict mapped/promote Gate A |
| P0-P4 bounded I/O and KV coordination | Complete | InterfaceIO priority/deadline/byte/QD admission, promotion/cancel race regression, P4 durable writer, and the warmed P4 interference run |
| Unified machine-readable metrics | Complete for available sensors | `IoMetricsEvent`, router JSONL, process monitor, summaries, matrix aggregator; power is explicitly unavailable on this host |
| Capacity and key ablations | Complete for v0.3 scope | capacity 64/96/128, QD 1/2/6, history top-1/2/4, MLP top-2, speculative budgets, mapped/promote, prefill on/off, P4 on/off |
| Other-MoE extensibility | Complete smoke | Qwen1.5-MoE Q8 config discovers 24 layers, 60 experts, top-4; cache CPU, mapped CUDA, and native route smokes cover all layers |
| Source validation | Complete | four final CMake targets build; InterfaceIO has 18 passing tests and FFI check; Python/JSON validation and both repository `git diff --check` pass |

## Semantic gates

The strict numerical oracle compares two delivery mechanisms using the same
CUDA operator: promote-copy versus mapped direct-use. It proves that delivery,
slot mapping, and lifecycle do not change the computed logits. CPU-native
versus CUDA is retained as a cross-backend semantic comparison because backend
accumulation order need not be bitwise identical.

The native router remains authoritative. Prediction results never replace
router IDs; late, rejected, missing, checksum-failed, or promotion-failed work
falls through to P0 demand loading.

## Negative and unavailable evidence

- Capacity 96 has an unresolved approximately 30-second reusable-slot tail in
  four of six measurements. It is a retained negative diagnostic and excluded
  from positive claims.
- Full-next-layer prefill is correct but slower for the current prompt/topology.
  The implementation is not described as a successful overlap optimization.
- This container exposes temperature but neither `tegrastats` nor a usable
  hwmon power rail. Power and J/token are unavailable, not zero.
- The expensive native-mixed and cache-CPU baselines have one measurement each;
  normal cache cases have one warmup and three measurements.
- Held-out two-token predictor evaluation covers 25/26 cyclic layer
  transitions. The missing last-layer-to-next-token transition is explicit in
  the predictor manifest.

## Publication extensions not claimed by v0.3

The original research roadmap proposes a larger publication grid than the
verified v0.3 matrix: at least 10,000 TPOT samples for robust p99, multiple
prompt lengths, isolated request-size/alignment/bandwidth curves, every
mmap/pread/registered-buffer combination, LFU/static-hot policies, and all
single/dual-ring or SQPOLL/IOPOLL variants. These have not all been measured.
No v0.3 result or summary should be used to claim those comparisons.

Before copying numbers into a paper table, rerun the selected publication
matrix from the final clean commit so every manifest records committed source,
fixed clocks/power mode where available, and the same model/cache state.
