# GPT-QModel upstream feature extraction and fork remainder

The working fork has been decomposed into **36 draft upstream proposals**: ten initial fixes/utilities and 26 additional feature layers. Each PR has a predecessor link and a comparison isolating its own feature. The final worktree branch is `work/upstream-everything-remainder-20261108`, rebased onto layer 36.

The full original integration/history is preserved in [fork sync PR #3](https://github.com/tpurtell/GPTQModel/pull/3) and local branch `archive/upstream-remainder-before-feature-slices-20261108`. [Fork remainder PR #4](https://github.com/tpurtell/GPTQModel/pull/4) is based on layer 36. No fork or upstream main was merged automatically.

## Resolve additions onto existing model support

The overlap audit used current upstream `cc15d5c13`; the extraction originally started at `6bf8b0b1b`.

| Family | Existing upstream capability | Proposed addition |
|---|---|---|
| DeepSeek V4 | Registered model, quantization tree | Causal target/MTP replay, bounded recovery and native export |
| GLM-5/5.1 | Registered DSA model and replay | Explicit bounded source/MTP adapters and recovery |
| GLM-5.3 Flash | Hybrid KDA/DSA model; untouched MTP tensor passthrough | Actual MTP calibration/quantization and bounded replay |
| Kimi K3/K3.1 | No matching registration/adapter found | Explicit AttnRes source/calibration contracts; K3 registration |
| Dots3 Note | No matching registration found | Separate-expert and router integration |
| Step-5 | No matching adapter found | Bounded native source/core/MTP calibration |
| DeepSeek V4.1 | Inventory documentation; no registered loader | Explicit native source/replay utilities, without automatic registration |

Existing model support stays authoritative. In particular, the GLM proposal keeps upstream's module tree, role metadata, dense shared experts, processor-loading policy and normal Defuser path. The fork's custom expert container/forward/converter was removed. MTP is attached before ordinary Defuser conversion; the existing `after_defuser_conversion` hook validates target/MTP expert coverage. Native expert identity and activation behavior are preserved. The global `_convert_model_with_defuser` function is unchanged from upstream.

DeepSeek V4's base module tree is identical to upstream. Its proposed changes extend auxiliary traversal, replay, recovery and export rather than introduce a parallel model registration. Native runner helpers reuse native model blocks and expose explicit calibration boundaries; they are not advertised as replacement default model implementations.

## Stack and individual reviews

GitHub cross-fork PRs target upstream `main`, so later Files changed views contain predecessors. Use the incremental comparisons below for first-line feedback. Land and rebase in dependency order; all proposals remain drafts. This pass carries previously used code and tests rather than requiring a new full validation campaign before asking which features are useful.

| Layer | Proposal | Incremental review |
|---|---|---|
| 1 | [fix(config): preserve per-layer-type RoPE dictionaries](https://github.com/ModelCloud/GPTQModel/pull/3579) | [feature diff](https://github.com/tpurtell/GPTQModel/compare/6bf8b0b1bbd4d9d24c4e9253923141c4095c0dc8...upstream%2F01-nested-rope-20261108) |
| 2 | [fix(loader): restore direct nonpersistent rotary buffers](https://github.com/ModelCloud/GPTQModel/pull/3580) | [feature diff](https://github.com/tpurtell/GPTQModel/compare/upstream%2F01-nested-rope-20261108...upstream%2F02-lazy-rotary-20261108) |
| 3 | [fix(writer): stream E8M0 safetensors scales](https://github.com/ModelCloud/GPTQModel/pull/3581) | [feature diff](https://github.com/tpurtell/GPTQModel/compare/upstream%2F02-lazy-rotary-20261108...upstream%2F03-e8m0-export-20261108) |
| 4 | [fix(exl3): match native half rounding in Torch codebooks](https://github.com/ModelCloud/GPTQModel/pull/3582) | [feature diff](https://github.com/tpurtell/GPTQModel/compare/upstream%2F03-e8m0-export-20261108...upstream%2F04-exl3-half-rounding-20261108) |
| 5 | [fix(exl3): preserve logical outputs over padded storage](https://github.com/ModelCloud/GPTQModel/pull/3583) | [feature diff](https://github.com/tpurtell/GPTQModel/compare/upstream%2F04-exl3-half-rounding-20261108...upstream%2F05-exl3-logical-padding-20261108) |
| 6 | [fix(writer): preserve existing shards and config extensions on export](https://github.com/ModelCloud/GPTQModel/pull/3584) | [feature diff](https://github.com/tpurtell/GPTQModel/compare/upstream%2F05-exl3-logical-padding-20261108...upstream%2F06-config-only-export-20261108) |
| 7 | [feat(writer): reuse content-authenticated safetensors shards](https://github.com/ModelCloud/GPTQModel/pull/3585) | [feature diff](https://github.com/tpurtell/GPTQModel/compare/upstream%2F06-config-only-export-20261108...upstream%2F07-authenticated-shard-resume-20261108) |
| 8 | [feat(offload): reference authenticated safetensors payloads](https://github.com/ModelCloud/GPTQModel/pull/3586) | [feature diff](https://github.com/tpurtell/GPTQModel/compare/upstream%2F07-authenticated-shard-resume-20261108...upstream%2F08-safetensors-reference-offload-20261108) |
| 9 | [feat(exl3): add bounded covariance accumulation with an FP64 outer sum](https://github.com/ModelCloud/GPTQModel/pull/3587) | [feature diff](https://github.com/tpurtell/GPTQModel/compare/upstream%2F08-safetensors-reference-offload-20261108...upstream%2F09-bounded-covariance-20261108) |
| 10 | [feat(looper): add authenticated disk-backed layer outputs](https://github.com/ModelCloud/GPTQModel/pull/3588) | [feature diff](https://github.com/tpurtell/GPTQModel/compare/upstream%2F09-bounded-covariance-20261108...upstream%2F10-disk-backed-layer-outputs-20261108) |
| 11 | [feat(exl3): preserve Hessian conditioning and projection diagnostics](https://github.com/ModelCloud/GPTQModel/pull/3591) | [feature diff](https://github.com/tpurtell/GPTQModel/compare/upstream%2F10-disk-backed-layer-outputs-20261108...upstream%2F11-exl3-numerical-diagnostics-20261108) |
| 12 | [feat(exl3): retain restart-idempotent projection and route ledgers](https://github.com/ModelCloud/GPTQModel/pull/3592) | [feature diff](https://github.com/tpurtell/GPTQModel/compare/upstream%2F11-exl3-numerical-diagnostics-20261108...upstream%2F12-exl3-error-ledgers-20261108) |
| 13 | [feat(exl3): checkpoint completed projections atomically](https://github.com/ModelCloud/GPTQModel/pull/3593) | [feature diff](https://github.com/tpurtell/GPTQModel/compare/upstream%2F12-exl3-error-ledgers-20261108...upstream%2F13-exl3-projection-checkpoints-20261108) |
| 14 | [feat(exl3): persist authenticated capture frontiers](https://github.com/ModelCloud/GPTQModel/pull/3594) | [feature diff](https://github.com/tpurtell/GPTQModel/compare/upstream%2F13-exl3-projection-checkpoints-20261108...upstream%2F14-exl3-capture-frontiers-20261108) |
| 15 | [feat(exl3): resume capture at calibration-batch boundaries](https://github.com/ModelCloud/GPTQModel/pull/3595) | [feature diff](https://github.com/tpurtell/GPTQModel/compare/upstream%2F14-exl3-capture-frontiers-20261108...upstream%2F15-exl3-capture-batch-spool-20261108) |
| 16 | [feat(moe): expose ranked router candidates for route recovery](https://github.com/ModelCloud/GPTQModel/pull/3596) | [feature diff](https://github.com/tpurtell/GPTQModel/compare/upstream%2F15-exl3-capture-batch-spool-20261108...upstream%2F16-exl3-router-recovery-20261108) |
| 17 | [feat(exl3): persist deterministic adjacent mixed-tier plans](https://github.com/ModelCloud/GPTQModel/pull/3597) | [feature diff](https://github.com/tpurtell/GPTQModel/compare/upstream%2F16-exl3-router-recovery-20261108...upstream%2F17-exl3-inline-mixed-tiers-20261108) |
| 18 | [feat(exl3): dispatch authenticated projections to pipelined workers](https://github.com/ModelCloud/GPTQModel/pull/3598) | [feature diff](https://github.com/tpurtell/GPTQModel/compare/upstream%2F17-exl3-inline-mixed-tiers-20261108...upstream%2F18-exl3-remote-workers-20261108) |
| 19 | [feat(loader): preserve lazy native sources across replay and export](https://github.com/ModelCloud/GPTQModel/pull/3599) | [feature diff](https://github.com/tpurtell/GPTQModel/compare/upstream%2F18-exl3-remote-workers-20261108...upstream%2F19-lazy-source-replay-contracts-20261108) |
| 20 | [feat(looper): carry causal replay and durable recovery callbacks](https://github.com/ModelCloud/GPTQModel/pull/3600) | [feature diff](https://github.com/tpurtell/GPTQModel/compare/upstream%2F19-lazy-source-replay-contracts-20261108...upstream%2F20-looper-causal-replay-20261108) |
| 21 | [feat(exl3): integrate resumable distributed and mixed-tier execution](https://github.com/ModelCloud/GPTQModel/pull/3601) | [feature diff](https://github.com/tpurtell/GPTQModel/compare/upstream%2F20-looper-causal-replay-20261108...upstream%2F21-exl3-durable-execution-20261108) |
| 22 | [feat(writer): compose native checkpoint passthrough and save overlays](https://github.com/ModelCloud/GPTQModel/pull/3602) | [feature diff](https://github.com/tpurtell/GPTQModel/compare/upstream%2F21-exl3-durable-execution-20261108...upstream%2F22-native-checkpoint-export-20261108) |
| 23 | [feat(deepseek): carry bounded V4 target and MTP calibration replay](https://github.com/ModelCloud/GPTQModel/pull/3603) | [feature diff](https://github.com/tpurtell/GPTQModel/compare/upstream%2F22-native-checkpoint-export-20261108...upstream%2F23-deepseek-v4-mtp-replay-20261108) |
| 24 | [feat(glm): integrate Flash target and MTP recovery and export](https://github.com/ModelCloud/GPTQModel/pull/3604) | [feature diff](https://github.com/tpurtell/GPTQModel/compare/upstream%2F23-deepseek-v4-mtp-replay-20261108...upstream%2F24-glm-integrated-mtp-20261108) |
| 25 | [feat(models): expose Dots3 Note routed experts individually](https://github.com/ModelCloud/GPTQModel/pull/3605) | [feature diff](https://github.com/tpurtell/GPTQModel/compare/upstream%2F24-glm-integrated-mtp-20261108...upstream%2F25-dots3-separate-experts-20261108) |
| 26 | [feat(kimi): add bounded K3 source and explicit AttnRes model contracts](https://github.com/ModelCloud/GPTQModel/pull/3606) | [feature diff](https://github.com/tpurtell/GPTQModel/compare/upstream%2F25-dots3-separate-experts-20261108...upstream%2F26-kimi-k3-native-source-20261108) |
| 27 | [feat(kimi): carry causal core calibration and padded EXL3 source storage](https://github.com/ModelCloud/GPTQModel/pull/3607) | [feature diff](https://github.com/tpurtell/GPTQModel/compare/upstream%2F26-kimi-k3-native-source-20261108...upstream%2F27-kimi-core-calibration-20261108) |
| 28 | [feat(kimi): carry K3.1 source and bounded calibration adapters](https://github.com/ModelCloud/GPTQModel/pull/3608) | [feature diff](https://github.com/tpurtell/GPTQModel/compare/upstream%2F27-kimi-core-calibration-20261108...upstream%2F28-kimi-k31-calibration-20261108) |
| 29 | [feat(glm): add bounded native core calibration and EXL3 sources](https://github.com/ModelCloud/GPTQModel/pull/3609) | [feature diff](https://github.com/tpurtell/GPTQModel/compare/upstream%2F28-kimi-k31-calibration-20261108...upstream%2F29-glm-core-calibration-20261108) |
| 30 | [feat(glm): carry native full-MTP alignment and source loading](https://github.com/ModelCloud/GPTQModel/pull/3610) | [feature diff](https://github.com/tpurtell/GPTQModel/compare/upstream%2F29-glm-core-calibration-20261108...upstream%2F30-glm-native-mtp-20261108) |
| 31 | [feat(glm): carry Flash core and MTP calibration adapters](https://github.com/ModelCloud/GPTQModel/pull/3611) | [feature diff](https://github.com/tpurtell/GPTQModel/compare/upstream%2F30-glm-native-mtp-20261108...upstream%2F31-glm-flash-native-calibration-20261108) |
| 32 | [feat(step5): carry bounded native core and MTP calibration](https://github.com/ModelCloud/GPTQModel/pull/3612) | [feature diff](https://github.com/tpurtell/GPTQModel/compare/upstream%2F31-glm-flash-native-calibration-20261108...upstream%2F32-step5-native-calibration-20261108) |
| 33 | [feat(deepseek): carry V4.1 native source and mapped PLE replay](https://github.com/ModelCloud/GPTQModel/pull/3613) | [feature diff](https://github.com/tpurtell/GPTQModel/compare/upstream%2F32-step5-native-calibration-20261108...upstream%2F33-v41-native-source-20261108) |
| 34 | [feat(deepseek): carry V4.1 routed capture and adjacent recovery](https://github.com/ModelCloud/GPTQModel/pull/3614) | [feature diff](https://github.com/tpurtell/GPTQModel/compare/upstream%2F33-v41-native-source-20261108...upstream%2F34-v41-routed-calibration-20261108) |
| 35 | [feat(deepseek): persist authenticated typed V4.1 frontiers](https://github.com/ModelCloud/GPTQModel/pull/3615) | [feature diff](https://github.com/tpurtell/GPTQModel/compare/upstream%2F34-v41-routed-calibration-20261108...upstream%2F35-v41-durable-frontiers-20261108) |
| 36 | [feat(deepseek): carry V4.1 main-input and dSpark replay adapters](https://github.com/ModelCloud/GPTQModel/pull/3616) | [feature diff](https://github.com/tpurtell/GPTQModel/compare/upstream%2F35-v41-durable-frontiers-20261108...upstream%2F36-v41-dspark-replay-20261108) |

The additional feature sequence covers quantizer diagnostics; error ledgers; projection checkpoints; capture frontiers and batches; router recovery; mixed tiers; authenticated remote scheduling; lazy source contracts; looper and processor integration; native export; and separate DeepSeek, GLM, Dots3, Kimi, Step-5 and V4.1 adapters. Flow/reference pages are carried in the relevant PRs and linked from README.

## What is left?

Before this analysis page, the final remainder is **four files, 21 added lines and three removed lines**:

- `gptqmodel/_banner.py` and `tests/test_startup_banner.py`: the fork's byte-output handling and corresponding regression.
- `pyproject.toml`: the historical TorchAO extra. Current upstream already requires `torchao>=0.18.0`, so the older extra alone does not make TorchAO optional.
- `tests/test_xing4_0_support.py`: one historical test-line cleanup.

The earlier 103-file, 32,023-added-line remainder has therefore been proposed as individual features. Duplicate GLM conversion machinery and the fork-specific default shared-expert quantization policy were reconciled with upstream rather than reintroduced into this final branch. Their original form remains in the archived integration history.

## Lightweight validation

The new extraction checked dependency order, per-layer Python parsing, whitespace and Mermaid fences. All 26 new reference pages are present; 99 carried Python-file occurrences parse across their feature layers. Existing tests/oracles were retained with their features.

For the upstream reconciliation, **10 tiny model-plan/registry tests passed** and **two tiny GLM integration tests passed**. The latter demonstrate native expert object/forward preservation and lazy checkpoint materialization through normal Defuser conversion. All used CPU-only offline execution with one BLAS/OpenMP thread. No model quantization, GPU/ROCm work, heavyweight CPU workload or full-model benchmark was run.

The initial extraction's recorded checks remain documented in its PRs: a 100-pass/36-hardware-skip focused smoke group, exact codebook format-law checks, and separate FP32 arithmetic versus activation-output rounding measurements. Those results describe that earlier validation point; they are not presented as a fresh full-stack runtime rerun.

Exact focused reconciliation commands, with `CUDA_VISIBLE_DEVICES=''`, `HIP_VISIBLE_DEVICES=''`, `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`, offline Hugging Face mode and pytest plugin autoload disabled:

```bash
python -m pytest -q tests/module_tree/test_dots3_note_fork_plan.py \
  tests/module_tree/test_kimi_k3_fork_plan.py \
  tests/module_tree/test_kimi_linear_fork_plan.py \
  tests/module_tree/test_registered_module_plans.py \
  -k 'fork_plan or explicit_plan_case or deepseek_v4 or glm5_next or glm_moe_dsa or dots3_note or kimi_k3 or kimi_linear'
# 10 passed, 196 deselected

python -m pytest -q tests/test_glm5_next_upstream_support.py \
  -k 'defuser_registry or lazy_turtle'
# 2 passed, 5 deselected
```
