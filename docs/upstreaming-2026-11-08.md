# Imagine if we upstreamed everything: what would we have left?

This records the 2026-11-08 extraction of the tpurtell fork into reviewable upstream proposals. The worktree ends on `work/upstream-everything-remainder-20261108`.

## Preserved history and resolved baseline

[The fork sync PR](https://github.com/tpurtell/GPTQModel/pull/3) merges upstream `6bf8b0b1bbd4d9d24c4e9253923141c4095c0dc8` into fork main `02c6ac672`. Its merge commit `3eaac09b17cd18e63f1f6b8a2f11bf40190e1ab7` preserves the original history. The original divergence was 156 fork commits and 303 upstream commits. Fifteen files required conflict resolution.

The resolution keeps upstream device filtering, EXL3 buffer dtype coercion, single-use Hessians, GLM dense low-rank attention inputs, and early auxiliary-tensor validation. It retains the fork's padded logical outputs, bounded native adapters, durable recovery, and integrated MTP behavior. Focused CPU checks caught and corrected incompatible test assumptions and writer validation order.

The extraction branches contain ten commits above upstream. The remainder was rebased onto the complete extraction stack. Its production tree matches the resolved baseline plus the atomic shard-write fix discovered while extracting export behavior. New standalone regression tests and chunk-boundary coverage are also retained. Neither fork main nor upstream main was merged or rewritten.

## Proposed upstream stack

All upstream PRs are drafts. GitHub cross-fork PRs target upstream `main`; later Files changed views therefore contain predecessors. The incremental comparisons below isolate each layer. Land/rebase in order, then rebase subsequent layers and the final remainder. No PR has automatic merge enabled by this work.

| Layer | Proposal | Incremental review |
|---|---|---|
| 1 | [fix(config): preserve per-layer-type RoPE dictionaries](https://github.com/ModelCloud/GPTQModel/pull/3579) | [one commit](https://github.com/tpurtell/GPTQModel/compare/6bf8b0b1bbd4d9d24c4e9253923141c4095c0dc8...upstream%2F01-nested-rope-20261108) |
| 2 | [fix(loader): restore direct nonpersistent rotary buffers](https://github.com/ModelCloud/GPTQModel/pull/3580) | [one commit](https://github.com/tpurtell/GPTQModel/compare/upstream%2F01-nested-rope-20261108...upstream%2F02-lazy-rotary-20261108) |
| 3 | [fix(writer): stream E8M0 safetensors scales](https://github.com/ModelCloud/GPTQModel/pull/3581) | [one commit](https://github.com/tpurtell/GPTQModel/compare/upstream%2F02-lazy-rotary-20261108...upstream%2F03-e8m0-export-20261108) |
| 4 | [fix(exl3): match native half rounding in Torch codebooks](https://github.com/ModelCloud/GPTQModel/pull/3582) | [one commit](https://github.com/tpurtell/GPTQModel/compare/upstream%2F03-e8m0-export-20261108...upstream%2F04-exl3-half-rounding-20261108) |
| 5 | [fix(exl3): preserve logical outputs over padded storage](https://github.com/ModelCloud/GPTQModel/pull/3583) | [one commit](https://github.com/tpurtell/GPTQModel/compare/upstream%2F04-exl3-half-rounding-20261108...upstream%2F05-exl3-logical-padding-20261108) |
| 6 | [fix(writer): preserve existing shards and config extensions on export](https://github.com/ModelCloud/GPTQModel/pull/3584) | [one commit](https://github.com/tpurtell/GPTQModel/compare/upstream%2F05-exl3-logical-padding-20261108...upstream%2F06-config-only-export-20261108) |
| 7 | [feat(writer): reuse content-authenticated safetensors shards](https://github.com/ModelCloud/GPTQModel/pull/3585) | [one commit](https://github.com/tpurtell/GPTQModel/compare/upstream%2F06-config-only-export-20261108...upstream%2F07-authenticated-shard-resume-20261108) |
| 8 | [feat(offload): reference authenticated safetensors payloads](https://github.com/ModelCloud/GPTQModel/pull/3586) | [one commit](https://github.com/tpurtell/GPTQModel/compare/upstream%2F07-authenticated-shard-resume-20261108...upstream%2F08-safetensors-reference-offload-20261108) |
| 9 | [feat(exl3): add bounded covariance accumulation with an FP64 outer sum](https://github.com/ModelCloud/GPTQModel/pull/3587) | [one commit](https://github.com/tpurtell/GPTQModel/compare/upstream%2F08-safetensors-reference-offload-20261108...upstream%2F09-bounded-covariance-20261108) |
| 10 | [feat(looper): add authenticated disk-backed layer outputs](https://github.com/ModelCloud/GPTQModel/pull/3588) | [one commit](https://github.com/tpurtell/GPTQModel/compare/upstream%2F09-bounded-covariance-20261108...upstream%2F10-disk-backed-layer-outputs-20261108) |

The first six layers fix config preservation, lazy rotary restoration, E8M0 serialization, native-format rounding, padded logical outputs, and checkpoint/config export. Export testing exposed a second hazard: truncating a shard that still backs a memory-mapped tensor. Shard writes now use a sibling temporary file and atomic replacement. This protects individual shards and mapped readers; it does not make an entire model save transactional.

The remaining layers add explicit reusable APIs: authenticated shard reuse, reference offload, bounded covariance accumulation, and disk-backed activation boundaries. Default pipeline integration for the latter utilities stays with the deferred model-specific work.

The old TorchAO-extra change was left in the remainder. Current upstream requires `torchao>=0.18.0` as a base dependency, so adding an older extra alone would not make it optional. Removing the dependency would require a fresh audit of current upstream consumers.

## What remains in the fork

Against layer 10, before adding this note, the remainder contains **103 files, 32,023 added lines and 756 removed lines**. These counts exclude the preceding stack. They include tests and small historical cleanup differences.

| Area | Files | Added | Removed |
|---|---:|---:|---:|
| Native model definitions and calibration/source helpers | 27 | 8,205 | 35 |
| EXL3 orchestration, recovery and durable policy | 9 | 9,139 | 115 |
| Shared loader/looper/writer integration and metadata | 18 | 2,803 | 287 |
| Tests for those changes | 49 | 11,876 | 319 |

The largest retained areas are:

- DeepSeek V4/V4.1 and dSpark continuation, PLE mapping, explicit residual/window state, native checkpoint layouts, and routed calibration.
- GLM core, Flash and MTP source/calibration runners; Kimi K3/K3.1 AttnRes, KDA and padded core formats; Step-5 fused-expert/indexer adapters; Dots3 expert replacement and router integration.
- EXL3 error ledgers, authenticated projection/frontier/batch recovery, remote worker scheduling and transport, device-local serialization, mixed adjacent tiers, and model-bound route recovery policy.
- Shared Hessian ownership, looper replay/materialization hooks, capture/output callbacks, save overlays, native checkpoint passthrough, and active source staging needed by those consumers.

These pieces are retained because they cross core execution/lifetime boundaries, depend on native model implementations, or encode operational recovery and publication policy. Their practical correctness needs native model and accelerator evidence. Small utility proposals do not establish that the complete runners are ready. The remainder PR preserves the implementation and dependency boundary for subsequent review; it proposes the remaining changes to the fork, with layer 10 as its base.

## Validation and resource limits

All new validation ran offline with `CUDA_VISIBLE_DEVICES=''`, `HIP_VISIBLE_DEVICES=''`, and `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`; pytest plugin autoload was disabled. No model quantization, GPU computation, model download, or heavyweight model-scale CPU workload was run. Tests use tiny tensors, metadata fixtures and mocked orchestration. Target-hardware accuracy, throughput, and full-model save/reload/quantization remain unverified.

The final rebased branch passed **100 tests with 36 hardware skips** in its focused smoke group:

```bash
python -m pytest -q tests/test_config_only_export.py \
  tests/test_exl3_native_half_rounding.py tests/test_exllamav3_torch_padding.py \
  tests/test_disk_backed_layer_outputs.py tests/test_out_of_model_tensors.py \
  tests/test_glm5_next_upstream_support.py
```

Earlier focused groups covered offload/reference export and state collection (69 passed, one hardware skip), shard reuse (five passed), bounded covariance (20 passed), and disk-backed outputs (six passed). Groups overlap; these are not additive totals. Each upstream PR records its focused command, coverage, and remaining gates.

The EXL3 codebook law matched all 65,536 states per codebook, with zero bit mismatches and zero absolute drift. Padded output tests passed 63 CPU cases and skipped 36 CUDA cases. A separate tiny CPU numerical probe used inputs `(1,33,128)`, logical outputs 128/576/1056, bias on/off, K6 MCG, and FP16/BF16/FP32 activations. Each output preserved its activation dtype. Against independently reconstructed Torch weights, with `rtol=atol=2e-3` or tighter:

| Activation | FP32 arithmetic max absolute error | Visible output vs raw FP32 | Visible output vs same-dtype oracle |
|---|---:|---:|---:|
| torch.float16 | 1.3038516e-08 | 1.49942935e-05 | 1.52587891e-05 |
| torch.bfloat16 | 1.11758709e-08 | 0.000120896846 | 6.10351562e-05 |
| torch.float32 | 1.49011612e-08 | 1.49011612e-08 | 1.49011612e-08 |

Visible-output columns include mandatory low-precision rounding; they are distinct from the FP32 arithmetic error. No GPU or model-scale inference accuracy claim follows from this probe. The covariance utility compares independent FP64 matrices with normalized error below `1e-6` and checks exact boundary storage laws; full-stream GPU validation is still a readiness gate.
