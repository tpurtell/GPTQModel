# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import copy
import math
import os
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import transformers
from torch.nn import Module
from torch.nn.modules.conv import _ConvNd

from ..exllamav3.modules.quant.exl3_lib.quantize import (
    EXL3_HESSIAN_NUMERICAL_CONTRACT,
    EXL3_HESSIAN_SYMMETRY_CONTRACT,
    quantize_exl3,
    reconstruct_exl3_tensors,
)
from ..looper.loop_processor import (
    DTYPE_SIZE_COLUMN,
    MODULE_FEATURE_COLUMN,
    ExecutionConfig,
    LoopProcessor,
)
from ..looper.named_module import NamedModule
from ..models import BaseQModel
from ..models.writer import (
    PROCESS_LOG_FWD_TIME,
    PROCESS_LOG_LAYER,
    PROCESS_LOG_MODULE,
    PROCESS_LOG_NAME,
    PROCESS_LOG_TIME,
    PROCESS_USED_MEMORY,
    QUANT_LOG_DAMP,
    QUANT_LOG_LOSS,
    QUANT_LOG_LOSS_KIND,
    QUANT_LOG_NSAMPLES,
)
from ..nn_modules.exllamav3 import ExllamaV3Linear
from ..quantization import QuantizeConfig
from ..quantization.config import FORMAT, METHOD, EXL3Config, GPTQConfig
from ..quantization.gptq import GPTQ
from ..utils.device import get_device
from ..utils.exl3_error_ledger import (
    JOURNAL_ENV,
    ROUTE_EVIDENCE_SCHEMA,
    ROUTE_EVIDENCE_SCHEMA_VERSION,
    append_exl3_error_journal,
    build_projection_record,
    route_evidence_required,
    routed_expert_identity,
)
from ..utils.exl3_projection_checkpoint import (
    EXL3ProjectionCheckpointStore,
    build_projection_request,
    canonical_json_bytes,
    checkpoint_root_from_provenance,
    sha256_bytes,
)
from ..utils.exl3_remote import (
    CoordinatorSlot,
    EXL3_HESSIAN_CAPTURE_CONTRACT,
    ExecutionSlotLease,
    RemoteEndpoint,
    exl3_quantization_failure_message,
    remote_client_from_provenance,
    validate_exl3_hessian_metrics,
)
from ..utils.exllamav3 import create_exllamav3_module
from ..utils.logger import setup_logger
from ..utils.module_locks import parent_module_lock
from ..utils.offload import offload_to_disk

setup_logger()

_EXL3_SIGMA_REG = 0.025
_OUT_SCALES_TO_ARG = {
    "always": True,
    "never": False,
    "auto": None,
    None: None,
}


def prepare_exl3_hessian(
    capture: GPTQ,
    *,
    target_device: torch.device,
    module_full_name: str,
) -> torch.Tensor:
    """Restore GPTQ's normalized capture to EXL3's raw X^T X contract."""

    capture.finalize_hessian(target_device=target_device)
    hessian = capture.H
    if hessian is None:
        raise RuntimeError(
            f"EXL3 failed to capture Hessian for module `{module_full_name}`."
        )
    if capture.nsamples <= 0:
        raise RuntimeError(
            f"EXL3 captured no calibration activations for module `{module_full_name}`."
        )
    # GPTQ capture stores 2/N * X^T X. ExLlamaV3's H_data finalizer divides
    # by `count`, so restore the raw X^T X sum before handing it over.
    hessian.mul_(float(capture.nsamples) / 2.0)
    return hessian


class _EXL3NaturalRouteCapture:
    """Accumulate per-expert exposure during one native-router subset replay."""

    def __init__(
        self,
        processor: "EXL3Processor",
        *,
        layer_module: Module,
        subset: Dict[str, NamedModule],
    ) -> None:
        self.processor = processor
        self.targets: dict[str, dict[str, Any]] = {}
        for task_name, named_module in subset.items():
            full_name = getattr(named_module, "full_name", None)
            identity = (
                routed_expert_identity(full_name)
                if isinstance(full_name, str)
                else None
            )
            if identity is not None and task_name in processor.tasks:
                self.targets[task_name] = identity
        family_ids = {
            (identity["block_namespace"], identity["logical_layer"])
            for identity in self.targets.values()
        }
        if not self.targets or len(family_ids) != 1:
            raise ValueError(
                "EXL3 natural-route capture requires one routed block per subset"
            )

        mlp = getattr(layer_module, "mlp", None)
        if mlp is None:
            mlp = getattr(layer_module, "ffn", None)
        self.router = getattr(mlp, "gate", None)
        if not isinstance(self.router, Module):
            raise ValueError("EXL3 natural-route capture could not resolve the router")

        self._lock = threading.Lock()
        self._handle = None
        self._router_calls = 0
        self._router_token_count = 0
        self._router_selected_route_count = 0
        self._router_top_k: int | None = None
        self._expert_counts: torch.Tensor | None = None
        self._gate_sums: torch.Tensor | None = None
        self._gate_squared_mass: torch.Tensor | None = None
        self._router_weight_dtypes: set[str] = set()
        self._mask_modes: set[str] = set()

    def __enter__(self) -> "_EXL3NaturalRouteCapture":
        self._handle = self.router.register_forward_hook(self._capture)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        del exc_value, traceback
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        if exc_type is None:
            self.processor._commit_natural_route_capture(self)
        return False

    def _capture(self, _module, _inputs, output) -> None:
        paused_tls = getattr(self.processor, "_hooks_paused_tls", None)
        if paused_tls is not None and getattr(paused_tls, "value", False):
            return
        if not isinstance(output, (tuple, list)) or len(output) != 3:
            raise RuntimeError("EXL3 natural-route capture requires router triplets")
        logits, weights, indices = output
        if (
            not isinstance(logits, torch.Tensor)
            or not isinstance(weights, torch.Tensor)
            or not isinstance(indices, torch.Tensor)
            or weights.shape != indices.shape
            or weights.ndim != 2
            or logits.ndim != 2
            or logits.shape[0] != weights.shape[0]
        ):
            raise RuntimeError(
                "EXL3 natural-route capture received malformed router output"
            )

        keep_mask = getattr(getattr(self.processor, "_mask_tls", None), "value", None)
        if keep_mask is None:
            mask_mode = "absent"
        else:
            flat_keep = keep_mask.reshape(-1).to(
                device=indices.device, dtype=torch.bool
            )
            if flat_keep.numel() != indices.shape[0]:
                raise RuntimeError(
                    "EXL3 natural-route mask does not align with router rows"
                )
            mask_mode = "all-valid" if bool(flat_keep.all().item()) else "filtered"
            weights = weights[flat_keep]
            indices = indices[flat_keep]

        num_experts = int(logits.shape[-1])
        top_k = int(indices.shape[-1])
        # CUDA weighted bincount uses atomic additions whose final low bits can
        # vary across otherwise identical runs. Route evidence is part of the
        # projection checkpoint identity, so reduce this small vector on CPU in
        # float64 and keep resume identities bit-stable.
        flat_indices = indices.detach().reshape(-1).to(
            device="cpu",
            dtype=torch.int64,
        )
        flat_weights = weights.detach().reshape(-1).to(
            device="cpu",
            dtype=torch.float64,
        )
        if flat_indices.numel() == 0:
            return
        counts = torch.bincount(flat_indices, minlength=num_experts)
        gate_sums = torch.bincount(
            flat_indices,
            weights=flat_weights,
            minlength=num_experts,
        )
        gate_squared_mass = torch.bincount(
            flat_indices,
            weights=flat_weights.square(),
            minlength=num_experts,
        )
        if any(
            value.numel() != num_experts
            for value in (counts, gate_sums, gate_squared_mass)
        ):
            raise RuntimeError(
                "EXL3 natural-route capture observed an invalid expert id"
            )
        counts = counts.to(dtype=torch.int64)
        gate_sums = gate_sums.to(dtype=torch.float64)
        gate_squared_mass = gate_squared_mass.to(dtype=torch.float64)

        with self._lock:
            if self._router_top_k is None:
                self._router_top_k = top_k
                self._expert_counts = torch.zeros_like(counts)
                self._gate_sums = torch.zeros_like(gate_sums)
                self._gate_squared_mass = torch.zeros_like(gate_squared_mass)
            elif (
                self._router_top_k != top_k
                or self._expert_counts.numel() != num_experts
            ):
                raise RuntimeError("EXL3 natural-route geometry changed during replay")
            self._expert_counts.add_(counts)
            self._gate_sums.add_(gate_sums)
            self._gate_squared_mass.add_(gate_squared_mass)
            self._router_calls += 1
            self._router_token_count += int(indices.shape[0])
            self._router_selected_route_count += int(flat_indices.numel())
            self._router_weight_dtypes.add(str(weights.dtype))
            self._mask_modes.add(mask_mode)

    def evidence_by_expert(self) -> dict[int, dict[str, Any]]:
        """Finalize the bounded route metrics needed by expert-family tiering."""

        if (
            self._router_calls <= 0
            or self._router_token_count <= 0
            or self._router_top_k is None
            or self._expert_counts is None
            or self._gate_sums is None
            or self._gate_squared_mass is None
        ):
            raise RuntimeError("EXL3 natural-route capture observed no router traffic")
        total_gate_sum = float(self._gate_sums.sum().item())
        total_gate_sq = float(self._gate_squared_mass.sum().item())
        if total_gate_sum <= 0 or total_gate_sq <= 0:
            raise RuntimeError(
                "EXL3 natural-route capture observed no positive gate mass"
            )
        namespace, logical_layer = next(
            iter(
                {
                    (identity["block_namespace"], identity["logical_layer"])
                    for identity in self.targets.values()
                }
            )
        )
        evidence: dict[int, dict[str, Any]] = {}
        for expert in sorted(
            {identity["expert"] for identity in self.targets.values()}
        ):
            route_count = int(self._expert_counts[expert].item())
            gate_sum = float(self._gate_sums[expert].item())
            gate_sq = float(self._gate_squared_mass[expert].item())
            evidence[expert] = {
                "schema": ROUTE_EVIDENCE_SCHEMA,
                "schema_version": ROUTE_EVIDENCE_SCHEMA_VERSION,
                "block_namespace": namespace,
                "logical_layer": logical_layer,
                "expert": expert,
                "router_calls": self._router_calls,
                "router_token_count": self._router_token_count,
                "router_selected_route_count": self._router_selected_route_count,
                "router_top_k": self._router_top_k,
                "expert_route_count": route_count,
                "expert_gate_weight_sum": gate_sum,
                "expert_gate_squared_mass": gate_sq,
                "total_gate_weight_sum": total_gate_sum,
                "total_gate_squared_mass": total_gate_sq,
                "expert_route_fraction": route_count
                / self._router_selected_route_count,
                "expert_gate_weight_mass_fraction": gate_sum / total_gate_sum,
                "expert_gate_squared_mass_fraction": gate_sq / total_gate_sq,
                "expert_gate_weight_mean": gate_sum / max(route_count, 1),
                "expert_gate_weight_rms": math.sqrt(gate_sq / max(route_count, 1)),
                "router_weight_dtypes": sorted(self._router_weight_dtypes),
                "mask_modes": sorted(self._mask_modes),
            }
        return evidence


def clone_exllamav3_config_for_module(
    qcfg: EXL3Config,
    module_full_name: str,
) -> Optional[EXL3Config]:
    """Clones and applies per-module EXL3 dynamic overrides, or skips the module."""

    if not qcfg.module_is_included(module_full_name):
        return None

    if qcfg.dynamic_get(layer_name=module_full_name) is False:
        return None

    qcfg_clone = copy.deepcopy(qcfg)

    if qcfg.dynamic is not None:
        qcfg_clone.bits = qcfg.dynamic_get(module_full_name, "bits", qcfg_clone.bits)
        qcfg_clone.head_bits = qcfg.dynamic_get(
            module_full_name, "head_bits", qcfg_clone.head_bits
        )

        out_scales_override = qcfg.dynamic_get(module_full_name, "out_scales", None)
        if out_scales_override is not None:
            qcfg_clone.out_scales = out_scales_override

        codebook_override = qcfg.dynamic_get(module_full_name, "codebook", None)
        if codebook_override is not None:
            qcfg_clone.codebook = codebook_override

        calibration_override = qcfg.dynamic_get(module_full_name, "calibration", None)
        if calibration_override is not None:
            qcfg_clone.calibration = calibration_override

    qcfg_clone.__post_init__()
    return qcfg_clone


class EXL3Processor(LoopProcessor):
    """Captures activations and repacks modules into ExLlamaV3 format."""

    def __init__(
        self,
        tokenizer,
        qcfg: QuantizeConfig,
        calibration,
        prepare_dataset_func,
        calibration_concat_size: Optional[int],
        calibration_sort: Optional[str],
        batch_size: int,
        require_fwd: bool = True,
        calibration_concat_separator: Optional[str] = None,
        lm_head_name: str = "lm_head",
    ):
        """Initializes EXL3 processing and tracks the lm_head naming convention."""

        super().__init__(
            tokenizer=tokenizer,
            qcfg=qcfg,
            calibration=calibration,
            calibration_concat_size=calibration_concat_size,
            calibration_sort=calibration_sort,
            calibration_concat_separator=calibration_concat_separator,
            prepare_dataset_func=prepare_dataset_func,
            batch_size=batch_size,
            execution_config=ExecutionConfig(
                require_fwd=require_fwd,
                fwd_replay_after_process=True,
                subset_forward_early_stop=True,
            ),
        )

        self.avg_losses = []
        self.lm_head_name = lm_head_name
        self._stats_lock = threading.Lock()
        self._natural_route_evidence_cache: dict[
            tuple[str, int], dict[int, dict[str, Any]]
        ] = {}
        self._remote_client_initialized = False
        self._remote_client = None
        self._projection_checkpoint_store_initialized = False
        self._projection_checkpoint_store = None
        self._distributed_local_quant_locks: dict[str, threading.Lock] = {}
        self.error_journal_path = os.getenv(JOURNAL_ENV) or str(
            Path(self.log_tmp_log_file_name).with_suffix(".exl3-error-ledger.jsonl")
        )

    def set_calibration_dataset(self, calibration_dataset):
        """Rejects dataset replacement because EXL3 capture is fixed at construction."""

        raise NotImplementedError(
            "EXL3Processor's calibration_dataset cannot be modified"
        )

    def _ledger_provenance(self) -> dict[str, Any] | None:
        """Return the run provenance shared by every EXL3 projection."""

        if not isinstance(self.qcfg.meta, dict):
            return None
        provenance = self.qcfg.meta.get("ds4rt_error_ledger")
        if provenance is not None and not isinstance(provenance, dict):
            raise ValueError(
                "EXL3 `meta.ds4rt_error_ledger` provenance must be a dictionary"
            )
        return provenance

    def _remote_client_for_run(self, provenance: dict[str, Any] | None):
        """Construct the immutable remote-worker client at most once per run."""

        lock = getattr(self, "_stats_lock", None)
        context = lock if lock is not None else nullcontext()
        with context:
            if not getattr(self, "_remote_client_initialized", False):
                self._remote_client = remote_client_from_provenance(provenance)
                self._remote_client_initialized = True
            return getattr(self, "_remote_client", None)

    def _projection_checkpoint_store_for_run(
        self, provenance: dict[str, Any] | None
    ) -> EXL3ProjectionCheckpointStore | None:
        """Construct one shared coordinator store and request index per run."""

        checkpoint_root = checkpoint_root_from_provenance(provenance)
        with self._stats_lock:
            if not getattr(
                self, "_projection_checkpoint_store_initialized", False
            ):
                self._projection_checkpoint_store = (
                    EXL3ProjectionCheckpointStore(checkpoint_root)
                    if checkpoint_root is not None
                    else None
                )
                self._projection_checkpoint_store_initialized = True
            store = getattr(self, "_projection_checkpoint_store", None)
        if store is not None and store.root != checkpoint_root:
            raise ValueError("EXL3 projection-checkpoint root changed during the run")
        return store

    def _distributed_local_quant_lock(self, device: torch.device) -> threading.Lock:
        """Serialize trellis search independently on each coordinator GPU."""

        device_key = str(device)
        with self._stats_lock:
            locks = getattr(self, "_distributed_local_quant_locks", None)
            if locks is None:
                locks = {}
                self._distributed_local_quant_locks = locks
            return locks.setdefault(device_key, threading.Lock())

    @staticmethod
    def _projection_assignment_key(module: NamedModule) -> str:
        """Return the durable scheduler identity for one independent projection."""

        return module.full_name

    def subset_forward_capture_context(
        self,
        *,
        layer_module: Module,
        subset: Dict[str, NamedModule],
    ):
        """Capture natural router exposure in the existing subset forward."""

        provenance = self._ledger_provenance()
        if not route_evidence_required(provenance):
            return nullcontext()
        targets = {
            task_name: identity
            for task_name, named_module in subset.items()
            if task_name in self.tasks
            and isinstance(getattr(named_module, "full_name", None), str)
            and (identity := routed_expert_identity(named_module.full_name)) is not None
        }
        if not targets:
            return nullcontext()
        family_ids = {
            (identity["block_namespace"], identity["logical_layer"])
            for identity in targets.values()
        }
        if len(family_ids) != 1:
            raise ValueError(
                "EXL3 natural-route capture requires one routed block per subset"
            )
        family_id = next(iter(family_ids))
        cache = getattr(self, "_natural_route_evidence_cache", None)
        if cache is None:
            cache = {}
            self._natural_route_evidence_cache = cache
        cached = cache.get(family_id)
        target_experts = {identity["expert"] for identity in targets.values()}
        if cached is not None and target_experts.issubset(cached):
            self._attach_natural_route_evidence(targets, cached)
            return nullcontext()
        return _EXL3NaturalRouteCapture(
            self,
            layer_module=layer_module,
            subset=subset,
        )

    def _commit_natural_route_capture(
        self,
        capture: _EXL3NaturalRouteCapture,
    ) -> None:
        """Bind one subset's route evidence to its projection capture tasks."""

        evidence_by_expert = capture.evidence_by_expert()
        family_id = next(
            iter(
                {
                    (identity["block_namespace"], identity["logical_layer"])
                    for identity in capture.targets.values()
                }
            )
        )
        cache = getattr(self, "_natural_route_evidence_cache", None)
        if cache is None:
            cache = {}
            self._natural_route_evidence_cache = cache
        previous_family = cache.get(family_id)
        if previous_family is not None:
            for expert, evidence in evidence_by_expert.items():
                previous = previous_family.get(expert)
                if previous is not None and previous != evidence:
                    raise RuntimeError(
                        f"EXL3 natural-route evidence changed for {family_id} expert {expert}"
                    )
            previous_family.update(evidence_by_expert)
            evidence_by_expert = previous_family
        else:
            cache[family_id] = evidence_by_expert
        self._attach_natural_route_evidence(capture.targets, evidence_by_expert)

    def _attach_natural_route_evidence(
        self,
        targets: dict[str, dict[str, Any]],
        evidence_by_expert: dict[int, dict[str, Any]],
    ) -> None:
        """Attach one immutable block distribution to projection tasks."""

        for task_name, identity in targets.items():
            task = self.tasks.get(task_name)
            if task is None:
                raise RuntimeError(
                    f"EXL3 route capture lost task `{task_name}` before commit"
                )
            evidence = evidence_by_expert.get(identity["expert"])
            if evidence is None:
                raise RuntimeError(
                    f"EXL3 route capture lacks expert {identity['expert']}"
                )
            previous = task.get("route_evidence")
            if previous is not None and previous != evidence:
                raise RuntimeError(
                    f"EXL3 route evidence changed for task `{task_name}`"
                )
            task["route_evidence"] = evidence

    def preprocess(self, module: NamedModule, fallback=None, **kwargs):
        """Builds the capture task and effective EXL3 config for one module."""

        del fallback, kwargs

        module_qcfg = clone_exllamav3_config_for_module(self.qcfg, module.full_name)
        if module_qcfg is None:
            return

        capture_qcfg = GPTQConfig(
            bits=max(1, module_qcfg.runtime_bits),
            group_size=-1,
            desc_act=False,
            sym=True,
            # This short-lived config owns only the in-memory Hessian capture
            # for one projection. Enabling the generic completed-module
            # offloader creates one unused TemporaryDirectory per projection
            # (768 per DeepSeek V4 MoE block) and repeats that setup every
            # layer. The enclosing EXL3 config remains responsible for actual
            # module offload through its explicitly configured shared path.
            offload_to_disk=False,
            device=module_qcfg.device,
            pack_dtype=module_qcfg.pack_dtype,
            act_group_aware=False,
        )

        task = GPTQ(module=module, qcfg=capture_qcfg)
        task.expected_nsamples = getattr(self, "total_calibration_tokens", None)
        task.quantizer.configure(perchannel=True)

        self.tasks[module.name] = {
            "capture": task,
            "qcfg": module_qcfg,
        }
        remote_client = self._remote_client_for_run(self._ledger_provenance())
        if remote_client is not None:
            if routed_expert_identity(module.full_name) is None:
                raise ValueError(
                    "EXL3 distributed dispatch only accepts routed-expert projections"
                )

    def is_skipped(self, module: NamedModule) -> bool:
        """Reports whether preprocessing omitted this module from EXL3 work."""

        return self.tasks.get(module.name, False) is False

    def pre_process_fwd_hook(
        self, name: str
    ) -> Callable[[Module, Tuple[torch.Tensor, ...], torch.Tensor], None]:
        """Returns the forward hook that feeds captured batches into the EXL3 task."""

        def tmp(module, inp: Tuple[torch.Tensor, ...], out: torch.Tensor):
            """Records one activation batch for the EXL3 capture task."""

            capture = self.tasks[name]["capture"]
            batch_idx = self.current_batch_index()
            capture.add_batch(inp[0].data, out.data, batch_index=batch_idx)
            del inp, out

        return tmp

    def _is_lm_head(self, module: NamedModule) -> bool:
        """Returns whether the named module corresponds to the model lm_head."""

        if module.full_name == self.lm_head_name:
            return True
        return module.full_name.endswith(f".{self.lm_head_name}")

    def _target_bits(self, module: NamedModule, module_qcfg: EXL3Config) -> int:
        """Chooses lm_head-specific bitwidth overrides when configured."""

        if self._is_lm_head(module) and module_qcfg.head_bits is not None:
            return max(1, int(module_qcfg.head_bits))
        return max(1, module_qcfg.runtime_bits)

    def _build_quant_args(
        self,
        module: NamedModule,
        module_qcfg: EXL3Config,
        device: torch.device,
    ) -> Dict[str, object]:
        """Builds the argument bundle passed into the EXL3 quantizer."""

        quant_args: Dict[str, object] = {
            "K": self._target_bits(module, module_qcfg),
            "devices": [device],
            "apply_out_scales": _OUT_SCALES_TO_ARG.get(module_qcfg.out_scales, None),
            "sigma_reg": _EXL3_SIGMA_REG,
            "seed": 787,
        }

        if module_qcfg.codebook == "mcg":
            quant_args["mcg"] = True
        elif module_qcfg.codebook == "mul1":
            quant_args["mul1"] = True

        return quant_args

    def _quant_input_weight(self, capture: GPTQ, device: torch.device) -> torch.Tensor:
        """Exports the captured dense weight matrix in EXL3 quantizer layout."""

        normalized = capture.clone_module(copy=True, device=device)
        return normalized.t().contiguous().to(torch.float32)

    def _restore_module_weight(
        self, module: NamedModule, quantized_weight: torch.Tensor
    ) -> torch.Tensor:
        """Reshapes the EXL3 output weight back into the wrapped module layout."""

        target = module.module if isinstance(module, NamedModule) else module

        if isinstance(target, transformers.Conv1D):
            return quantized_weight.contiguous().view_as(target.weight.data)

        if isinstance(target, (torch.nn.Linear, _ConvNd)):
            return quantized_weight.t().contiguous().view_as(target.weight.data)

        raise NotImplementedError(
            f"Unsupported EXL3 module type: {target.__class__.__name__}"
        )

    def process(
        self,
        module: NamedModule,
        device: torch.device = None,
        subset: Optional[Dict[str, NamedModule]] = None,
        previous_subset: Optional[Dict[str, NamedModule]] = None,
        subset_index: Optional[int] = None,
        subset_total: Optional[int] = None,
    ):
        """Dynamically lease one slot, then quantize and stage one projection."""

        del subset, previous_subset, subset_index, subset_total

        ledger_provenance = self._ledger_provenance()
        remote_client = self._remote_client_for_run(ledger_provenance)
        execution_lease = None
        execution_slot = None
        if remote_client is not None:
            if routed_expert_identity(module.full_name) is None:
                raise ValueError(
                    "EXL3 distributed dispatch only accepts routed-expert projections"
                )
            execution_lease = remote_client.acquire_slot(
                self._projection_assignment_key(module)
            )
            execution_slot = execution_lease.slot
        try:
            return self._process_on_slot(
                module=module,
                device=device,
                ledger_provenance=ledger_provenance,
                remote_client=remote_client,
                execution_slot=execution_slot,
                execution_lease=execution_lease,
            )
        finally:
            if execution_lease is not None:
                execution_lease.release()

    def _process_on_slot(
        self,
        *,
        module: NamedModule,
        device: torch.device | None,
        ledger_provenance: dict[str, Any] | None,
        remote_client,
        execution_slot,
        execution_lease: ExecutionSlotLease | None,
    ):
        """Run one projection while its physical execution slot is exclusively held."""

        base_title = f"Quantizing {module.name} in layer"
        self.draw_progress(base_title)

        task_entry = self.tasks[module.name]
        capture: GPTQ = task_entry["capture"]
        module_qcfg: EXL3Config = task_entry["qcfg"]

        target_device = device or get_device(module.module)
        if isinstance(execution_slot, CoordinatorSlot):
            target_device = execution_slot.device
        target_device = torch.device(target_device)
        if target_device.type != "cuda":
            raise ValueError("EXL3 quantization requires CUDA/HIP execution.")

        hessian = prepare_exl3_hessian(
            capture,
            target_device=target_device,
            module_full_name=module.full_name,
        )

        h_data = {
            "H": hessian,
            "count": capture.nsamples,
            "finalized": False,
        }

        remote_endpoint = None
        execution_contract = None
        projection_provenance = ledger_provenance
        if remote_client is not None:
            if routed_expert_identity(module.full_name) is None:
                raise ValueError(
                    "EXL3 remote dispatch only accepts routed-expert projections"
                )
            if not isinstance(execution_slot, (CoordinatorSlot, RemoteEndpoint)):
                raise ValueError("EXL3 distributed execution slot is invalid")
            remote_endpoint = (
                execution_slot if isinstance(execution_slot, RemoteEndpoint) else None
            )
            execution_contract = remote_client.execution_contract(execution_slot)
            projection_provenance = copy.deepcopy(ledger_provenance)
            projection_provenance["execution"] = copy.deepcopy(execution_contract)
        quant_args = self._build_quant_args(module, module_qcfg, target_device)
        input_weight = self._quant_input_weight(capture, target_device)
        checkpoint_store = self._projection_checkpoint_store_for_run(
            ledger_provenance
        )
        if remote_client is not None and checkpoint_store is None:
            raise ValueError(
                "EXL3 remote dispatch requires a coordinator projection checkpoint"
            )
        checkpoint_request = None
        checkpoint_result = None
        checkpoint_hit = False
        if checkpoint_store is not None:
            family_join = (
                ledger_provenance.get("family_join")
                if isinstance(ledger_provenance, dict)
                else None
            )
            quantizer_contract = {
                "bits": int(quant_args["K"]),
                "codebook": module_qcfg.codebook,
                "apply_out_scales": quant_args["apply_out_scales"],
                "sigma_reg": float(quant_args["sigma_reg"]),
                "seed": int(quant_args["seed"]),
                "hessian_capture": EXL3_HESSIAN_CAPTURE_CONTRACT,
                "hessian_numerical": EXL3_HESSIAN_NUMERICAL_CONTRACT,
                "hessian_symmetry": EXL3_HESSIAN_SYMMETRY_CONTRACT,
            }
            if execution_contract is not None:
                quantizer_contract["execution"] = copy.deepcopy(execution_contract)
            checkpoint_request = build_projection_request(
                module_full_name=module.full_name,
                layer_index=module.layer_index,
                input_weight=input_weight,
                hessian=hessian,
                sample_count=capture.nsamples,
                quantizer_contract=quantizer_contract,
                family_join=family_join,
                route_evidence=task_entry.get("route_evidence"),
            )
            checkpoint_store.reserve_module_request(checkpoint_request)
            loaded_checkpoint = checkpoint_store.load(checkpoint_request)
        else:
            loaded_checkpoint = None

        if loaded_checkpoint is None:
            if remote_endpoint is not None:
                remote_started = time.perf_counter()
                out_tensors, remote_result, transport = remote_client.quantize(
                    endpoint=remote_endpoint,
                    request_manifest=checkpoint_request,
                    input_weight=input_weight,
                    hessian=hessian,
                )
                remote_elapsed = time.perf_counter() - remote_started
                duration = remote_result.get("duration_seconds")
                proxy_err = remote_result.get("proxy_error")
                device_names = remote_result.get("device_names")
                quantizer_metrics = remote_result.get("quantizer_metrics")
                worker = remote_result.get("worker")
                worker_queue_wait_seconds = remote_result.get(
                    "worker_queue_wait_seconds"
                )
                if (
                    isinstance(duration, bool)
                    or not isinstance(duration, (int, float))
                    or isinstance(proxy_err, bool)
                    or not isinstance(proxy_err, (int, float))
                    or not isinstance(device_names, list)
                    or not all(isinstance(value, str) for value in device_names)
                    or not isinstance(quantizer_metrics, dict)
                    or not isinstance(worker, dict)
                    or isinstance(worker_queue_wait_seconds, bool)
                    or not isinstance(worker_queue_wait_seconds, (int, float))
                    or not math.isfinite(worker_queue_wait_seconds)
                    or worker_queue_wait_seconds < 0
                ):
                    raise RuntimeError(
                        f"EXL3 remote worker returned malformed results for `{module.full_name}`."
                    )
                execution_result = {
                    "kind": "remote_worker",
                    "worker": copy.deepcopy(worker),
                    "transport": copy.deepcopy(transport),
                    "coordinator_elapsed_seconds": remote_elapsed,
                    "scheduler_wait_seconds": execution_lease.wait_seconds,
                    "scheduler_new_assignment": execution_lease.new_assignment,
                    "scheduler_assignment_key": execution_lease.assignment_key,
                    "worker_queue_wait_seconds": worker_queue_wait_seconds,
                }
            else:
                wait_started = time.perf_counter()
                quant_lock = (
                    self._distributed_local_quant_lock(target_device)
                    if remote_client is not None
                    else nullcontext()
                )
                with quant_lock:
                    quant_started = time.perf_counter()
                    try:
                        _weight_q, proxy_err, out_tensors = quantize_exl3(
                            weight=input_weight,
                            H_data=h_data,
                            quant_args=quant_args,
                            return_weight_q=False,
                        )
                    except Exception as error:
                        raise RuntimeError(
                            exl3_quantization_failure_message(
                                error=error,
                                module_full_name=module.full_name,
                                request_sha256=(
                                    checkpoint_request.get("request_sha256")
                                    if checkpoint_request is not None
                                    else None
                                ),
                                hessian=hessian,
                                sample_count=capture.nsamples,
                                sigma_reg=float(quant_args["sigma_reg"]),
                            )
                        ) from error
                    del _weight_q
                    duration = time.perf_counter() - quant_started
                device_names = [str(device) for device in quant_args["devices"]]
                quantizer_metrics = quant_args.get("error_metrics")
                execution_result = {
                    "kind": "coordinator",
                    "coordinator_quant_lock_wait_seconds": quant_started - wait_started,
                }
                if execution_lease is not None:
                    execution_result.update(
                        {
                            "scheduler_wait_seconds": execution_lease.wait_seconds,
                            "scheduler_new_assignment": execution_lease.new_assignment,
                            "scheduler_assignment_key": execution_lease.assignment_key,
                        }
                    )
            if not isinstance(quantizer_metrics, dict):
                raise RuntimeError(
                    f"EXL3 quantizer returned no structured error metrics for `{module.full_name}`."
                )
            validate_exl3_hessian_metrics(
                quantizer_metrics,
                sample_count=capture.nsamples,
                sigma_reg=float(quant_args["sigma_reg"]),
            )
            if isinstance(proxy_err, torch.Tensor):
                proxy_err = proxy_err.item()
            encoded_bytes = sum(
                tensor.numel() * tensor.element_size()
                for tensor in out_tensors.values()
                if isinstance(tensor, torch.Tensor)
            )
            ledger_record = build_projection_record(
                module_full_name=module.full_name,
                layer_index=module.layer_index,
                bits=int(quant_args["K"]),
                codebook=module_qcfg.codebook,
                sample_count=capture.nsamples,
                duration_seconds=duration,
                encoded_bytes=encoded_bytes,
                device_names=device_names,
                quantizer_metrics=quantizer_metrics,
                provenance=projection_provenance,
                route_evidence=task_entry.get("route_evidence"),
            )
            checkpoint_result = {
                "duration_seconds": duration,
                "proxy_error": proxy_err,
                "device_names": device_names,
                "quantizer_metrics": quantizer_metrics,
                "ledger_record": ledger_record,
                "execution_contract": execution_contract,
                "execution_result": execution_result,
            }
            if checkpoint_store is not None:
                checkpoint_store.commit(
                    checkpoint_request,
                    out_tensors,
                    checkpoint_result,
                )
        else:
            out_tensors, checkpoint_result = loaded_checkpoint
            duration = checkpoint_result.get("duration_seconds")
            proxy_err = checkpoint_result.get("proxy_error")
            device_names = checkpoint_result.get("device_names")
            quantizer_metrics = checkpoint_result.get("quantizer_metrics")
            ledger_record = checkpoint_result.get("ledger_record")
            stored_execution_contract = checkpoint_result.get("execution_contract")
            execution_result = checkpoint_result.get("execution_result")
            encoded_bytes = sum(
                tensor.numel() * tensor.element_size()
                for tensor in out_tensors.values()
                if isinstance(tensor, torch.Tensor)
            )
            if (
                not isinstance(duration, (int, float))
                or isinstance(duration, bool)
                or not isinstance(device_names, list)
                or not all(isinstance(value, str) for value in device_names)
                or not isinstance(quantizer_metrics, dict)
                or not isinstance(ledger_record, dict)
                or stored_execution_contract != execution_contract
                or (remote_client is not None and not isinstance(execution_result, dict))
            ):
                raise ValueError("EXL3 projection checkpoint result is malformed")
            expected_ledger_record = build_projection_record(
                module_full_name=module.full_name,
                layer_index=module.layer_index,
                bits=int(quant_args["K"]),
                codebook=module_qcfg.codebook,
                sample_count=capture.nsamples,
                duration_seconds=duration,
                encoded_bytes=encoded_bytes,
                device_names=device_names,
                quantizer_metrics=quantizer_metrics,
                provenance=projection_provenance,
                route_evidence=task_entry.get("route_evidence"),
            )
            if ledger_record != expected_ledger_record:
                raise ValueError("EXL3 projection checkpoint ledger is inconsistent")
            checkpoint_hit = True

        if execution_lease is not None and isinstance(execution_slot, RemoteEndpoint):
            # The Spark is free as soon as the packed result is durable. Do not
            # strand it while the coordinator reconstructs/stages that result.
            execution_lease.release()

        # The packed result and its exact ledger are now durable when the run
        # opted into projection checkpoints. The journal remains the ordered
        # coordinator commit barrier before tensors enter async save state.
        with self._stats_lock:
            ledger_record_sha256 = append_exl3_error_journal(
                self.error_journal_path,
                ledger_record,
            )

        stream_payload = dict(out_tensors)
        if module.bias is not None:
            stream_payload["bias"] = module.bias.detach()
        module.stream_state_payload_to_cpu(stream_payload)

        runtime_weight = reconstruct_exl3_tensors(
            out_tensors,
            device=target_device,
            dtype=module.weight.dtype,
        )
        restored_weight = self._restore_module_weight(module, runtime_weight)
        module.weight.data = restored_weight.to(dtype=module.weight.dtype)

        workspace_summary = getattr(capture, "_borrow_workspace_last_summary", None)
        workspace_totals = getattr(capture, "_borrow_workspace_totals", None)

        if isinstance(proxy_err, str):
            loss_display = proxy_err
        else:
            loss_display = (
                f"{proxy_err:.10f}"
                if isinstance(proxy_err, (int, float))
                else "unknown"
            )

        stat = {
            PROCESS_LOG_NAME: self.name(),
            PROCESS_LOG_LAYER: module.layer_index,
            PROCESS_LOG_MODULE: module.name,
            MODULE_FEATURE_COLUMN: self.module_feature_summary(module),
            DTYPE_SIZE_COLUMN: self.module_dtype_size_summary(module),
            QUANT_LOG_LOSS: loss_display,
            QUANT_LOG_NSAMPLES: f"{capture.nsamples}",
            QUANT_LOG_DAMP: f"{_EXL3_SIGMA_REG:.5f}",
            PROCESS_LOG_TIME: f"{duration:.3f}",
            PROCESS_LOG_FWD_TIME: self.formatted_fwd_time(),
            PROCESS_USED_MEMORY: self.device_memory_report(),
            QUANT_LOG_LOSS_KIND: quantizer_metrics["reported_metric_kind"],
            "exl3_error_ledger_record": ledger_record,
            "exl3_error_journal": self.error_journal_path,
            "exl3_error_record_sha256": ledger_record_sha256,
            "exl3_projection_checkpoint": (
                checkpoint_request.get("request_sha256")
                if checkpoint_request is not None
                else None
            ),
            "exl3_projection_checkpoint_hit": checkpoint_hit,
            "exl3_execution_contract": execution_contract,
            "exl3_execution_result": execution_result,
        }

        if workspace_summary:
            requests = int(workspace_summary.get("requests", 0) or 0)
            if requests:
                hit_rate = float(workspace_summary.get("hit_rate", 0.0) or 0.0)
                chunk_rows = workspace_summary.get("chunk_rows")
                stat["workspace_cache_requests"] = str(requests)
                stat["workspace_cache_hit_rate"] = f"{hit_rate:.1%}"
                stat["workspace_stage_dtype"] = workspace_summary.get(
                    "staging_dtype", ""
                )
                if chunk_rows is not None:
                    stat["workspace_chunk_rows"] = str(chunk_rows)
        if workspace_totals:
            total_requests = int(workspace_totals.get("requests", 0) or 0)
            if total_requests:
                cumulative_hit_rate = (
                    float(workspace_totals.get("materialized_hits", 0) or 0.0)
                    / total_requests
                )
                stat["workspace_total_requests"] = str(total_requests)
                stat["workspace_total_hit_rate"] = f"{cumulative_hit_rate:.1%}"

        if self.qcfg.dynamic is not None:
            stat["dynamic"] = self.qcfg.dynamic_get(layer_name=module.full_name)

        with self._stats_lock:
            self.durations.append(duration)
            if isinstance(proxy_err, (int, float)):
                self.avg_losses.append(proxy_err)
            self.module_names.append(f"layer-{module.layer_index}-{module.name}")
            self.log.append(stat)

        self.log_new_row(stat)

        capture.free()
        del input_weight, runtime_weight, restored_weight, out_tensors, stream_payload

    def submodule_finalize(self, module: NamedModule, model: BaseQModel, **kwargs):
        """Builds and installs the ExLlamaV3 module from the staged tensors."""

        del kwargs

        module.stream_sync()

        tensors: Dict[str, torch.Tensor] = {}
        with self._stats_lock:
            module.state.pop("w", None)
            for tensor_name in (
                "trellis",
                "suh",
                "svh",
                "su",
                "sv",
                "bias",
                "mcg",
                "mul1",
            ):
                tensor = module.state.pop(tensor_name, None)
                if tensor is not None:
                    tensors[tensor_name] = tensor.clone()

        parent_key = getattr(module, "full_name", getattr(module, "name", None))
        with parent_module_lock(parent_key):
            create_exllamav3_module(
                module_root=model.model,
                name=module.full_name,
                submodule=module,
                tensors=tensors,
            )

        module.unregister_parameter("weight")
        if getattr(module, "bias", None) is not None:
            module.unregister_parameter("bias")

    def completed_layer_checkpoint_entries(
        self, layer_index: int
    ) -> list[dict[str, str]]:
        """Return the packed/error identities committed for one decoder layer."""

        entries: dict[str, dict[str, str]] = {}
        with self._stats_lock:
            log_snapshot = list(self.log)
        for stat in log_snapshot:
            if stat.get(PROCESS_LOG_LAYER) != layer_index:
                continue
            record = stat.get("exl3_error_ledger_record")
            request_sha256 = stat.get("exl3_projection_checkpoint")
            record_sha256 = stat.get("exl3_error_record_sha256")
            module_name = record.get("module") if isinstance(record, dict) else None
            if not all(
                isinstance(value, str) and value
                for value in (module_name, request_sha256, record_sha256)
            ):
                raise RuntimeError(
                    f"EXL3 layer {layer_index} has an uncommitted projection result"
                )
            entry = {
                "module": module_name,
                "request_sha256": request_sha256,
                "record_sha256": record_sha256,
            }
            previous = entries.setdefault(module_name, entry)
            if previous != entry:
                raise RuntimeError(
                    f"EXL3 layer {layer_index} has conflicting projection identities"
                )
        return [entries[name] for name in sorted(entries)]

    def restore_completed_layer_checkpoints(
        self,
        *,
        model: BaseQModel,
        layer_index: int,
        projection_entries: list[dict[str, str]],
        materialize_device: torch.device | str | None = None,
    ) -> None:
        """Install packed modules directly, without Hessian/corpus reconstruction."""

        ledger_provenance = self._ledger_provenance()
        checkpoint_root = checkpoint_root_from_provenance(ledger_provenance)
        if checkpoint_root is None:
            raise RuntimeError("EXL3 layer restore requires projection checkpoints")
        checkpoint_store = EXL3ProjectionCheckpointStore(checkpoint_root)
        family_join = (
            ledger_provenance.get("family_join")
            if isinstance(ledger_provenance, dict)
            else None
        )
        offload_path = getattr(self.qcfg, "offload_to_disk_path", None)
        if not getattr(self.qcfg, "offload_to_disk", False) or not offload_path:
            raise RuntimeError("EXL3 layer restore requires durable disk offload")

        restored_stats: list[dict[str, Any]] = []
        seen_modules: set[str] = set()
        for entry in sorted(projection_entries, key=lambda item: item.get("module", "")):
            module_name = entry.get("module")
            request_sha256 = entry.get("request_sha256")
            expected_record_sha256 = entry.get("record_sha256")
            if (
                not isinstance(module_name, str)
                or module_name in seen_modules
                or not isinstance(request_sha256, str)
                or not isinstance(expected_record_sha256, str)
            ):
                raise RuntimeError(
                    f"EXL3 layer {layer_index} restore index is malformed"
                )
            loaded = checkpoint_store.load_committed(request_sha256)
            if loaded is None:
                raise RuntimeError(
                    f"EXL3 packed checkpoint disappeared for `{module_name}`"
                )
            request, tensors, result = loaded
            ledger_record = result.get("ledger_record")
            calculated_record_sha256 = (
                sha256_bytes(canonical_json_bytes(ledger_record))
                if isinstance(ledger_record, dict)
                else None
            )
            if (
                request.get("module") != module_name
                or request.get("processor_layer_index") != layer_index
                or request.get("family_join") != family_join
                or calculated_record_sha256 != expected_record_sha256
                or not isinstance(ledger_record, dict)
                or ledger_record.get("module") != module_name
                or ledger_record.get("processor_layer_index") != layer_index
            ):
                raise RuntimeError(
                    f"EXL3 packed checkpoint identity differs for `{module_name}`"
                )
            actual_record_sha256 = append_exl3_error_journal(
                self.error_journal_path, ledger_record
            )
            try:
                source_module = model.model.get_submodule(module_name)
            except AttributeError as error:
                raise RuntimeError(
                    f"EXL3 restore target is absent: `{module_name}`"
                ) from error
            if isinstance(source_module, ExllamaV3Linear):
                raise RuntimeError(f"EXL3 restore target is already packed: `{module_name}`")
            bias = getattr(source_module, "bias", None)
            if bias is not None:
                if getattr(bias, "is_meta", False):
                    raise RuntimeError(
                        f"EXL3 restore cannot recover META bias for `{module_name}`"
                    )
                tensors = {**tensors, "bias": bias.detach().to(device="cpu")}
            relative_name = module_name.removeprefix(
                f"model.layers.{layer_index}."
            )
            named = NamedModule(
                source_module,
                name=relative_name,
                full_name=module_name,
                layer_index=layer_index,
            )
            packed = create_exllamav3_module(
                module_root=model.model,
                name=module_name,
                submodule=named,
                tensors=tensors,
            )
            if materialize_device is None:
                offload_to_disk(
                    model=model.model,
                    module=packed,
                    disk_path=offload_path,
                )
            else:
                target_device = torch.device(materialize_device)
                if target_device.type == "cpu":
                    raise RuntimeError(
                        "EXL3 catch-up replay requires an accelerator device"
                    )
                packed.to(device=target_device)
                setattr(packed, "target_device", target_device)

            duration = result.get("duration_seconds")
            proxy_error = result.get("proxy_error")
            quantizer_metrics = result.get("quantizer_metrics")
            if (
                isinstance(duration, bool)
                or not isinstance(duration, (int, float))
                or not isinstance(quantizer_metrics, dict)
            ):
                raise RuntimeError(
                    f"EXL3 packed checkpoint result is malformed for `{module_name}`"
                )
            restored_stats.append(
                {
                    PROCESS_LOG_NAME: self.name(),
                    PROCESS_LOG_LAYER: layer_index,
                    PROCESS_LOG_MODULE: relative_name,
                    MODULE_FEATURE_COLUMN: self.module_feature_summary(named),
                    DTYPE_SIZE_COLUMN: self.module_dtype_size_summary(named),
                    QUANT_LOG_LOSS: (
                        f"{proxy_error:.10f}"
                        if isinstance(proxy_error, (int, float))
                        else str(proxy_error)
                    ),
                    QUANT_LOG_NSAMPLES: str(request.get("sample_count")),
                    QUANT_LOG_DAMP: f"{_EXL3_SIGMA_REG:.5f}",
                    PROCESS_LOG_TIME: f"{float(duration):.3f}",
                    PROCESS_LOG_FWD_TIME: "0.000",
                    PROCESS_USED_MEMORY: "restored",
                    QUANT_LOG_LOSS_KIND: quantizer_metrics.get(
                        "reported_metric_kind", "unknown"
                    ),
                    "exl3_error_ledger_record": ledger_record,
                    "exl3_error_journal": self.error_journal_path,
                    "exl3_error_record_sha256": actual_record_sha256,
                    "exl3_projection_checkpoint": request_sha256,
                    "exl3_projection_checkpoint_hit": True,
                    "exl3_execution_contract": result.get("execution_contract"),
                    "exl3_execution_result": result.get("execution_result"),
                    "exl3_layer_boundary_restore": True,
                }
            )
            seen_modules.add(module_name)

        with self._stats_lock:
            for stat in restored_stats:
                self.log.append(stat)
                self.module_names.append(
                    f"layer-{layer_index}-{stat[PROCESS_LOG_MODULE]}"
                )
                proxy_error = stat[QUANT_LOG_LOSS]
                try:
                    self.avg_losses.append(float(proxy_error))
                except (TypeError, ValueError):
                    pass
                self.durations.append(float(stat[PROCESS_LOG_TIME]))

    def offload_restored_layer_checkpoints(
        self,
        *,
        model: BaseQModel,
        layer_index: int,
    ) -> None:
        """Offload a replayed packed layer before its boundary is promoted."""

        offload_path = getattr(self.qcfg, "offload_to_disk_path", None)
        if not getattr(self.qcfg, "offload_to_disk", False) or not offload_path:
            raise RuntimeError("EXL3 layer restore requires durable disk offload")
        entries = self.completed_layer_checkpoint_entries(layer_index)
        if not entries:
            raise RuntimeError(
                f"EXL3 restored layer {layer_index} contains no projections"
            )
        for entry in entries:
            module_name = entry["module"]
            packed = model.model.get_submodule(module_name)
            if not isinstance(packed, ExllamaV3Linear):
                raise RuntimeError(
                    f"EXL3 restored layer target is not packed: `{module_name}`"
                )
            offload_to_disk(
                model=model.model,
                module=packed,
                disk_path=offload_path,
            )

    def finalize(self, model: BaseQModel, **kwargs):
        """Marks the model as EXL3-quantized and runs shared finalization logic."""

        model.quantized = True
        model.quantize_config.method = METHOD.EXL3
        model.quantize_config.format = FORMAT.EXL3
        model.qlinear_kernel = ExllamaV3Linear
        super().finalize(model=model, **kwargs)

    def verify_calibration_dataset(self, processor_index: int) -> bool:
        """Ensures EXL3 received calibration data before the quantization loop starts."""

        del processor_index
        if self.calibration_dataset is None:
            raise ValueError("EXL3Processor's calibration_dataset must be provided.")
        return True

    def name(self) -> str:
        """Returns the processor label used in logs and lifecycle reporting."""

        return "exl3"


__all__ = ["EXL3Processor", "clone_exllamav3_config_for_module"]
