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
    quantize_exl3,
    reconstruct_exl3_tensors,
)
from ..looper.loop_processor import (
    DTYPE_SIZE_COLUMN,
    ExecutionConfig,
    MODULE_FEATURE_COLUMN,
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
from ..quantization.config import EXL3Config, FORMAT, GPTQConfig, METHOD
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
from ..utils.exllamav3 import create_exllamav3_module
from ..utils.exl3_projection_checkpoint import (
    EXL3ProjectionCheckpointStore,
    build_projection_request,
    checkpoint_root_from_provenance,
)
from ..utils.logger import setup_logger
from ..utils.module_locks import parent_module_lock


setup_logger()

_EXL3_SIGMA_REG = 0.025
_OUT_SCALES_TO_ARG = {
    "always": True,
    "never": False,
    "auto": None,
    None: None,
}


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
        flat_indices = indices.detach().reshape(-1).to(dtype=torch.int64)
        flat_weights = weights.detach().reshape(-1).to(dtype=torch.float64)
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
        counts = counts.to(device="cpu", dtype=torch.int64)
        gate_sums = gate_sums.to(device="cpu", dtype=torch.float64)
        gate_squared_mass = gate_squared_mass.to(device="cpu", dtype=torch.float64)

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
        """Runs EXL3 quantization for one module and stages its packed tensors."""

        del subset, previous_subset, subset_index, subset_total

        base_title = f"Quantizing {module.name} in layer"
        self.draw_progress(base_title)

        task_entry = self.tasks[module.name]
        capture: GPTQ = task_entry["capture"]
        module_qcfg: EXL3Config = task_entry["qcfg"]

        target_device = device or get_device(module.module)
        target_device = torch.device(target_device)
        if target_device.type != "cuda":
            raise ValueError("EXL3 quantization requires CUDA/HIP execution.")

        start_time = time.perf_counter()
        capture.finalize_hessian(target_device=target_device)
        hessian = capture.H
        if hessian is None:
            raise RuntimeError(
                f"EXL3 failed to capture Hessian for module `{module.full_name}`."
            )
        if capture.nsamples <= 0:
            raise RuntimeError(
                f"EXL3 captured no calibration activations for module `{module.full_name}`."
            )

        h_data = {
            "H": hessian,
            "count": capture.nsamples,
            "finalized": False,
        }

        ledger_provenance = self._ledger_provenance()
        quant_args = self._build_quant_args(module, module_qcfg, target_device)
        input_weight = self._quant_input_weight(capture, target_device)
        checkpoint_root = checkpoint_root_from_provenance(ledger_provenance)
        checkpoint_store = (
            EXL3ProjectionCheckpointStore(checkpoint_root)
            if checkpoint_root is not None
            else None
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
            checkpoint_request = build_projection_request(
                module_full_name=module.full_name,
                layer_index=module.layer_index,
                input_weight=input_weight,
                hessian=hessian,
                sample_count=capture.nsamples,
                quantizer_contract={
                    "bits": int(quant_args["K"]),
                    "codebook": module_qcfg.codebook,
                    "apply_out_scales": quant_args["apply_out_scales"],
                    "sigma_reg": float(quant_args["sigma_reg"]),
                    "seed": int(quant_args["seed"]),
                },
                family_join=family_join,
                route_evidence=task_entry.get("route_evidence"),
            )
            loaded_checkpoint = checkpoint_store.load(checkpoint_request)
        else:
            loaded_checkpoint = None

        if loaded_checkpoint is None:
            _weight_q, proxy_err, out_tensors = quantize_exl3(
                weight=input_weight,
                H_data=h_data,
                quant_args=quant_args,
                return_weight_q=False,
            )
            del _weight_q
            duration = time.perf_counter() - start_time
            quantizer_metrics = quant_args.get("error_metrics")
            if not isinstance(quantizer_metrics, dict):
                raise RuntimeError(
                    f"EXL3 quantizer returned no structured error metrics for `{module.full_name}`."
                )
            if isinstance(proxy_err, torch.Tensor):
                proxy_err = proxy_err.item()
            device_names = [str(device) for device in quant_args["devices"]]
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
                provenance=ledger_provenance,
                route_evidence=task_entry.get("route_evidence"),
            )
            checkpoint_result = {
                "duration_seconds": duration,
                "proxy_error": proxy_err,
                "device_names": device_names,
                "quantizer_metrics": quantizer_metrics,
                "ledger_record": ledger_record,
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
                provenance=ledger_provenance,
                route_evidence=task_entry.get("route_evidence"),
            )
            if ledger_record != expected_ledger_record:
                raise ValueError("EXL3 projection checkpoint ledger is inconsistent")
            checkpoint_hit = True

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
