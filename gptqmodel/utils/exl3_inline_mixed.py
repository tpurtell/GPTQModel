# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Deterministic, crash-consistent inline EXL3 K2/K3 tier selection."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable


INLINE_MIXED_META_KEY = "ds4rt_inline_mixed"
INLINE_MIXED_SCHEMA = "gptqmodel.exl3-inline-mixed"
INLINE_MIXED_SCHEMA_VERSION = 1
INLINE_MIXED_SCORE = (
    "k2-hessian-weighted-relative-error-times-natural-gate-squared-mass-v1"
)
INLINE_MIXED_CHECKPOINT_ROLE = "inline_mixed"
PROJECTION_ORDER = ("w1", "w3", "w2")
PROJECTION_NAMES = {"w1": "gate", "w3": "up", "w2": "down"}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"EXL3 inline-mixed {label} must be a positive integer")
    return value


@dataclass(frozen=True)
class InlineMixedPolicy:
    """Validated immutable policy for one base or MTP quantization namespace."""

    namespace: str
    base_bits: int
    upgrade_bits: int
    extra_bits: Fraction
    projection_ratio: tuple[int, int, int]
    tier_plan_root: Path
    score_kind: str = INLINE_MIXED_SCORE

    @property
    def target_bpw(self) -> Fraction:
        return Fraction(self.base_bits, 1) + self.extra_bits

    @property
    def policy_body(self) -> dict[str, Any]:
        return {
            "schema": INLINE_MIXED_SCHEMA,
            "schema_version": INLINE_MIXED_SCHEMA_VERSION,
            "namespace": self.namespace,
            "base_bits": self.base_bits,
            "upgrade_bits": self.upgrade_bits,
            "extra_bits": {
                "numerator": self.extra_bits.numerator,
                "denominator": self.extra_bits.denominator,
            },
            "target_bpw": (
                f"{self.target_bpw.numerator}/{self.target_bpw.denominator}"
            ),
            "projection_ratio": {
                projection: ratio
                for projection, ratio in zip(PROJECTION_ORDER, self.projection_ratio)
            },
            "score_kind": self.score_kind,
        }

    @property
    def policy_sha256(self) -> str:
        return sha256_json(self.policy_body)

    def namespace_quotas(
        self, *, layer_count: int, experts_per_layer: int
    ) -> dict[str, int]:
        """Return exact largest-remainder projection totals for the namespace."""

        _positive_integer(layer_count, "layer_count")
        _positive_integer(experts_per_layer, "experts_per_layer")
        if self.upgrade_bits - self.base_bits != 1:
            raise ValueError("EXL3 inline-mixed currently requires adjacent K tiers")
        candidate_count = layer_count * experts_per_layer * len(PROJECTION_ORDER)
        ideal_upgrades = Fraction(candidate_count) * self.extra_bits
        # Nearest representable projection count, with deterministic half-up ties.
        total_upgrades = (
            2 * ideal_upgrades.numerator + ideal_upgrades.denominator
        ) // (2 * ideal_upgrades.denominator)
        if not 0 <= total_upgrades <= candidate_count:
            raise ValueError("EXL3 inline-mixed target is outside the K2/K3 range")

        ratio_sum = sum(self.projection_ratio)
        ideals = {
            projection: Fraction(total_upgrades * ratio, ratio_sum)
            for projection, ratio in zip(PROJECTION_ORDER, self.projection_ratio)
        }
        quotas = {
            projection: ideal.numerator // ideal.denominator
            for projection, ideal in ideals.items()
        }
        remainder = total_upgrades - sum(quotas.values())
        ranked = sorted(
            PROJECTION_ORDER,
            key=lambda projection: (
                -(ideals[projection] - quotas[projection]),
                PROJECTION_ORDER.index(projection),
            ),
        )
        for projection in ranked[:remainder]:
            quotas[projection] += 1
        if any(quota > layer_count * experts_per_layer for quota in quotas.values()):
            raise ValueError(
                "EXL3 inline-mixed projection ratio exceeds one projection class"
            )
        return quotas

    def layer_quotas(
        self,
        *,
        layer_index: int,
        layer_count: int,
        experts_per_layer: int,
    ) -> dict[str, int]:
        """Spread exact namespace quotas causally across decoder layers."""

        if not 0 <= layer_index < layer_count:
            raise ValueError("EXL3 inline-mixed layer index is invalid")
        totals = self.namespace_quotas(
            layer_count=layer_count,
            experts_per_layer=experts_per_layer,
        )
        return {
            projection: (
                ((layer_index + 1) * total) // layer_count
                - (layer_index * total) // layer_count
            )
            for projection, total in totals.items()
        }


def inline_mixed_policy(meta: Any) -> InlineMixedPolicy | None:
    """Parse the private exact-rational policy without accepting float BPW."""

    if not isinstance(meta, dict) or INLINE_MIXED_META_KEY not in meta:
        return None
    raw = meta[INLINE_MIXED_META_KEY]
    if not isinstance(raw, dict):
        raise ValueError("EXL3 inline-mixed metadata must be an object")
    extra = raw.get("extra_bits")
    ratio = raw.get("projection_ratio")
    root = raw.get("tier_plan_root")
    namespace = raw.get("namespace")
    if (
        raw.get("schema") != INLINE_MIXED_SCHEMA
        or raw.get("schema_version") != INLINE_MIXED_SCHEMA_VERSION
        or namespace not in {"base", "mtp"}
        or not isinstance(extra, dict)
        or not isinstance(ratio, dict)
        or set(ratio) != set(PROJECTION_ORDER)
        or not isinstance(root, str)
        or not root
        or raw.get("score_kind", INLINE_MIXED_SCORE) != INLINE_MIXED_SCORE
    ):
        raise ValueError("EXL3 inline-mixed metadata has an invalid contract")
    numerator = _positive_integer(extra.get("numerator"), "extra numerator")
    denominator = _positive_integer(extra.get("denominator"), "extra denominator")
    fraction = Fraction(numerator, denominator)
    if not 0 < fraction < 1:
        raise ValueError("EXL3 inline-mixed extra bits must be between zero and one")
    base_bits = _positive_integer(raw.get("base_bits"), "base_bits")
    upgrade_bits = _positive_integer(raw.get("upgrade_bits"), "upgrade_bits")
    policy = InlineMixedPolicy(
        namespace=namespace,
        base_bits=base_bits,
        upgrade_bits=upgrade_bits,
        extra_bits=fraction,
        projection_ratio=tuple(
            _positive_integer(ratio[projection], f"{projection} ratio")
            for projection in PROJECTION_ORDER
        ),
        tier_plan_root=Path(root).expanduser().resolve(),
    )
    if policy.upgrade_bits != policy.base_bits + 1:
        raise ValueError("EXL3 inline-mixed requires adjacent integer tiers")
    return policy


def projection_score(record: dict[str, Any]) -> float:
    """Compute the accepted K2-only risk proxy from one projection ledger."""

    metrics = record.get("quantizer_metrics")
    routes = record.get("route_evidence")
    error = (
        metrics.get("hessian_weighted_relative_error")
        if isinstance(metrics, dict)
        else None
    )
    gate_mass = (
        routes.get("expert_gate_squared_mass_fraction")
        if isinstance(routes, dict)
        else None
    )
    if (
        not isinstance(error, (int, float))
        or isinstance(error, bool)
        or not math.isfinite(float(error))
        or float(error) < 0
        or not isinstance(gate_mass, (int, float))
        or isinstance(gate_mass, bool)
        or not math.isfinite(float(gate_mass))
        or not 0 <= float(gate_mass) <= 1
    ):
        raise ValueError("EXL3 inline-mixed candidate lacks finite error/route evidence")
    return float(error) * float(gate_mass)


def build_layer_tier_plan(
    *,
    policy: InlineMixedPolicy,
    layer_index: int,
    layer_count: int,
    candidate_records: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Select the highest-risk K2 projections under fixed projection quotas."""

    records = list(candidate_records)
    identities: dict[tuple[int, str], tuple[dict[str, Any], float]] = {}
    experts: set[int] = set()
    for record in records:
        expert = record.get("expert")
        projection = record.get("projection")
        if (
            record.get("record_kind") != "projection"
            or record.get("block_namespace") != policy.namespace
            or record.get("logical_layer") != layer_index
            or record.get("bits") != policy.base_bits
            or isinstance(expert, bool)
            or not isinstance(expert, int)
            or expert < 0
            or projection not in PROJECTION_ORDER
        ):
            raise ValueError("EXL3 inline-mixed candidate identity is invalid")
        key = (expert, projection)
        if key in identities:
            raise ValueError("EXL3 inline-mixed candidate projection is duplicated")
        identities[key] = (record, projection_score(record))
        experts.add(expert)
    if not experts or experts != set(range(max(experts) + 1)):
        raise ValueError("EXL3 inline-mixed expert IDs are incomplete")
    experts_per_layer = len(experts)
    if len(identities) != experts_per_layer * len(PROJECTION_ORDER):
        raise ValueError("EXL3 inline-mixed layer has incomplete projection coverage")

    quotas = policy.layer_quotas(
        layer_index=layer_index,
        layer_count=layer_count,
        experts_per_layer=experts_per_layer,
    )
    selected: list[dict[str, Any]] = []
    for projection in PROJECTION_ORDER:
        ranked = sorted(
            (
                (score, expert, record)
                for (expert, candidate_projection), (record, score) in identities.items()
                if candidate_projection == projection
            ),
            key=lambda item: (-item[0], item[1]),
        )
        for score, expert, record in ranked[: quotas[projection]]:
            selected.append(
                {
                    "module": record["module"],
                    "expert": expert,
                    "projection": projection,
                    "score": score,
                    "candidate_record_sha256": sha256_json(record),
                }
            )
    selected.sort(key=lambda item: (PROJECTION_ORDER.index(item["projection"]), item["expert"]))
    body = {
        **policy.policy_body,
        "policy_sha256": policy.policy_sha256,
        "layer_index": layer_index,
        "layer_count": layer_count,
        "experts_per_layer": experts_per_layer,
        "quotas": quotas,
        "selected": selected,
    }
    return {**body, "tier_plan_sha256": sha256_json(body)}


class InlineMixedTierPlanStore:
    """Atomically publish immutable per-layer tier maps before K3 execution."""

    def __init__(self, policy: InlineMixedPolicy) -> None:
        self.policy = policy
        self.root = policy.tier_plan_root

    def path(self, layer_index: int) -> Path:
        if layer_index < 0:
            raise ValueError("EXL3 inline-mixed layer index is invalid")
        return self.root / self.policy.namespace / f"layer-{layer_index:06d}.json"

    def commit(self, plan: dict[str, Any]) -> dict[str, Any]:
        layer_index = plan.get("layer_index")
        digest = plan.get("tier_plan_sha256")
        body = {key: value for key, value in plan.items() if key != "tier_plan_sha256"}
        if (
            isinstance(layer_index, bool)
            or not isinstance(layer_index, int)
            or not isinstance(digest, str)
            or sha256_json(body) != digest
            or plan.get("policy_sha256") != self.policy.policy_sha256
            or plan.get("namespace") != self.policy.namespace
        ):
            raise ValueError("EXL3 inline-mixed tier plan is invalid")
        path = self.path(layer_index)
        payload = canonical_json_bytes(plan) + b"\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.parent.is_symlink():
            raise ValueError("EXL3 inline-mixed tier-plan root is unsafe")
        if path.exists():
            if not path.is_file() or path.is_symlink() or path.read_bytes() != payload:
                raise ValueError("EXL3 inline-mixed tier plan changed on resume")
            return plan
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", dir=path.parent
        )
        try:
            with os.fdopen(descriptor, "wb") as target:
                target.write(payload)
                target.flush()
                os.fsync(target.fileno())
            os.replace(temporary_name, path)
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except BaseException:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise
        return plan


__all__ = [
    "INLINE_MIXED_CHECKPOINT_ROLE",
    "INLINE_MIXED_META_KEY",
    "INLINE_MIXED_SCHEMA",
    "INLINE_MIXED_SCHEMA_VERSION",
    "INLINE_MIXED_SCORE",
    "InlineMixedPolicy",
    "InlineMixedTierPlanStore",
    "PROJECTION_NAMES",
    "PROJECTION_ORDER",
    "build_layer_tier_plan",
    "inline_mixed_policy",
    "projection_score",
    "sha256_json",
]
