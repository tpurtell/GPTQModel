# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Exact learned-router ranking used by EXL3 recovery capture and replay."""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn


RECOVERY_SCORE_FN_ATTR = "_gptqmodel_recovery_score_fn"
RECOVERY_GROUP_POLICY_ATTR = "_gptqmodel_recovery_group_policy"
RECOVERY_GROUP_POLICY_CONTRACT = "gptqmodel.learned-router-group-policy-v1"
ROUTER_CANDIDATE_CAPTURE_PAYLOAD_CONTRACT = (
    "gptqmodel.exl3-router-candidate-capture-v2"
)


def _router_score_fn(router: nn.Module) -> Callable[[torch.Tensor], torch.Tensor]:
    score_fn = getattr(router, RECOVERY_SCORE_FN_ATTR, None)
    if score_fn is None:
        score_fn = getattr(router, "score_fn", None)
    if not callable(score_fn):
        raise RuntimeError(
            "EXL3 learned-router recovery requires an explicit score function"
        )
    return score_fn


def _apply_group_policy(
    router: nn.Module,
    choice_scores: torch.Tensor,
) -> torch.Tensor:
    policy = getattr(router, RECOVERY_GROUP_POLICY_ATTR, None)
    if policy is None:
        return choice_scores
    if not isinstance(policy, dict):
        raise RuntimeError("EXL3 learned-router group policy is malformed")
    num_groups = policy.get("num_groups")
    topk_groups = policy.get("topk_groups")
    if (
        policy.get("contract") != RECOVERY_GROUP_POLICY_CONTRACT
        or isinstance(num_groups, bool)
        or not isinstance(num_groups, int)
        or num_groups <= 0
        or isinstance(topk_groups, bool)
        or not isinstance(topk_groups, int)
        or not 0 < topk_groups <= num_groups
        or choice_scores.shape[-1] % num_groups != 0
        or choice_scores.shape[-1] // num_groups < 2
    ):
        raise RuntimeError("EXL3 learned-router group policy is invalid")
    experts_per_group = choice_scores.shape[-1] // num_groups
    group_scores = (
        choice_scores.view(-1, num_groups, experts_per_group)
        .topk(2, dim=-1)[0]
        .sum(dim=-1)
    )
    group_indices = torch.topk(
        group_scores,
        k=topk_groups,
        dim=-1,
        sorted=False,
    ).indices
    group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
    group_mask.scatter_(1, group_indices, True)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(-1, num_groups, experts_per_group)
        .reshape_as(choice_scores)
    )
    return choice_scores.masked_fill(~score_mask, float("-inf"))


def learned_router_ranked_choices(
    router: nn.Module,
    logits: torch.Tensor,
    *,
    rank_max: int,
    selected_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact corrected router ranks and verify them against live top-k.

    Model adapters must expose non-standard score and group-selection rules via
    the attributes above. Failing closed is intentional: zero-width candidate
    evidence cannot safely support a later under-coverage recovery.
    """

    correction = getattr(router, "e_score_correction_bias", None)
    if (
        not isinstance(logits, torch.Tensor)
        or logits.ndim != 2
        or isinstance(rank_max, bool)
        or not isinstance(rank_max, int)
        or not 0 < rank_max <= logits.shape[-1]
        or not isinstance(correction, torch.Tensor)
        or correction.ndim != 1
        or correction.numel() != logits.shape[-1]
    ):
        raise RuntimeError("EXL3 learned-router recovery geometry is invalid")

    scores = _router_score_fn(router)(logits.float())
    if not isinstance(scores, torch.Tensor) or scores.shape != logits.shape:
        raise RuntimeError("EXL3 learned-router score function output is invalid")
    correction = correction.to(device=logits.device, dtype=torch.float32)
    scores = scores.to(device=logits.device, dtype=torch.float32)
    scores_are_finite = torch.isfinite(scores).all() & torch.isfinite(correction).all()
    choice_scores = _apply_group_policy(
        router,
        scores + correction,
    )
    ranked_scores, ranked_indices = torch.topk(
        choice_scores,
        rank_max,
        dim=-1,
        largest=True,
        sorted=True,
    )
    if not bool(
        (scores_are_finite & torch.isfinite(ranked_scores[:, -1]).all()).item()
    ):
        raise RuntimeError(
            "EXL3 learned-router scores are non-finite or the group policy "
            "exposes fewer experts than rank_max"
        )

    if selected_indices is not None:
        if (
            not isinstance(selected_indices, torch.Tensor)
            or selected_indices.ndim != 2
            or selected_indices.shape[0] != logits.shape[0]
            or not 0 < selected_indices.shape[1] <= rank_max
            or selected_indices.dtype != torch.int64
        ):
            raise RuntimeError(
                "EXL3 learned-router selected indices have invalid geometry"
            )
        expected = torch.sort(
            ranked_indices[:, : selected_indices.shape[1]].to(torch.int64),
            dim=-1,
        ).values
        actual = torch.sort(
            selected_indices.to(device=expected.device),
            dim=-1,
        ).values
        if not torch.equal(expected, actual):
            raise RuntimeError(
                "EXL3 recovery ranking does not reproduce the live router top-k"
            )

    return ranked_scores, ranked_indices


def bind_sigmoid_grouped_router_recovery(router: nn.Module) -> None:
    """Bind the score and group policy implemented by GLM's learned router."""

    num_groups = getattr(router, "num_group", None)
    topk_groups = getattr(router, "topk_group", None)
    if (
        isinstance(num_groups, bool)
        or not isinstance(num_groups, int)
        or isinstance(topk_groups, bool)
        or not isinstance(topk_groups, int)
    ):
        raise RuntimeError("GLM learned router lacks its group-selection geometry")
    setattr(router, RECOVERY_SCORE_FN_ATTR, torch.sigmoid)
    setattr(
        router,
        RECOVERY_GROUP_POLICY_ATTR,
        {
            "contract": RECOVERY_GROUP_POLICY_CONTRACT,
            "num_groups": num_groups,
            "topk_groups": topk_groups,
        },
    )


__all__ = [
    "RECOVERY_GROUP_POLICY_ATTR",
    "RECOVERY_GROUP_POLICY_CONTRACT",
    "RECOVERY_SCORE_FN_ATTR",
    "ROUTER_CANDIDATE_CAPTURE_PAYLOAD_CONTRACT",
    "bind_sigmoid_grouped_router_recovery",
    "learned_router_ranked_choices",
]
