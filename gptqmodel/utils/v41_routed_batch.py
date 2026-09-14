"""Reusable owned FFN input frontier for efficient expert-subset calibration."""
from dataclasses import dataclass

import torch

from .v41_replay import owned_tree


@dataclass
class V41RoutedBatch:
    hidden: torch.Tensor
    logits: torch.Tensor
    weights: torch.Tensor
    indices: torch.Tensor

    @classmethod
    @torch.no_grad()
    def from_replay(cls, block, state, device):
        """Run attention once, stop just after routing and before expert GEMMs.

        Every expert subset reuses these identical normalized FFN inputs and
        natural routes. This is not the post-quantization propagation frontier.
        """
        captured = []

        class Captured(Exception):
            pass

        def hook(module, args, output):
            logits, weights, indices = output
            captured.append(cls(owned_tree(args[0].reshape(-1, args[0].shape[-1]), "cpu"),
                                owned_tree(logits, "cpu"), owned_tree(weights, "cpu"),
                                owned_tree(indices, "cpu")))
            raise Captured()

        handle = block.mlp.gate.register_forward_hook(hook)
        try:
            try:
                state.advance(block, device)
            except Captured:
                pass
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError("V4.1 routed frontier was not captured exactly once")
        return captured[0]
