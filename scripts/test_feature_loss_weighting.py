#!/usr/bin/env python3
"""Smoke-test the VQ-VAE per-feature reconstruction-loss weighting."""

from __future__ import annotations

import math

import torch
from lightning import LightningModule

from heptokens.models.vq_vae import LitVqVae


class LossHarness(LitVqVae):
    """Minimal model that exercises the production loss implementation."""

    def __init__(self, weights: list[float] | None):
        LightningModule.__init__(self)
        self.feature_loss_weights = weights

    def decode(self, z_q: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return z_q


def evaluate(weights: list[float] | None) -> tuple[float, torch.Tensor]:
    # The first four features are kinematics and have an absolute error of 2.
    # The other four features have an absolute error of 1.
    reconstruction = torch.tensor(
        [[[2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0]]],
        requires_grad=True,
    )
    batch = {
        "csts": torch.zeros_like(reconstruction),
        "mask": torch.ones((1, 1), dtype=torch.bool),
    }
    loss, returned = LossHarness(weights)._reconstruction_loss_and_prediction(
        reconstruction,
        batch,
    )
    assert returned is reconstruction
    loss.backward()
    return float(loss.detach()), reconstruction.grad.detach().abs()[0, 0]


def assert_close(actual: float, expected: float) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-7):
        raise AssertionError(f"expected {expected}, got {actual}")


def main() -> None:
    cases = [
        ("existing unweighted", None, 1.5, 1.0),
        ("explicit 1x", [1.0] * 8, 1.5, 1.0),
        ("2x kinematics", [2.0] * 4 + [1.0] * 4, 5.0 / 3.0, 2.0),
        ("4x kinematics", [4.0] * 4 + [1.0] * 4, 1.8, 4.0),
    ]

    print("Feature order: pt, eta, phi, charge, LHMedium, LHTight, ptvarcone30, topoetcone20")
    for label, weights, expected_loss, expected_gradient_ratio in cases:
        loss, gradients = evaluate(weights)
        gradient_ratio = float(gradients[:4].mean() / gradients[4:].mean())
        assert_close(loss, expected_loss)
        assert_close(gradient_ratio, expected_gradient_ratio)
        print(
            f"PASS  {label:21s} loss={loss:.6f} "
            f"kinematic/other gradient ratio={gradient_ratio:.1f}"
        )

    try:
        evaluate([2.0] * 7)
    except ValueError as exc:
        print(f"PASS  invalid length rejected: {exc}")
    else:
        raise AssertionError("A seven-element feature weight vector was not rejected")

    print("\nAll feature-loss weighting checks passed.")


if __name__ == "__main__":
    main()
