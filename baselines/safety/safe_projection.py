"""
baselines/safety/safe_projection.py  —  Concept-Conditioned Safe Projection
==============================================================================
Implements Layer 2 of the HA-C2G architecture: a **differentiable,
concept-conditioned action projection** that sits between the policy
network's raw output and the final action.

This layer provides two guarantees:

  Step 1 — Architectural bound via sigmoid:
    a_bounded = σ(a_raw) ∈ (0, 1)^{N_a}
    For C2G actions [throttle, pump, hvac] ∈ [0,1] this eliminates
    out-of-bounds actions by construction. For BESS ∈ [-1,1] we use
    2σ(·) − 1 instead.

  Step 2 — Concept-conditioned gate:
    g = σ(MLP_φ(concepts)) ∈ (0, 1)^{N_a}
    a_final = a_bounded ⊙ g
    The gate learns WHEN to attenuate actions based on concept values.

Key design: the gate is **actively trained** (not just near-pass-through):
  - An auxiliary gate supervision loss provides an explicit training signal
  - Gate target: g* = 1 − α · max(cooling_demand_A, cooling_demand_B)
    meaning: when cooling demand is high, throttle down the batch action
  - The gate is DIFFERENTIABLE: gradients flow through both sigmoid and
    gate during PPO training, so the policy learns to cooperate.

This is adapted from the SC26 HA-CompOpt paper's Safe Projection Layer
with the gate supervision experiment (Section 5.4) integrated into the
main training loop.

References
----------
  [SC26]  HA-CompOpt: Section 3.2 (Safe Projection Layer)
          and Section 5.4 (Gate Supervision Experiment)
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


if _TORCH_AVAILABLE:

    class SafeProjectionGate(nn.Module):
        """
        Concept-conditioned action gate (the trainable component).

        Maps concepts(K) → per-action scaling factor ∈ (0, 1)^{N_a}.
        Initialised near pass-through (bias=2.0 → σ(2.0) ≈ 0.88) but
        **actively trained** via gate supervision loss.

        Parameters
        ----------
        concept_dim : int
            Number of input concepts.
        action_dim : int
            Number of action dimensions (4 for C2GFastEnv).
        hidden : int
            Hidden layer width.
        init_bias : float
            Initial bias for the output layer. Higher = more pass-through
            at initialisation.
        """

        def __init__(self, concept_dim: int, action_dim: int,
                     hidden: int = 32, init_bias: float = 2.0):
            super().__init__()
            self.action_dim = action_dim
            self.gate = nn.Sequential(
                nn.Linear(concept_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, action_dim),
                nn.Sigmoid(),
            )
            # Initialise near pass-through
            nn.init.constant_(self.gate[2].bias, init_bias)

        def forward(self, concepts: torch.Tensor) -> torch.Tensor:
            """
            Parameters
            ----------
            concepts : (batch, K) tensor of concept predictions.

            Returns
            -------
            gate : (batch, action_dim) tensor ∈ (0, 1).
            """
            return self.gate(concepts)


    class SafeProjectionLayer(nn.Module):
        """
        Full safe projection layer: sigmoid bound + concept-conditioned gate.

        This is applied to the raw policy output to produce bounded,
        concept-modulated actions.

        For C2GFastEnv:
          - Actions 0-2 (throttle, pump, hvac) ∈ [0, 1] → sigmoid
          - Action 3 (BESS dispatch) ∈ [-1, 1] → 2·sigmoid − 1

        Parameters
        ----------
        concept_dim : int
        action_dim : int
        hidden : int
        init_bias : float
        """

        def __init__(self, concept_dim: int, action_dim: int = 4,
                     hidden: int = 32, init_bias: float = 2.0):
            super().__init__()
            self.action_dim = action_dim
            self.safety_gate = SafeProjectionGate(
                concept_dim=concept_dim,
                action_dim=action_dim,
                hidden=hidden,
                init_bias=init_bias,
            )

        def forward(
            self,
            raw_action: torch.Tensor,
            concepts: torch.Tensor,
        ) -> torch.Tensor:
            """
            Project raw policy output into safe, bounded action space.

            Parameters
            ----------
            raw_action : (batch, action_dim) — unbounded policy output
            concepts : (batch, K) — concept predictions

            Returns
            -------
            safe_action : (batch, action_dim) — bounded and gated
            """
            # Step 1: Architectural bound
            # Actions 0-2: sigmoid → [0, 1]
            # Action 3: 2·sigmoid − 1 → [-1, 1]
            bounded = torch.sigmoid(raw_action)
            if self.action_dim >= 4:
                # BESS action: rescale to [-1, 1]
                bounded = bounded.clone()
                bounded[:, 3] = 2.0 * bounded[:, 3] - 1.0

            # Step 2: Concept-conditioned gate
            gate = self.safety_gate(concepts)

            return bounded * gate


    class GateSupervisionLoss:
        """
        Auxiliary loss that trains the SafeProjectionGate to attenuate
        actions when safety concepts indicate danger.

        Gate target computation:
          g*_throttle = 1.0 − α · max(cooling_demand_A, cooling_demand_B)
          g*_pump     = 1.0  (always allow full pump — pumping is safe)
          g*_hvac     = 1.0  (always allow full HVAC — cooling is safe)
          g*_bess     = 1.0 − β · (1.0 − bess_headroom)

        When cooling demand is high, the gate learns to reduce throttle
        (batch compute), which is the primary heat source.

        Parameters
        ----------
        gate_alpha : float
            How much to attenuate throttle when cooling demand = 1.
            Default 0.5 (reduces throttle gate to 0.5 at max demand).
        gate_beta : float
            How much to attenuate BESS when headroom is low.
        loss_weight : float
            Weight of the gate loss relative to the concept loss.
        """

        def __init__(
            self,
            gate_alpha: float = 0.5,
            gate_beta: float = 0.3,
            loss_weight: float = 0.1,
        ):
            self.gate_alpha = gate_alpha
            self.gate_beta = gate_beta
            self.loss_weight = loss_weight

        def compute(
            self,
            gate_values: torch.Tensor,
            concept_targets: torch.Tensor,
        ) -> torch.Tensor:
            """
            Compute gate supervision loss.

            Parameters
            ----------
            gate_values : (batch, 4) — predicted gate activations
            concept_targets : (batch, 10) — ground-truth concept vectors

            Returns
            -------
            loss : scalar tensor
            """
            # Concept indices (from C2G_CONCEPT_NAMES):
            # 5 = cooling_demand_A, 6 = cooling_demand_B, 9 = bess_headroom
            cooling_demand = torch.max(
                concept_targets[:, 5],
                concept_targets[:, 6],
            )  # (batch,)
            bess_headroom = concept_targets[:, 9]  # (batch,)

            # Per-action gate targets
            batch_size = gate_values.shape[0]
            gate_target = torch.ones_like(gate_values)

            # Throttle (action 0): attenuate when cooling demand high
            gate_target[:, 0] = 1.0 - self.gate_alpha * cooling_demand

            # Pump (action 1): always pass through — more cooling is safe
            gate_target[:, 1] = 1.0

            # HVAC (action 2): always pass through
            gate_target[:, 2] = 1.0

            # BESS (action 3): attenuate when headroom is low
            gate_target[:, 3] = 1.0 - self.gate_beta * (1.0 - bess_headroom)

            loss = F.mse_loss(gate_values, gate_target) * self.loss_weight
            return loss


    class ConceptAndGateSupervisionCallback:
        """
        Joint callback for training both the concept encoder and the
        safety gate. Designed to be used with SB3's BaseCallback system.

        This combines:
          1. Concept alignment loss (MSE, decaying weight: 1.0 → 0.1)
          2. Gate supervision loss (MSE on gate targets)

        Both losses are optimised jointly with a single Adam optimiser.

        Usage (inside an SB3 training loop)
        ------------------------------------
          cb = ConceptAndGateSupervisionCallback(
              concept_encoder, safety_gate,
              total_timesteps=300_000,
          )
          # Call on each SB3 step:
          cb.update(model, num_timesteps)
        """

        def __init__(
            self,
            concept_encoder: "C2GConceptEncoder",
            safety_gate: SafeProjectionGate,
            total_timesteps: int,
            gate_alpha: float = 0.5,
            gate_beta: float = 0.3,
            gate_loss_weight: float = 0.1,
            concept_initial_weight: float = 1.0,
            concept_decay_to: float = 0.1,
            lr: float = 1e-3,
        ):
            self.concept_encoder = concept_encoder
            self.safety_gate = safety_gate
            self.total_timesteps = total_timesteps
            self.concept_initial_weight = concept_initial_weight
            self.concept_decay_to = concept_decay_to

            self.gate_loss_fn = GateSupervisionLoss(
                gate_alpha=gate_alpha,
                gate_beta=gate_beta,
                loss_weight=gate_loss_weight,
            )

            # Joint optimiser for encoder + gate
            self.optimizer = torch.optim.Adam(
                list(concept_encoder.parameters()) +
                list(safety_gate.parameters()),
                lr=lr,
            )

            self.concept_losses: list[float] = []
            self.gate_losses: list[float] = []

        def update(
            self,
            model,
            num_timesteps: int,
            obs_batch: torch.Tensor,
            concept_targets: torch.Tensor,
        ) -> dict[str, float]:
            """
            Run one supervision step.

            Parameters
            ----------
            model : SB3 model (for device)
            num_timesteps : current training step
            obs_batch : (batch, obs_dim) tensor
            concept_targets : (batch, 10) tensor

            Returns
            -------
            dict with 'concept_loss' and 'gate_loss' values
            """
            progress = min(1.0, num_timesteps / self.total_timesteps)
            concept_weight = (
                self.concept_initial_weight -
                (self.concept_initial_weight - self.concept_decay_to) * progress
            )

            # Forward: concept predictions
            pred_concepts = self.concept_encoder(obs_batch)
            n_targets = min(pred_concepts.shape[1], concept_targets.shape[1])

            # Concept alignment loss
            concept_loss = F.mse_loss(
                pred_concepts[:, :n_targets],
                concept_targets[:, :n_targets],
            ) * concept_weight

            # Gate supervision loss
            gate_values = self.safety_gate(pred_concepts.detach())
            gate_loss = self.gate_loss_fn.compute(gate_values, concept_targets)

            # Combined backward
            total_loss = concept_loss + gate_loss
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            self.concept_losses.append(concept_loss.item())
            self.gate_losses.append(gate_loss.item())

            return {
                "concept_loss": concept_loss.item(),
                "gate_loss": gate_loss.item(),
            }
