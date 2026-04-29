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

    Step 2 — Concept-conditioned gate + action prior:
        g = σ(MLP_φ(concepts)) ∈ (0, 1)^{N_a}
        a_prior = f(concepts, obs)
        a_final = g ⊙ a_policy + (1 − g) ⊙ a_prior
        The gate learns how much of the policy action to keep versus how much
        to blend toward a concept-guided safe prior.

Key design: the gate is **actively trained** (not just near-pass-through):
    - An auxiliary gate supervision loss provides an explicit training signal
    - Gate targets encode environment-dependent pass-through values rather than
        all-ones / pure pass-through behavior
    - Action priors encode meaningful cooling / BESS responses from concepts
    - The layer is DIFFERENTIABLE: gradients flow through both the gate and the
        policy action path during PPO training, so the policy learns to cooperate.

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

    def build_gate_targets(
        concepts: torch.Tensor,
        gate_alpha: float = 0.5,
        gate_beta: float = 0.3,
    ) -> torch.Tensor:
        """Build concept-guided pass-through targets for Layer 2."""
        cooling_A = concepts[:, 5]
        cooling_B = concepts[:, 6]
        cooling_demand = torch.maximum(cooling_A, cooling_B)
        grid_urgency = concepts[:, 7]

        gate_target = torch.ones(concepts.shape[0], 4, device=concepts.device, dtype=concepts.dtype)
        gate_target[:, 0] = 1.0 - gate_alpha * cooling_demand
        gate_target[:, 1] = 1.0 - 0.75 * gate_alpha * cooling_A
        gate_target[:, 2] = 1.0 - 0.75 * gate_alpha * cooling_B
        gate_target[:, 3] = 1.0 - gate_beta * grid_urgency
        return torch.clamp(gate_target, 0.0, 1.0)


    def build_action_priors(
        concepts: torch.Tensor,
        obs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build concept-guided safe action priors for Layer 2 blending."""
        cooling_A = concepts[:, 5]
        cooling_B = concepts[:, 6]
        cooling_demand = torch.maximum(cooling_A, cooling_B)
        batch_pressure = concepts[:, 8]
        grid_urgency = concepts[:, 7]
        bess_headroom = concepts[:, 9]

        priors = torch.zeros(concepts.shape[0], 4, device=concepts.device, dtype=concepts.dtype)
        priors[:, 0] = torch.clamp(cooling_demand - 0.5 * batch_pressure + 0.25, 0.0, 1.0)
        priors[:, 1] = torch.clamp(cooling_A, 0.0, 1.0)
        priors[:, 2] = torch.clamp(cooling_B, 0.0, 1.0)

        regd = torch.zeros(concepts.shape[0], device=concepts.device, dtype=concepts.dtype)
        if obs is not None and obs.shape[1] > 6:
            regd = torch.clamp(obs[:, 6], -1.0, 1.0)
        priors[:, 3] = torch.clamp(torch.sign(regd) * grid_urgency * bess_headroom, -1.0, 1.0)
        return priors


    def blend_with_action_priors(
        policy_action: torch.Tensor,
        gate_values: torch.Tensor,
        action_priors: torch.Tensor,
    ) -> torch.Tensor:
        """Blend policy actions toward concept-guided priors."""
        return gate_values * policy_action + (1.0 - gate_values) * action_priors


    def compute_layer2_action(
        policy_action: torch.Tensor,
        concepts: torch.Tensor,
        obs: torch.Tensor | None = None,
        safety_gate: "SafeProjectionGate | None" = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply Layer 2 using learned pass-through gates and action priors."""
        gate_values = safety_gate(concepts) if safety_gate is not None else torch.ones_like(policy_action)
        action_priors = build_action_priors(concepts, obs=obs)
        safe_action = blend_with_action_priors(policy_action, gate_values, action_priors)
        return safe_action, gate_values, action_priors

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
        Full safe projection layer: sigmoid bound + concept-guided prior blend.

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
            observations: torch.Tensor | None = None,
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
            safe_action, _, _ = compute_layer2_action(
                bounded,
                concepts,
                obs=observations,
                safety_gate=self.safety_gate,
            )
            return safe_action


    class GateSupervisionLoss:
        """
        Auxiliary loss that trains the SafeProjectionGate to encode
        meaningful concept-conditioned pass-through behavior.

        Gate target computation:
          g*_throttle = 1.0 − α · max(cooling_demand_A, cooling_demand_B)
          g*_pump     = 1.0 − 0.75·α·cooling_demand_A
          g*_hvac     = 1.0 − 0.75·α·cooling_demand_B
          g*_bess     = 1.0 − β · grid_urgency

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
            gate_target = build_gate_targets(
                concept_targets,
                gate_alpha=self.gate_alpha,
                gate_beta=self.gate_beta,
            )

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
