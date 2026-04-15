"""
baselines/safety/concept_bottleneck.py  —  Concept Bottleneck for C2G-Bench
=============================================================================
Implements a differentiable Concept Bottleneck Model (CBM) [Koh et al.,
ICML 2020] adapted to the C2G-Bench domain. The concept encoder maps
the 16/17-D raw observation to ~10 human-interpretable concepts that
capture the key physical states relevant to safety and control.

Concepts are chosen to be:
  1. Observable — computable from the raw obs vector
  2. Interpretable — meaningful to a DC operator
  3. Safety-relevant — directly tied to the 5 hard constraints
  4. Disentangled — each concept covers a distinct physical aspect

The concept encoder is jointly trained with the PPO policy via the
SB3 feature extractor API. An auxiliary supervision loss aligns
the learned concepts with hand-crafted ground-truth labels.

Architecture
------------
  obs(16/17) → MLP(hidden, hidden) → sigmoid → concepts(K)
  output to policy = [obs ; concepts]  (N + K dimensional)

This preserves full observability while providing interpretability.
Gradients flow back through the concept encoder, implementing
"differentiable attention regulation" from the SC26 HA-CompOpt paper.

Usage
-----
  from baselines.safety.concept_bottleneck import (
      C2GConceptEncoder, C2GConcepts, C2GConceptFeatureExtractor,
  )

  # Hand-crafted concept labels
  concepts = C2GConcepts.from_obs(obs)
  print(concepts.thermal_margin_A)  # → 0.83

  # Neural concept encoder
  encoder = C2GConceptEncoder(obs_dim=17, n_concepts=10)
  pred_concepts = encoder(obs_tensor)  # shape (batch, 10)

References
----------
  [Koh 2020]  P. Koh et al., "Concept Bottleneck Models", ICML 2020.
  [SC26 paper] HA-CompOpt: three-layer neuro-symbolic framework.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from numpy.typing import NDArray

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ─── C2G observation indices ──────────────────────────────────────
_I_TEMP_A   = 0   # T_A / T_safe
_I_TEMP_B   = 1   # T_B / T_safe
_I_SOC      = 2
_I_P_BASE   = 3
_I_P_FLEX   = 4
_I_P_FAC    = 5
_I_REGD     = 6   # ∈ [-1, 1]
_I_LMP      = 7
_I_GLOAD    = 8
_I_SPIKE    = 9
_I_PREV_THR = 10
_I_PREV_PMP = 11
_I_PUE      = 12
_I_T_AMB    = 13
_I_FREQ_DEV = 14
_I_VPCC     = 15
_I_BACKLOG  = 16  # may not exist in 16-D obs

_T_SAFE = 35.0
_T_WARN = 33.0
_SOC_MIN = 0.10
_SOC_MAX = 0.95


# ═══════════════════════════════════════════════════════════════════
# CONCEPT NAMES — The 10 human-interpretable concepts for C2G-Bench
# ═══════════════════════════════════════════════════════════════════

C2G_CONCEPT_NAMES: list[str] = [
    "thermal_margin_A",     # 0: How far Zone A is from T_safe (1=safe, 0=at limit)
    "thermal_margin_B",     # 1: How far Zone B is from T_safe
    "soc_health",           # 2: SOC distance from nearest bound (1=centre, 0=edge)
    "freq_stability",       # 3: Grid frequency stability (1=nominal, 0=at UFLS)
    "voltage_margin",       # 4: Voltage margin above UV relay (1=nominal, 0=trip)
    "cooling_demand_A",     # 5: Zone A cooling urgency (1=critical, 0=relaxed)
    "cooling_demand_B",     # 6: Zone B cooling urgency (1=critical, 0=relaxed)
    "grid_urgency",         # 7: How urgently BESS must respond to RegD signal
    "batch_pressure",       # 8: Backlog pressure on batch throughput
    "bess_headroom",        # 9: BESS can still charge or discharge meaningfully
]


@dataclass
class C2GConcepts:
    """
    Ground-truth concept values computed from a raw C2G observation.

    All concepts are normalised to [0, 1]:
      - Safety concepts (0-4): 1 = safe/healthy, 0 = at constraint boundary
      - Demand concepts (5-8): 1 = high urgency, 0 = relaxed
      - Headroom concept (9): 1 = plenty, 0 = exhausted
    """
    thermal_margin_A: float
    thermal_margin_B: float
    soc_health: float
    freq_stability: float
    voltage_margin: float
    cooling_demand_A: float
    cooling_demand_B: float
    grid_urgency: float
    batch_pressure: float
    bess_headroom: float

    @staticmethod
    def n_concepts() -> int:
        return 10

    @classmethod
    def from_obs(cls, obs: NDArray) -> "C2GConcepts":
        """
        Compute ground-truth concepts from a raw observation vector.

        Parameters
        ----------
        obs : ndarray, shape (16,) or (17,)
            Raw (normalised) observation from C2GFastEnv.
        """
        T_A = float(obs[_I_TEMP_A]) * _T_SAFE
        T_B = float(obs[_I_TEMP_B]) * _T_SAFE
        soc = float(obs[_I_SOC])
        freq_dev = abs(float(obs[_I_FREQ_DEV])) * 0.5  # Hz
        v_pcc = float(obs[_I_VPCC])
        regd = float(obs[_I_REGD])
        is_spike = float(obs[_I_SPIKE]) if len(obs) > _I_SPIKE else 0.0
        backlog = float(obs[_I_BACKLOG]) if len(obs) > _I_BACKLOG else 0.0

        # Thermal margins: 1 at 25°C, 0 at T_safe
        thermal_margin_A = np.clip((_T_SAFE - T_A) / (_T_SAFE - 20.0), 0.0, 1.0)
        thermal_margin_B = np.clip((_T_SAFE - T_B) / (_T_SAFE - 20.0), 0.0, 1.0)

        # SOC health: 1 at centre (0.5), 0 at edges (SOC_min or SOC_max)
        soc_dist = min(soc - _SOC_MIN, _SOC_MAX - soc)
        soc_health = np.clip(soc_dist / 0.4, 0.0, 1.0)  # 0.4 = half range

        # Frequency stability: 1 at nominal, 0 at UFLS (0.5 Hz)
        freq_stability = np.clip(1.0 - freq_dev / 0.5, 0.0, 1.0)

        # Voltage margin: 1 at 1.0 pu, 0 at 0.90 pu
        voltage_margin = np.clip((v_pcc - 0.90) / 0.10, 0.0, 1.0)

        # Cooling demand A: high when T_A > T_warn, or during GenAI spike
        cool_urgency_A = np.clip((T_A - (_T_WARN - 2.0)) / 4.0, 0.0, 1.0)
        cool_urgency_A = min(1.0, cool_urgency_A + 0.3 * is_spike)

        # Cooling demand B: high when T_B > T_warn
        cool_urgency_B = np.clip((T_B - (_T_WARN - 2.0)) / 4.0, 0.0, 1.0)

        # Grid urgency: how strongly the RegD signal demands response
        grid_urgency = float(np.clip(abs(regd), 0.0, 1.0))

        # Batch pressure: high backlog → high pressure
        batch_pressure = float(np.clip(backlog / 1.5, 0.0, 1.0))

        # BESS headroom: can still charge or discharge meaningfully
        # Low near SOC bounds, high in the middle
        bess_headroom = float(np.clip(soc_dist / 0.3, 0.0, 1.0))

        return cls(
            thermal_margin_A=float(thermal_margin_A),
            thermal_margin_B=float(thermal_margin_B),
            soc_health=float(soc_health),
            freq_stability=float(freq_stability),
            voltage_margin=float(voltage_margin),
            cooling_demand_A=float(cool_urgency_A),
            cooling_demand_B=float(cool_urgency_B),
            grid_urgency=float(grid_urgency),
            batch_pressure=float(batch_pressure),
            bess_headroom=float(bess_headroom),
        )

    def to_vector(self) -> NDArray:
        """Convert to (10,) float32 array."""
        return np.array([
            self.thermal_margin_A,
            self.thermal_margin_B,
            self.soc_health,
            self.freq_stability,
            self.voltage_margin,
            self.cooling_demand_A,
            self.cooling_demand_B,
            self.grid_urgency,
            self.batch_pressure,
            self.bess_headroom,
        ], dtype=np.float32)

    def to_dict(self) -> dict[str, float]:
        return {name: val for name, val in zip(C2G_CONCEPT_NAMES, self.to_vector())}


# ═══════════════════════════════════════════════════════════════════
# NEURAL CONCEPT ENCODER
# ═══════════════════════════════════════════════════════════════════

if _TORCH_AVAILABLE:

    class C2GConceptEncoder(nn.Module):
        """
        Differentiable concept bottleneck: obs(N) → concepts(K=10).

        Jointly trained with the PPO policy. Sigmoid output ensures
        all concept predictions are in (0, 1), matching ground-truth
        semantics.

        Parameters
        ----------
        obs_dim : int
            Observation dimension (16 or 17 for C2GFastEnv).
        n_concepts : int
            Number of concept outputs. Default 10.
        hidden : int
            Hidden layer width. Default 64.
        """

        def __init__(self, obs_dim: int, n_concepts: int = 10, hidden: int = 64):
            super().__init__()
            self.obs_dim = obs_dim
            self.n_concepts = n_concepts
            self.net = nn.Sequential(
                nn.Linear(obs_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, n_concepts),
            )

        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            return torch.sigmoid(self.net(obs))


    class C2GConceptFeatureExtractor(BaseFeaturesExtractor):
        """
        SB3-compatible feature extractor with concept bottleneck.

        output = [obs(N) ; concepts(K)] → (N + K)-dim feature vector

        This is the **Layer 1** of the HA-C2G architecture. The concept
        branch provides interpretability; the obs pass-through preserves
        full observability for the policy MLP.

        Parameters
        ----------
        observation_space : gym.Space
            From the environment.
        n_concepts : int
            Number of concept outputs.
        hidden : int
            Concept encoder hidden layer width.
        features_dim : int or None
            If None, computed as obs_dim + n_concepts.
        """

        def __init__(self, observation_space, n_concepts: int = 10,
                     hidden: int = 64, features_dim: int | None = None):
            obs_dim = observation_space.shape[0]
            if features_dim is None:
                features_dim = obs_dim + n_concepts
            super().__init__(observation_space, features_dim)

            self.concept_encoder = C2GConceptEncoder(
                obs_dim=obs_dim, n_concepts=n_concepts, hidden=hidden)
            self.n_concepts = n_concepts

        def forward(self, observations: torch.Tensor) -> torch.Tensor:
            concepts = self.concept_encoder(observations)
            return torch.cat([observations, concepts], dim=-1)


    class C2GGatedConceptFeatureExtractor(BaseFeaturesExtractor):
        """
        SB3-compatible feature extractor with concept bottleneck AND
        safe projection gate values.

        output = [obs(N) ; concepts(K) ; gate(A)] → (N + K + A)-dim

        This is used by the full HA-C2G training pipeline where the
        gate values are visible to the policy, enabling cooperation
        between the policy and the safe projection layer.

        Parameters
        ----------
        observation_space : gym.Space
        n_concepts : int
        action_dim : int
            Action space dimension (4 for C2GFastEnv).
        hidden : int
        """

        def __init__(self, observation_space, n_concepts: int = 10,
                     action_dim: int = 4, hidden: int = 64):
            obs_dim = observation_space.shape[0]
            features_dim = obs_dim + n_concepts + action_dim
            super().__init__(observation_space, features_dim)

            self.concept_encoder = C2GConceptEncoder(
                obs_dim=obs_dim, n_concepts=n_concepts, hidden=hidden)
            # Import here to avoid circular dependency
            from baselines.safety.safe_projection import SafeProjectionGate
            self.safety_gate = SafeProjectionGate(
                concept_dim=n_concepts, action_dim=action_dim)
            self.n_concepts = n_concepts
            self.action_dim = action_dim

        def forward(self, observations: torch.Tensor) -> torch.Tensor:
            concepts = self.concept_encoder(observations)
            gate_values = self.safety_gate(concepts)
            return torch.cat([observations, concepts, gate_values], dim=-1)
