"""
baselines.safety  —  High-Assurance Safety Modules for C2G-Bench
=================================================================
This package provides a taxonomy of safety controllers organised into
three tiers:

**Tier 1 — Hard-Guarantee Methods (provable safety)**
  - ``cbf_shield``       : Control Barrier Function safety filter (QP)
  - ``hj_shield``        : Hamilton-Jacobi reachability safety filter
  - ``mpc_safety_filter``: Model-Predictive Safety Filter (NLP)
  - ``safety_shield``    : Simplex-style analytic safety shield

**Tier 2 — Soft-Guarantee Methods (statistical safety)**
  - ``train_ppo_lagrangian`` : Adaptive Lagrange multipliers
  - ``train_cpo``            : Constrained Policy Optimization
  - ``train_shield_reward_shaping`` : Fixed penalty reward shaping
  - ``train_shielded_ppo``  : PPO with Simplex safety shield

**Tier 3 — Neuro-Symbolic / Interpretable High-Assurance**
  - ``concept_bottleneck``: Differentiable concept encoder for C2G
  - ``safe_projection``  : Concept-conditioned gate (Layer 2, *trained*)
  - ``proof_tree``       : Hierarchical audit proof trees
  - ``train_ha_c2g``     : Full HA-C2G (3-layer neuro-symbolic)
  - ``train_cbm_only``   : CBM-only ablation
  - ``train_cbm_gate``   : CBM+Gate ablation
  - ``train_cbm_shield`` : CBM+Shield ablation

All Tier 1 filters expose the same ``filter(action, obs)`` API as
``SafetyShield`` in ``baselines.safety.safety_shield``, enabling
drop-in replacement and fair comparison.
"""

from baselines.safety.cbf_shield import CBFShield, CBFShieldedEnv
from baselines.safety.hj_shield import HJShield, HJShieldedEnv
from baselines.safety.mpc_safety_filter import MPCSafetyFilter, MPCSFShieldedEnv
from baselines.safety.safety_shield import SafetyShield, ShieldedEnv, ShieldedAgent, ShieldStats
from baselines.safety.concept_bottleneck import C2GConcepts, C2G_CONCEPT_NAMES
from baselines.safety.proof_tree import ProofTree, ProofNode

__all__ = [
    "CBFShield", "CBFShieldedEnv",
    "HJShield", "HJShieldedEnv",
    "MPCSafetyFilter", "MPCSFShieldedEnv",
    "SafetyShield", "ShieldedEnv", "ShieldedAgent", "ShieldStats",
    "C2GConcepts", "C2G_CONCEPT_NAMES",
    "ProofTree", "ProofNode",
]

# Torch-dependent classes (optional)
try:
    from baselines.safety.concept_bottleneck import C2GConceptEncoder
    from baselines.safety.safe_projection import SafeProjectionLayer, SafeProjectionGate
    __all__ += [
        "C2GConceptEncoder",
        "SafeProjectionLayer", "SafeProjectionGate",
    ]
except ImportError:
    pass
