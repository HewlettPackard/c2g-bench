"""
baselines.safety  —  High-Assurance Safety Modules for C2G-Bench
=================================================================
This package provides a taxonomy of safety controllers organised into
three tiers:

**Tier 1 — Hard-Guarantee Methods (provable safety)**
  - ``cbf_shield``       : Control Barrier Function safety filter (QP)
  - ``hj_shield``        : Hamilton-Jacobi reachability safety filter
  - ``mpc_safety_filter``: Model-Predictive Safety Filter (NLP)

**Tier 2 — Soft-Guarantee Methods (statistical safety)**
  - PPO-Lagrangian       : (in ``baselines/train_ppo_lagrangian.py``)
  - CPO                  : (in ``baselines/train_cpo.py``)
  - Shield reward shaping: (in ``baselines/train_shield_reward_shaping.py``)

**Tier 3 — Neuro-Symbolic / Interpretable High-Assurance**
  - ``concept_bottleneck``: Differentiable concept encoder for C2G
  - ``safe_projection``  : Concept-conditioned gate (Layer 2, *trained*)
  - ``proof_tree``       : Hierarchical audit proof trees

All Tier 1 filters expose the same ``filter(action, obs)`` API as the
existing ``SafetyShield`` in ``baselines/safety_shield.py``, enabling
drop-in replacement and fair comparison.
"""

from baselines.safety.cbf_shield import CBFShield, CBFShieldedEnv
from baselines.safety.hj_shield import HJShield, HJShieldedEnv
from baselines.safety.mpc_safety_filter import MPCSafetyFilter, MPCSFShieldedEnv
from baselines.safety.concept_bottleneck import C2GConcepts, C2G_CONCEPT_NAMES
from baselines.safety.proof_tree import ProofTree, ProofNode

__all__ = [
    "CBFShield", "CBFShieldedEnv",
    "HJShield", "HJShieldedEnv",
    "MPCSafetyFilter", "MPCSFShieldedEnv",
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
