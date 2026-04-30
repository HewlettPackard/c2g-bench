"""
baselines/safety/proof_tree.py  —  Hierarchical Proof Trees
==============================================================
Generates a structured audit artifact at every timestep, documenting
exactly which safety rules were checked, which passed or failed, and
the sensor readings that grounded each decision.

The proof tree serves two purposes:
  1. **Runtime safety** — a failed rule triggers the corresponding
     override (links to the physics rule shield)
  2. **Post-hoc audit** — operators and regulators can trace any
     decision back to the specific sensor readings that caused it

This is adapted from the SC26 HA-CompOpt paper's hierarchical proof
tree mechanism (Section 3.4), extended to cover all 5 C2G-Bench
constraints.

Tree Structure
--------------
  SYSTEM_SAFE
  ├── THERMAL_OK
  │   ├── C1: T_A < T_safe  [PASS/FAIL, T_A=28.3°C]
  │   └── C2: T_B < T_safe  [PASS/FAIL, T_B=27.1°C]
  ├── BESS_OK
  │   └── C3: SOC ∈ [min,max]  [PASS/FAIL, SOC=0.45]
  ├── GRID_OK
  │   ├── C4: |Δf| < 0.5 Hz  [PASS/FAIL, Δf=0.12 Hz]
  │   └── C5: V_pcc > 0.90   [PASS/FAIL, V=0.98 pu]
  └── CONCEPT_OK
      ├── thermal_margin_A: 0.83  (encoder vs ground-truth)
      ├── cooling_demand_A: 0.21
      └── ...

Usage
-----
  from baselines.safety.proof_tree import ProofTree

  tree = ProofTree.from_step(obs, action, safe_action, concepts, shield_info)
  print(tree.summary())
  tree.to_dict()  # JSON-serialisable
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from c2g_env.obs_indices import Fast as _F


class RuleStatus(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARN = "WARN"
    SKIP = "SKIP"


@dataclass
class ProofNode:
    """
    A node in the hierarchical proof tree.

    Attributes
    ----------
    name : str
        Rule or sub-goal identifier (e.g., "C1_THERMAL_A").
    status : RuleStatus
        PASS, FAIL, WARN, or SKIP.
    severity : float
        0.0 (informational) to 1.0 (critical).
    evidence : dict
        Sensor readings and computed values grounding this node.
    correction : str or None
        Description of action correction if status is FAIL.
    children : list[ProofNode]
        Sub-rules or sub-goals.
    """
    name: str
    status: RuleStatus
    severity: float = 0.0
    evidence: dict = field(default_factory=dict)
    correction: str | None = None
    children: list["ProofNode"] = field(default_factory=list)

    def is_safe(self) -> bool:
        return self.status in (RuleStatus.PASS, RuleStatus.SKIP)

    def n_failures(self) -> int:
        count = 0 if self.is_safe() else 1
        for child in self.children:
            count += child.n_failures()
        return count

    def depth(self) -> int:
        if not self.children:
            return 1
        return 1 + max(c.depth() for c in self.children)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status.value,
            "severity": self.severity,
            "evidence": self.evidence,
            "correction": self.correction,
            "children": [c.to_dict() for c in self.children],
        }

    def summary_lines(self, indent: int = 0) -> list[str]:
        prefix = "  " * indent
        icon = "✓" if self.is_safe() else "✗"
        line = f"{prefix}{icon} {self.name} [{self.status.value}]"
        if self.evidence:
            vals = ", ".join(f"{k}={v}" for k, v in self.evidence.items())
            line += f"  ({vals})"
        if self.correction:
            line += f"  → {self.correction}"
        lines = [line]
        for child in self.children:
            lines.extend(child.summary_lines(indent + 1))
        return lines


# ─── C2G observation indices ──────────────────────────────────
_T_SAFE     = 35.0
_T_WARN     = 33.0
_SOC_MIN    = 0.10
_SOC_MAX    = 0.95


class ProofTree:
    """
    Constructs and manages hierarchical proof trees for C2G-Bench.

    Usage
    -----
      tree = ProofTree.from_step(obs, action, safe_action, concepts, shield_info)
      print(tree.summary())
      d = tree.to_dict()
    """

    def __init__(self, root: ProofNode):
        self.root = root

    @classmethod
    def from_step(
        cls,
        obs: NDArray,
        raw_action: NDArray,
        safe_action: NDArray,
        concepts: dict[str, float] | None = None,
        shield_info: dict | None = None,
    ) -> "ProofTree":
        """
        Build a proof tree from a single environment step.

        Parameters
        ----------
        obs : ndarray
            Current (normalised) observation.
        raw_action : ndarray
            Action proposed by the RL agent.
        safe_action : ndarray
            Action after safety filtering.
        concepts : dict or None
            Concept name → value pairs from the concept encoder.
        shield_info : dict or None
            Info dict from the safety shield's filter() call.
        """
        # Decode physical values
        T_A = float(obs[_F.TEMP_A]) * _T_SAFE
        T_B = float(obs[_F.TEMP_B]) * _T_SAFE
        soc = float(obs[_F.SOC])
        freq_dev = float(obs[_F.FREQ_DEV]) * 0.5
        v_pcc = float(obs[_F.VPCC])

        was_modified = not np.allclose(raw_action, safe_action, atol=1e-4)

        # ── C1: Thermal A ────────────────────────────────────────
        c1_pass = T_A < _T_SAFE
        c1_warn = T_A > _T_WARN
        c1_status = (RuleStatus.FAIL if not c1_pass else
                     RuleStatus.WARN if c1_warn else RuleStatus.PASS)
        c1 = ProofNode(
            name="C1_THERMAL_A",
            status=c1_status,
            severity=1.0 if not c1_pass else (0.5 if c1_warn else 0.0),
            evidence={"T_A": round(T_A, 2), "T_safe": _T_SAFE, "T_warn": _T_WARN},
            correction=f"throttle: {raw_action[0]:.2f}→{safe_action[0]:.2f}, "
                       f"pump: {raw_action[1]:.2f}→{safe_action[1]:.2f}"
                       if was_modified and (not c1_pass or c1_warn) else None,
        )

        # ── C2: Thermal B ────────────────────────────────────────
        c2_pass = T_B < _T_SAFE
        c2_warn = T_B > _T_WARN
        c2_status = (RuleStatus.FAIL if not c2_pass else
                     RuleStatus.WARN if c2_warn else RuleStatus.PASS)
        c2 = ProofNode(
            name="C2_THERMAL_B",
            status=c2_status,
            severity=1.0 if not c2_pass else (0.5 if c2_warn else 0.0),
            evidence={"T_B": round(T_B, 2), "T_safe": _T_SAFE, "T_warn": _T_WARN},
            correction=f"hvac: {raw_action[2]:.2f}→{safe_action[2]:.2f}"
                       if was_modified and (not c2_pass or c2_warn) else None,
        )

        # ── C3: SOC ──────────────────────────────────────────────
        c3_pass = _SOC_MIN <= soc <= _SOC_MAX
        c3 = ProofNode(
            name="C3_SOC_BOUNDS",
            status=RuleStatus.PASS if c3_pass else RuleStatus.FAIL,
            severity=0.0 if c3_pass else 0.8,
            evidence={"SOC": round(soc, 4), "SOC_min": _SOC_MIN, "SOC_max": _SOC_MAX},
            correction=f"bess: {raw_action[3]:.2f}→{safe_action[3]:.2f}"
                       if was_modified and not c3_pass else None,
        )

        # ── C4: Frequency ────────────────────────────────────────
        c4_pass = abs(freq_dev) < 0.5
        c4 = ProofNode(
            name="C4_FREQUENCY",
            status=RuleStatus.PASS if c4_pass else RuleStatus.FAIL,
            severity=0.0 if c4_pass else 1.0,
            evidence={"freq_dev_Hz": round(freq_dev, 3), "limit_Hz": 0.5},
        )

        # ── C5: Voltage ──────────────────────────────────────────
        c5_pass = v_pcc > 0.90
        c5 = ProofNode(
            name="C5_VOLTAGE",
            status=RuleStatus.PASS if c5_pass else RuleStatus.FAIL,
            severity=0.0 if c5_pass else 1.0,
            evidence={"V_pcc_pu": round(v_pcc, 4), "limit_pu": 0.90},
        )

        # ── Sub-goals ────────────────────────────────────────────
        thermal_ok = ProofNode(
            name="THERMAL_OK",
            status=(RuleStatus.PASS if c1.is_safe() and c2.is_safe()
                    else RuleStatus.FAIL),
            children=[c1, c2],
        )
        bess_ok = ProofNode(
            name="BESS_OK",
            status=c3.status,
            children=[c3],
        )
        grid_ok = ProofNode(
            name="GRID_OK",
            status=(RuleStatus.PASS if c4.is_safe() and c5.is_safe()
                    else RuleStatus.FAIL),
            children=[c4, c5],
        )

        # ── Concept sub-tree (if concepts provided) ──────────────
        concept_children: list[ProofNode] = []
        if concepts:
            for name, val in concepts.items():
                concept_children.append(ProofNode(
                    name=f"concept_{name}",
                    status=RuleStatus.PASS,
                    evidence={"value": round(float(val), 4)},
                ))
        concept_ok = ProofNode(
            name="CONCEPT_OK",
            status=RuleStatus.PASS,
            children=concept_children,
        )

        # ── Root ─────────────────────────────────────────────────
        all_safe = (thermal_ok.is_safe() and bess_ok.is_safe() and
                    grid_ok.is_safe())
        root = ProofNode(
            name="SYSTEM_SAFE",
            status=RuleStatus.PASS if all_safe else RuleStatus.FAIL,
            evidence={
                "action_modified": was_modified,
                "shield_type": shield_info.get("shield_type", "unknown")
                               if shield_info else "none",
            },
            children=[thermal_ok, bess_ok, grid_ok, concept_ok],
        )

        return cls(root=root)

    def summary(self) -> str:
        """Human-readable summary of the proof tree."""
        return "\n".join(self.root.summary_lines())

    def to_dict(self) -> dict:
        """JSON-serialisable dict representation."""
        return self.root.to_dict()

    @property
    def is_safe(self) -> bool:
        return self.root.is_safe()

    @property
    def n_failures(self) -> int:
        return self.root.n_failures()

    @property
    def depth(self) -> int:
        return self.root.depth()
