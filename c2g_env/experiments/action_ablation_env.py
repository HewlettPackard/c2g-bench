"""
c2g_env.experiments.action_ablation_env
========================================
ActionAblationFastEnv — a subclass of C2GFastEnv that supports
action-level ablation studies.

During ``step()``, named actions can be marked *unavailable*; their
values are replaced by analyst-chosen fixed setpoints (or per-action
defaults) regardless of what the agent requests.  The info dict is
augmented with full auditing keys so downstream analysis can track
what was requested vs. applied.

Usage
-----
  from c2g_env.experiments import ActionAblationFastEnv

  env = ActionAblationFastEnv(
      scenario="default",
      unavailable_actions=("bess_dispatch", "hvac_effort"),
      fixed_action_values={"bess_dispatch": 0.0, "hvac_effort": 0.8},
  )
  obs, info = env.reset(seed=0)
  obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
  # info["requested_action"] — what the agent asked for
  # info["applied_action"]   — what was actually executed
  # info["action_unavailability"] — per-action audit trail
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from c2g_env.env_low_level import C2GFastEnv

# ---------------------------------------------------------------------------
# Action metadata (mirrors the structure that was previously inlined in
# C2GFastEnv but now lives exclusively here so the base env stays clean)
# ---------------------------------------------------------------------------

_ACTION_NAMES: tuple[str, ...] = (
    "throttle_batch",
    "pump_speed_A",
    "hvac_effort",
    "bess_dispatch",
)

_ACTION_INDEX: dict[str, int] = {name: idx for idx, name in enumerate(_ACTION_NAMES)}

_ACTION_BOUNDS: dict[str, tuple[float, float]] = {
    "throttle_batch": (0.0, 1.0),
    "pump_speed_A":   (0.0, 1.0),
    "hvac_effort":    (0.0, 1.0),
    "bess_dispatch":  (-1.0, 1.0),
}

# Default fixed values used when an action is disabled but no explicit
# fixed value is provided by the caller.
_ABLATION_DEFAULTS: dict[str, float] = {
    "throttle_batch": 1.0,
    "pump_speed_A":   1.0,
    "hvac_effort":    1.0,
    "bess_dispatch":  0.0,
}


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _normalise_action_names(action_names: Iterable[str] | None) -> tuple[str, ...]:
    """Validate and deduplicate an iterable of action names."""
    if action_names is None:
        return ()

    cleaned: list[str] = []
    invalid: list[str] = []
    for name in action_names:
        if name not in _ACTION_INDEX:
            invalid.append(name)
            continue
        if name not in cleaned:
            cleaned.append(name)

    if invalid:
        valid = ", ".join(_ACTION_NAMES)
        bad   = ", ".join(sorted(str(n) for n in invalid))
        raise ValueError(f"Unknown unavailable actions: {bad}. Valid actions: {valid}")

    return tuple(cleaned)


def _normalise_fixed_action_values(
    fixed_action_values: Mapping[str, float] | None,
) -> dict[str, float]:
    """Validate and clip fixed action values to their declared bounds."""
    if fixed_action_values is None:
        return {}

    cleaned: dict[str, float] = {}
    invalid: list[str] = []
    for name, value in fixed_action_values.items():
        if name not in _ACTION_BOUNDS:
            invalid.append(name)
            continue
        low, high = _ACTION_BOUNDS[name]
        cleaned[name] = float(np.clip(float(value), low, high))

    if invalid:
        valid = ", ".join(_ACTION_NAMES)
        bad   = ", ".join(sorted(str(n) for n in invalid))
        raise ValueError(f"Unknown fixed_action_values keys: {bad}. Valid actions: {valid}")

    return cleaned


# ---------------------------------------------------------------------------
# Subclass
# ---------------------------------------------------------------------------

class ActionAblationFastEnv(C2GFastEnv):
    """
    C2GFastEnv with action-level ablation support.

    All physics, reward, observation, and termination logic is inherited
    unchanged from C2GFastEnv.  This subclass only intercepts ``step()``
    to substitute disabled-action values before forwarding to the parent,
    and augments the returned info dict with an ablation audit trail.

    Parameters
    ----------
    scenario : str
        Passed through to C2GFastEnv.
    config_path : str or Path, optional
        Passed through to C2GFastEnv.
    unavailable_actions : iterable of str, optional
        Action names to lock during ``step()``.  The agent's requested
        value is recorded but replaced by the fixed/default value.
    fixed_action_values : mapping of str -> float, optional
        Fixed setpoints applied in a separate override pass. Values are
        clipped to the action bounds. When omitted for an unavailable
        action, the per-action ``ABLATION_DEFAULTS`` are used.
    **kwargs
        Forwarded to C2GFastEnv.
    """

    # Class-level metadata exposed for introspection
    ACTION_NAMES     = _ACTION_NAMES
    ACTION_INDEX     = _ACTION_INDEX
    ACTION_BOUNDS     = _ACTION_BOUNDS
    ABLATION_DEFAULTS = _ABLATION_DEFAULTS

    def __init__(
        self,
        scenario: str = "default",
        config_path: str | Path | None = None,
        unavailable_actions: Iterable[str] | None = None,
        fixed_action_values: Mapping[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(scenario=scenario, config_path=config_path, **kwargs)
        self._unavailable_actions = _normalise_action_names(unavailable_actions)
        self._fixed_action_values = _normalise_fixed_action_values(fixed_action_values)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def unavailable_actions(self) -> tuple[str, ...]:
        """Names of actions locked by the ablation configuration."""
        return self._unavailable_actions

    @property
    def fixed_action_values(self) -> dict[str, float]:
        """Copy of the fixed setpoints for disabled actions."""
        return dict(self._fixed_action_values)

    # ------------------------------------------------------------------
    # Overridden step
    # ------------------------------------------------------------------

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Apply ablation overrides then delegate to C2GFastEnv.step().

                Disabled actions are replaced by their default values before
                explicit fixed-action overrides are applied in a second pass.
                the physics simulation runs. The info dict is augmented with:
                    requested_action      — the raw action from the agent (clipped)
                    applied_action        — the action actually executed
                    unavailable_actions   — tuple of disabled action names
                    fixed_action_values   — copy of the fixed setpoint mapping
                    action_unavailability — per-action audit: requested / applied
        """
        overlap = set(self._unavailable_actions) & set(self._fixed_action_values.keys())
        if overlap:
            raise ValueError(
                "Overlapping action names in unavailable_actions and fixed_action_values: "
                + ", ".join(sorted(overlap))
            )

        requested_action: dict[str, float] = {
            name: float(value)
            for name, value in zip(
                self.ACTION_NAMES,
                np.clip(action.astype(np.float32), self._ACT_LOW, self._ACT_HIGH),
                strict=False,
            )
        }

        clipped = np.fromiter(requested_action.values(), dtype=np.float32)

        applied_arr = clipped.copy()

        for action_name in self._unavailable_actions:
            applied_value = float(self.ABLATION_DEFAULTS[action_name])
            applied_arr[self.ACTION_INDEX[action_name]] = applied_value

        for action_name, applied_value in self._fixed_action_values.items():
            applied_arr[self.ACTION_INDEX[action_name]] = applied_value

        applied_action: dict[str, float] = {
            name: float(applied_arr[idx]) for idx, name in enumerate(self.ACTION_NAMES)
        }

        obs, reward, terminated, truncated, info = super().step(applied_arr)

        return obs, reward, terminated, truncated, info
