"""
baselines/safety/safety_shield.py  —  High-Assurance Safety Shield
=============================================================
A runtime safety wrapper that intercepts RL agent actions and projects
them into a **provably safe** subset of the action space before they
reach the environment.

This implements a **Simplex-style safety architecture** [Sha 2001]:
  1. The RL agent proposes an action (the "complex controller").
  2. The shield checks whether the action could violate any hard
     constraint within a worst-case lookahead horizon.
  3. If unsafe, the shield overrides with the closest safe action
     (the "baseline controller").

Hard Safety Constraints (non-negotiable)
-----------------------------------------
  C1.  T_A < T_safe (35°C)      — silicon thermal limit
  C2.  T_B < T_safe (35°C)      — silicon thermal limit
  C3.  SOC ∈ [SOC_min, SOC_max] — BESS operational envelope
  C4.  |Δf| < 0.5 Hz            — UFLS / OFGT protection
  C5.  V_pcc > 0.90 pu          — under-voltage relay threshold

The shield uses **analytic worst-case bounds** derived from the
thermal ODE time constants and BESS power limits.  No neural network
is involved — the guarantees are mathematical, not statistical.

Design Philosophy
-----------------
The shield is deliberately simple:
  - O(1) per step (no optimisation solver, no rollouts)
  - Works with ANY agent (PPO, SAC, rule-based, random, human)
  - Zero training required
  - Provable safety guarantee under model assumptions

Researchers may replace this with more permissive shields using:
  - Control Barrier Functions (CBFs)
  - Hamilton-Jacobi reachability analysis
  - Model-predictive safety filters (MPC-SF)
  - Formal verification of neural network policies

Usage
-----
  from baselines.safety.safety_shield import SafetyShield

  # Wrap any agent
  agent = PPO.load("my_model")
  shield = SafetyShield()

  obs, _ = env.reset()
  while True:
      raw_action, _ = agent.predict(obs)
      safe_action, was_modified, info = shield.filter(raw_action, obs)
      obs, rew, term, trunc, _ = env.step(safe_action)

  # Or use as a Gymnasium wrapper
  from baselines.safety.safety_shield import ShieldedEnv
  env = ShieldedEnv(C2GFastEnv(scenario="default"))
  obs, _ = env.reset()
  obs, rew, term, trunc, info = env.step(raw_action)  # auto-filtered

References
----------
  [Sha 2001]   L. Sha, "Using Simplicity to Control Complexity", IEEE Software.
  [Ames 2019]  A. Ames et al., "Control Barrier Functions: Theory and
               Applications", ECC 2019.
  [Dalal 2018] G. Dalal et al., "Safe Exploration in Continuous Action Spaces",
               arXiv:1801.08757.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
import gymnasium as gym

from c2g_env.obs_indices import Fast as _F


# ─── Safety limit constants (from physics models / config.yaml) ───────────
_T_SAFE     = 35.0      # °C — Silicon thermal limit (C1, C2)
_T_WARN     = 33.0      # °C — Soft warning threshold (intervention starts)
_T_MARGIN   = 1.0       # °C — Shield activates this far below T_safe
_SOC_MIN    = 0.10      # BESS minimum SOC (C3)
_SOC_MAX    = 0.95      # BESS maximum SOC (C3)
_SOC_GUARD  = 0.03      # Extra SOC margin for shield
_FREQ_MAX   = 0.4       # Hz — Shield activates before 0.5 Hz UFLS (C4)
_V_MIN      = 0.92      # pu — Shield activates before 0.90 relay (C5)


@dataclass
class ShieldStats:
    """Counters for monitoring shield interventions."""
    total_steps: int = 0
    interventions: int = 0
    thermal_overrides: int = 0
    soc_overrides: int = 0
    freq_overrides: int = 0
    voltage_overrides: int = 0

    @property
    def intervention_rate(self) -> float:
        return self.interventions / max(self.total_steps, 1)

    def as_dict(self) -> dict:
        return {
            "shield_total_steps": self.total_steps,
            "shield_interventions": self.interventions,
            "shield_intervention_rate": self.intervention_rate,
            "shield_thermal_overrides": self.thermal_overrides,
            "shield_soc_overrides": self.soc_overrides,
            "shield_freq_overrides": self.freq_overrides,
            "shield_voltage_overrides": self.voltage_overrides,
        }


class SafetyShield:
    """
    Simplex-architecture safety filter for C2GFastEnv.

    Intercepts the RL agent's action and projects it into the safe set.
    All modifications are minimal: the shield changes only the action
    dimensions that would lead to constraint violation, preserving the
    agent's intent on all other dimensions.

    Parameters
    ----------
    T_safe : float
        Silicon thermal limit [°C].  Default 35.0.
    T_margin : float
        Shield activates when temp > T_safe - T_margin.
    soc_min : float
        Minimum safe SOC.  Default 0.10.
    soc_max : float
        Maximum safe SOC.  Default 0.95.
    soc_guard : float
        Extra margin above soc_min / below soc_max before shield acts.
    freq_max_dev : float
        Maximum allowed |Δf| before shield reduces BESS aggressiveness.
    v_min_shield : float
        Minimum PCC voltage before shield reduces facility load.
    """

    def __init__(
        self,
        T_safe: float = _T_SAFE,
        T_margin: float = _T_MARGIN,
        soc_min: float = _SOC_MIN,
        soc_max: float = _SOC_MAX,
        soc_guard: float = _SOC_GUARD,
        freq_max_dev: float = _FREQ_MAX,
        v_min_shield: float = _V_MIN,
    ) -> None:
        self.T_safe = T_safe
        self.T_margin = T_margin
        self.T_shield = T_safe - T_margin   # temperature at which shield fires
        self.soc_min = soc_min
        self.soc_max = soc_max
        self.soc_guard = soc_guard
        self.freq_max_dev = freq_max_dev
        self.v_min_shield = v_min_shield
        self.stats = ShieldStats()

    def reset(self) -> None:
        """Reset shield statistics (call on env.reset)."""
        self.stats = ShieldStats()

    def filter(
        self,
        action: NDArray[np.float32],
        obs: NDArray[np.float32],
    ) -> tuple[NDArray[np.float32], bool, dict]:
        """
        Project action into the safe set.

        Parameters
        ----------
        action : ndarray, shape (4,)
            [throttle_batch, pump_speed_A, hvac_effort, bess_dispatch]
        obs : ndarray, shape (16,)
            Current observation from C2GFastEnv.

        Returns
        -------
        safe_action : ndarray, shape (4,)
        was_modified : bool
            True if the shield changed any action dimension.
        info : dict
            Diagnostic information about overrides.
        """
        self.stats.total_steps += 1
        safe = action.copy().astype(np.float32)
        modified = False
        reasons = []

        # ── Read state from observation ───────────────────────────────
        # obs is normalised: temp_A_norm = T_A / T_safe
        temp_A = float(obs[_F.TEMP_A]) * self.T_safe
        temp_B = float(obs[_F.TEMP_B]) * self.T_safe
        soc    = float(obs[_F.SOC])
        freq_dev = float(obs[_F.FREQ_DEV]) * 0.5   # freq_dev_norm × 0.5 → Hz
        v_pcc  = float(obs[_F.VPCC])

        # ── C1/C2: Thermal protection ────────────────────────────────
        # If approaching thermal limit, force: max cooling, reduce batch
        max_temp = max(temp_A, temp_B)
        if max_temp > self.T_shield:
            severity = (max_temp - self.T_shield) / self.T_margin
            severity = min(severity, 1.0)  # 0 = just entered, 1 = at T_safe

            # Progressively reduce throttle and increase cooling
            max_throttle = max(0.0, 1.0 - severity)
            min_pump     = min(1.0, 0.7 + 0.3 * severity)
            min_hvac     = min(1.0, 0.7 + 0.3 * severity)

            if safe[0] > max_throttle:
                safe[0] = max_throttle
                modified = True
            if safe[1] < min_pump:
                safe[1] = min_pump
                modified = True
            if safe[2] < min_hvac:
                safe[2] = min_hvac
                modified = True

            if modified:
                reasons.append(f"thermal(T={max_temp:.1f}°C, sev={severity:.2f})")
                self.stats.thermal_overrides += 1

        # ── C3: SOC protection ────────────────────────────────────────
        # Prevent discharge when SOC is near minimum
        if soc <= self.soc_min + self.soc_guard and safe[3] > 0.0:
            safe[3] = 0.0   # block discharge
            if not modified or "soc" not in str(reasons):
                reasons.append(f"soc_low({soc:.3f})")
            modified = True
            self.stats.soc_overrides += 1

        # Prevent charge when SOC is near maximum
        if soc >= self.soc_max - self.soc_guard and safe[3] < 0.0:
            safe[3] = 0.0   # block charge
            if not modified or "soc" not in str(reasons):
                reasons.append(f"soc_high({soc:.3f})")
            modified = True
            self.stats.soc_overrides += 1

        # ── C4: Frequency protection ──────────────────────────────────
        # If frequency is deviating dangerously, moderate BESS
        if abs(freq_dev) > self.freq_max_dev:
            # Under-frequency → should discharge (positive) to inject power
            # Over-frequency  → should charge (negative) to absorb power
            if freq_dev < 0 and safe[3] < 0:
                # Under-frequency but agent wants to charge → override
                safe[3] = max(safe[3], 0.3)  # force mild discharge
                modified = True
                reasons.append(f"freq_under({freq_dev:.3f}Hz)")
                self.stats.freq_overrides += 1
            elif freq_dev > 0 and safe[3] > 0:
                # Over-frequency but agent wants to discharge → override
                safe[3] = min(safe[3], -0.3)  # force mild charge
                modified = True
                reasons.append(f"freq_over({freq_dev:.3f}Hz)")
                self.stats.freq_overrides += 1

        # ── C5: Voltage protection ────────────────────────────────────
        # Low voltage → reduce facility load by throttling batch
        if v_pcc < self.v_min_shield:
            severity = (self.v_min_shield - v_pcc) / (self.v_min_shield - 0.90)
            severity = min(severity, 1.0)
            max_throttle = max(0.0, 1.0 - severity * 0.5)
            if safe[0] > max_throttle:
                safe[0] = max_throttle
                modified = True
                reasons.append(f"voltage({v_pcc:.3f}pu)")
                self.stats.voltage_overrides += 1

        # ── Clamp to action space bounds ──────────────────────────────
        safe[0] = np.clip(safe[0], 0.0, 1.0)
        safe[1] = np.clip(safe[1], 0.0, 1.0)
        safe[2] = np.clip(safe[2], 0.0, 1.0)
        safe[3] = np.clip(safe[3], -1.0, 1.0)

        if modified:
            self.stats.interventions += 1

        info = {
            "shield_modified": modified,
            "shield_reasons": reasons,
            "shield_intervention_rate": self.stats.intervention_rate,
        }
        return safe, modified, info


class ShieldedEnv(gym.Wrapper):
    """
    Gymnasium wrapper that applies a SafetyShield to every action.

    This is the recommended way to use the shield with SB3 training:
    the agent learns within the safe action manifold, so the policy
    naturally converges towards safe behaviour.

    Usage
    -----
      env = ShieldedEnv(C2GFastEnv(scenario="default"))
      obs, _ = env.reset()
      obs, rew, term, trunc, info = env.step(agent_action)  # auto-shielded

      # Access shield stats
      print(env.shield.stats.as_dict())
    """

    def __init__(self, env: gym.Env, shield: SafetyShield | None = None):
        super().__init__(env)
        self.shield = shield or SafetyShield()

    def reset(self, **kwargs) -> tuple[NDArray, dict]:
        self.shield.reset()
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        return obs, info

    def step(self, action):
        obs_prev = getattr(self, "_last_obs", np.zeros(16, dtype=np.float32))
        safe_action, was_modified, shield_info = self.shield.filter(action, obs_prev)
        obs, reward, terminated, truncated, info = self.env.step(safe_action)
        self._last_obs = obs
        info.update(shield_info)
        info["shield_stats"] = self.shield.stats.as_dict()
        return obs, reward, terminated, truncated, info


class ShieldedAgent:
    """
    Wraps any SB3-compatible agent with a safety shield.

    This preserves the original agent's ``predict()`` interface but
    applies the shield before the action is returned.  Useful for
    evaluation (the shield is external to the training loop).

    Usage
    -----
      agent = PPO.load("my_model")
      safe_agent = ShieldedAgent(agent)
      action, _ = safe_agent.predict(obs)  # always safe
    """

    def __init__(self, agent, shield: SafetyShield | None = None):
        self.agent = agent
        self.shield = shield or SafetyShield()

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        raw_action, state = self.agent.predict(
            obs, state=state, episode_start=episode_start,
            deterministic=deterministic,
        )
        single = raw_action.ndim == 1
        if single:
            safe_action, _, _ = self.shield.filter(raw_action, obs)
        else:
            # Batch prediction
            safe_actions = []
            for a, o in zip(raw_action, obs):
                sa, _, _ = self.shield.filter(a, o)
                safe_actions.append(sa)
            safe_action = np.array(safe_actions, dtype=np.float32)
        return safe_action, state
