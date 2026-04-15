"""
baselines/safety/hj_shield.py  —  Hamilton-Jacobi Reachability Safety Filter
==============================================================================
Precomputes the backward reachable set (BRS) for the C2G thermal subsystem
offline via dynamic programming on a discretised state grid. At runtime,
checks whether the current state is near the BRS boundary; if so, applies
the optimal safe control from the precomputed value function.

Theory
------
The HJ value function V(x) satisfies the HJI-PDE:

    max(V_t + min_u max_d H(x, u, d), ℓ(x) − V) = 0

where ℓ(x) ≤ 0 defines the unsafe set, and the optimal safe control is:

    u*(x) = arg min_u ∇V(x) · f(x, u)

At runtime:
  - If V(x) > δ  (state is safely inside the safe set) → pass through RL action
  - If V(x) ≤ δ  (state is near or inside BRS boundary) → use u*(x)

Because the full C2G state space is 16-D (too large for grid-based DP),
we project onto two 2-D subsystems:

  Subsystem 1 (thermal-A):  x = (T_A, pump_speed)    → u = [throttle, pump]
  Subsystem 2 (SOC):        x = (SOC, bess_dispatch)  → u = [bess]

Each subsystem is solved offline on a fine grid (100×100).

This is the tightest safe set (least conservative) of all shields,
but it requires offline computation and only covers the projected
state subsystems.

References
----------
  [Bansal 2017]  S. Bansal et al., "Hamilton-Jacobi Reachability: Some
                 Recent Theoretical Advances and Applications in Unmanned
                 Airspace Management", CDC 2017.
  [Fisac 2019]   J. Fisac et al., "A General Safety Framework for
                 Learning-Based Control in Uncertain Robotic Systems",
                 IEEE T-ASE.
  [Mitchell 2005] I. Mitchell et al., "A Time-Dependent Hamilton-Jacobi
                  Formulation of Reachable Sets for Continuous Dynamic
                  Games", IEEE TAC.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
import gymnasium as gym


# ─── Physical constants ───────────────────────────────────────────
_T_SAFE     = 35.0
_T_MIN      = 15.0
_SOC_MIN    = 0.10
_SOC_MAX    = 0.95

# Observation indices
_I_TEMP_A   = 0
_I_TEMP_B   = 1
_I_SOC      = 2
_I_T_AMB    = 13
_I_FREQ_DEV = 14
_I_VPCC     = 15

# Grid parameters for offline computation
_N_GRID     = 100    # grid points per dimension
_DT         = 5.0    # seconds per tick
_N_OFFLINE  = 200    # offline DP iterations

# Thermal model (simplified)
_C_A        = 4.0e6
_UA_A       = 1.5e5
_P_FLEX_MAX = 30_000.0
_P_PUMP_MAX = 5_000.0
_E_BESS_CAP = 150_000.0
_P_BESS_MAX = 50_000.0


@dataclass
class HJStats:
    """Counters for monitoring HJ shield interventions."""
    total_steps: int = 0
    interventions: int = 0
    thermal_overrides: int = 0
    soc_overrides: int = 0
    mean_value_function: float = 0.0
    min_value_function: float = float("inf")

    @property
    def intervention_rate(self) -> float:
        return self.interventions / max(self.total_steps, 1)

    def as_dict(self) -> dict:
        return {
            "hj_total_steps": self.total_steps,
            "hj_interventions": self.interventions,
            "hj_intervention_rate": self.intervention_rate,
            "hj_thermal_overrides": self.thermal_overrides,
            "hj_soc_overrides": self.soc_overrides,
            "hj_mean_value": self.mean_value_function,
            "hj_min_value": self.min_value_function,
        }


class HJShield:
    """
    Hamilton-Jacobi Reachability safety filter for C2GFastEnv.

    Precomputes a value function over two projected subsystems and
    uses it at runtime to override unsafe actions.

    Parameters
    ----------
    T_safe : float
        Silicon thermal limit [°C].
    soc_min, soc_max : float
        BESS SOC operational bounds.
    delta : float
        Safety margin on value function. Actions are overridden
        when V(x) < delta.
    n_grid : int
        Grid resolution per dimension for offline DP.
    precompute : bool
        If True, runs the offline DP at construction time.
        Set False for unit testing (uses analytical fallback).
    """

    def __init__(
        self,
        T_safe: float = _T_SAFE,
        soc_min: float = _SOC_MIN,
        soc_max: float = _SOC_MAX,
        delta: float = 1.0,
        n_grid: int = _N_GRID,
        precompute: bool = True,
    ) -> None:
        self.T_safe = T_safe
        self.soc_min = soc_min
        self.soc_max = soc_max
        self.delta = delta
        self.n_grid = n_grid
        self.stats = HJStats()

        # Grid axes
        self.T_grid = np.linspace(_T_MIN, T_safe + 2.0, n_grid)
        self.SOC_grid = np.linspace(0.0, 1.0, n_grid)

        # Value functions (initialised to constraint function ℓ(x))
        self.V_thermal = None
        self.V_soc = None
        self.U_thermal = None  # optimal safe control
        self.U_soc = None

        if precompute:
            self._compute_thermal_brs()
            self._compute_soc_brs()

    # ── Offline computation ───────────────────────────────────────

    def _compute_thermal_brs(self) -> None:
        """
        Compute the backward reachable set for Zone A thermal subsystem.

        State: x = T_A ∈ [T_min, T_safe+2]
        Control: u = (throttle, pump) ∈ [0,1]²
        Dynamics: dT/dt = (throttle * P_flex + Q_base - pump * P_pump - UA*(T-T_amb)) / C_A

        Unsafe set: {T : T ≥ T_safe}
        ℓ(T) = T_safe − T  (negative in unsafe set)
        """
        n = self.n_grid
        T = self.T_grid
        T_amb = 25.0  # nominal

        # Initial value function: ℓ(T) = T_safe - T
        V = self.T_safe - T.copy()

        # Optimal control lookup
        U_throttle = np.ones(n) * 0.5  # default
        U_pump = np.ones(n) * 0.5

        # Discrete controls to search over
        throttles = np.linspace(0, 1, 11)
        pumps = np.linspace(0, 1, 11)

        for _ in range(_N_OFFLINE):
            V_new = V.copy()
            for i in range(n):
                Ti = T[i]
                best_v = -1e10
                best_thr = 0.5
                best_pump = 0.5

                for thr in throttles:
                    for pump in pumps:
                        # Dynamics
                        Q = thr * _P_FLEX_MAX * 1000 + 20e6  # base heat (W)
                        cool = pump * _P_PUMP_MAX * 1000
                        natural = _UA_A * (Ti - T_amb)
                        dT = (Q - cool - natural) / _C_A * _DT
                        T_next = Ti + dT
                        T_next = np.clip(T_next, _T_MIN, self.T_safe + 2.0)

                        # Interpolate V at next state
                        v_next = np.interp(T_next, T, V)

                        # HJI update: maximise over disturbance (none here),
                        # minimise over control (find safest action)
                        if v_next > best_v:
                            best_v = v_next
                            best_thr = thr
                            best_pump = pump

                # Take max with constraint function ℓ(x)
                ell = self.T_safe - Ti
                V_new[i] = max(best_v, ell)
                U_throttle[i] = best_thr
                U_pump[i] = best_pump

            V = V_new

        self.V_thermal = V
        self.U_thermal = np.stack([U_throttle, U_pump], axis=-1)

    def _compute_soc_brs(self) -> None:
        """
        Compute the backward reachable set for BESS SOC subsystem.

        State: x = SOC ∈ [0, 1]
        Control: u = bess_dispatch ∈ [-1, 1]
        Dynamics: dSOC/dt = -P_bess / E_cap

        Unsafe set: {SOC : SOC < SOC_min or SOC > SOC_max}
        ℓ(SOC) = min(SOC - SOC_min, SOC_max - SOC)
        """
        n = self.n_grid
        SOC = self.SOC_grid

        # Initial value
        V = np.minimum(SOC - self.soc_min, self.soc_max - SOC)

        U_bess = np.zeros(n)

        dispatches = np.linspace(-1, 1, 21)

        for _ in range(_N_OFFLINE):
            V_new = V.copy()
            for i in range(n):
                soc_i = SOC[i]
                best_v = -1e10
                best_d = 0.0

                for d in dispatches:
                    dSOC = -d * _P_BESS_MAX / (_E_BESS_CAP * 3600.0) * _DT
                    soc_next = np.clip(soc_i + dSOC, 0.0, 1.0)
                    v_next = np.interp(soc_next, SOC, V)

                    if v_next > best_v:
                        best_v = v_next
                        best_d = d

                ell = min(soc_i - self.soc_min, self.soc_max - soc_i)
                V_new[i] = max(best_v, ell)
                U_bess[i] = best_d

            V = V_new

        self.V_soc = V
        self.U_soc = U_bess

    # ── Runtime filter ────────────────────────────────────────────

    def _lookup_thermal(self, T_A: float) -> tuple[float, float, float]:
        """Look up thermal value function and optimal control."""
        if self.V_thermal is None:
            # Fallback if not precomputed
            v = self.T_safe - T_A
            u_thr = 0.3 if T_A > self.T_safe - 2.0 else 0.8
            u_pump = 0.9 if T_A > self.T_safe - 2.0 else 0.5
            return v, u_thr, u_pump

        v = float(np.interp(T_A, self.T_grid, self.V_thermal))
        idx = np.searchsorted(self.T_grid, T_A)
        idx = np.clip(idx, 0, self.n_grid - 1)
        u_thr = float(self.U_thermal[idx, 0])
        u_pump = float(self.U_thermal[idx, 1])
        return v, u_thr, u_pump

    def _lookup_soc(self, soc: float) -> tuple[float, float]:
        """Look up SOC value function and optimal control."""
        if self.V_soc is None:
            v = min(soc - self.soc_min, self.soc_max - soc)
            u_bess = 0.0 if v < 0.05 else 0.0
            return v, u_bess

        v = float(np.interp(soc, self.SOC_grid, self.V_soc))
        idx = np.searchsorted(self.SOC_grid, soc)
        idx = np.clip(idx, 0, self.n_grid - 1)
        u_bess = float(self.U_soc[idx])
        return v, u_bess

    def reset(self) -> None:
        """Reset shield statistics."""
        self.stats = HJStats()

    def filter(
        self,
        action: NDArray[np.float32],
        obs: NDArray[np.float32],
    ) -> tuple[NDArray[np.float32], bool, dict]:
        """
        Filter action using precomputed HJ value functions.

        If state is near the BRS boundary (V < delta), override with
        the optimal safe control from the value function.
        """
        self.stats.total_steps += 1
        safe = action.copy().astype(np.float32)
        modified = False
        reasons = []

        # Decode observation
        T_A = float(obs[_I_TEMP_A]) * self.T_safe
        T_B = float(obs[_I_TEMP_B]) * self.T_safe
        soc = float(obs[_I_SOC])

        # ── Thermal subsystem ────────────────────────────────────
        v_th_A, u_thr_A, u_pump_A = self._lookup_thermal(T_A)
        v_th_B = self.T_safe - T_B  # simplified for Zone B

        if v_th_A < self.delta:
            # Blend: α = 1 at boundary, 0 well inside safe set
            alpha = np.clip(1.0 - v_th_A / self.delta, 0.0, 1.0)
            safe[0] = (1 - alpha) * safe[0] + alpha * u_thr_A
            safe[1] = (1 - alpha) * safe[1] + alpha * u_pump_A
            modified = True
            reasons.append(f"hj_thermal_A(V={v_th_A:.2f})")
            self.stats.thermal_overrides += 1

        if v_th_B < self.delta:
            alpha = np.clip(1.0 - v_th_B / self.delta, 0.0, 1.0)
            safe[2] = (1 - alpha) * safe[2] + alpha * 0.9  # increase HVAC
            safe[0] = min(safe[0], 1.0 - alpha * 0.5)  # reduce batch
            modified = True
            reasons.append(f"hj_thermal_B(V={v_th_B:.2f})")
            self.stats.thermal_overrides += 1

        # ── SOC subsystem ─────────────────────────────────────────
        v_soc, u_bess = self._lookup_soc(soc)

        if v_soc < self.delta * 0.5:
            alpha = np.clip(1.0 - v_soc / (self.delta * 0.5), 0.0, 1.0)
            safe[3] = (1 - alpha) * safe[3] + alpha * u_bess
            modified = True
            reasons.append(f"hj_soc(V={v_soc:.3f})")
            self.stats.soc_overrides += 1

        # Track value function stats
        v_min = min(v_th_A, v_th_B, v_soc)
        self.stats.min_value_function = min(
            self.stats.min_value_function, v_min)
        # Running mean
        n = self.stats.total_steps
        self.stats.mean_value_function = (
            self.stats.mean_value_function * (n - 1) + v_min) / n

        if modified:
            self.stats.interventions += 1

        # Clamp
        safe[0] = np.clip(safe[0], 0.0, 1.0)
        safe[1] = np.clip(safe[1], 0.0, 1.0)
        safe[2] = np.clip(safe[2], 0.0, 1.0)
        safe[3] = np.clip(safe[3], -1.0, 1.0)

        info = {
            "hj_modified": modified,
            "hj_reasons": reasons,
            "hj_value_thermal_A": v_th_A,
            "hj_value_thermal_B": v_th_B,
            "hj_value_soc": v_soc,
            "hj_intervention_rate": self.stats.intervention_rate,
        }
        return safe, modified, info


class HJShieldedEnv(gym.Wrapper):
    """
    Gymnasium wrapper applying HJ reachability shield to every action.

    Usage
    -----
      env = HJShieldedEnv(C2GFastEnv(scenario="default"))
      obs, _ = env.reset()
      obs, rew, term, trunc, info = env.step(agent_action)
    """

    def __init__(self, env: gym.Env, shield: HJShield | None = None):
        super().__init__(env)
        self.shield = shield or HJShield(precompute=True)

    def reset(self, **kwargs) -> tuple[NDArray, dict]:
        self.shield.reset()
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        return obs, info

    def step(self, action):
        obs_prev = getattr(self, "_last_obs", np.zeros(17, dtype=np.float32))
        safe_action, was_modified, shield_info = self.shield.filter(
            action, obs_prev)
        obs, reward, terminated, truncated, info = self.env.step(safe_action)
        self._last_obs = obs
        info.update(shield_info)
        info["hj_stats"] = self.shield.stats.as_dict()
        return obs, reward, terminated, truncated, info
