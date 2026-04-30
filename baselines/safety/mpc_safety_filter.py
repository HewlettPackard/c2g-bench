"""
baselines/safety/mpc_safety_filter.py  —  Model-Predictive Safety Filter
=========================================================================
A receding-horizon constrained optimisation filter that minimally modifies
the RL agent's proposed action to guarantee constraint satisfaction over
a short prediction horizon.

At each step, the filter solves:

    min_{u₀,…,u_{H-1}}  ‖u₀ − u_RL‖² + ε Σ_{k=1}^{H-1} ‖u_k‖²
    s.t.   x_{k+1} = f(x_k, u_k)           ∀ k = 0,…,H−1
           g_j(x_k) ≤ 0                     ∀ j, ∀ k = 0,…,H
           u_k ∈ [u_min, u_max]              ∀ k

Only u₀ is applied to the environment; the rest serve as a feasibility
proof that the constraint can be maintained over the horizon.

This is **the most permissive** online filter because it considers
future control authority, but it is also the most computationally
expensive: O(H × n_constraints) per step, requiring an NLP solver.

The MPC-SF uses a simplified thermal + BESS model for prediction and
SLSQP from scipy for the NLP solve.

References
----------
  [Wabersich 2021]  T. Wabersich & M. Zeilinger, "A Predictive Safety
                    Filter for Learning-Based Control of Constrained
                    Nonlinear Dynamical Systems", Automatica 2021.
  [Wabersich 2023]  T. Wabersich et al., "Data-Driven Safety Filters:
                    Hamilton-Jacobi Reachability, Control Barrier Functions,
                    and Predictive Methods", Ann. Rev. Control 2023.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
import gymnasium as gym

try:
    from scipy.optimize import minimize as scipy_minimize
except ImportError:
    scipy_minimize = None

from c2g_env.obs_indices import Fast as _F


# ─── Physical constants ───────────────────────────────────────────
_T_SAFE     = 35.0
_T_WARN     = 33.0
_SOC_MIN    = 0.10
_SOC_MAX    = 0.95
_FREQ_MAX   = 0.5
_V_MIN      = 0.90
_DT         = 5.0

# Thermal model
_C_A        = 4.0e6
_C_B        = 6.0e6
_UA_A       = 1.5e5
_UA_B       = 2.0e5
_P_FLEX_MAX = 30_000.0
_P_PUMP_MAX = 5_000.0
_P_HVAC_MAX = 8_000.0
_E_BESS_CAP = 150_000.0
_P_BESS_MAX = 50_000.0


@dataclass
class MPCSFStats:
    """Counters for monitoring MPC-SF interventions."""
    total_steps: int = 0
    interventions: int = 0
    nlp_solves: int = 0
    nlp_failures: int = 0
    mean_solve_time_ms: float = 0.0
    _solve_time_sum: float = field(default=0.0, repr=False)

    @property
    def intervention_rate(self) -> float:
        return self.interventions / max(self.total_steps, 1)

    def as_dict(self) -> dict:
        return {
            "mpcsf_total_steps": self.total_steps,
            "mpcsf_interventions": self.interventions,
            "mpcsf_intervention_rate": self.intervention_rate,
            "mpcsf_nlp_solves": self.nlp_solves,
            "mpcsf_nlp_failures": self.nlp_failures,
            "mpcsf_mean_solve_time_ms": self.mean_solve_time_ms,
        }


class MPCSafetyFilter:
    """
    Model-Predictive Safety Filter for C2GFastEnv.

    Solves a receding-horizon NLP at each step to find the closest
    safe action to the RL agent's proposal, considering future
    control authority over the prediction horizon.

    Parameters
    ----------
    horizon : int
        Number of prediction steps. Longer horizon = more permissive
        but slower. Default 5 (= 25 seconds lookahead).
    T_safe : float
        Thermal limit [°C].
    soc_min, soc_max : float
        BESS SOC bounds.
    margin : float
        Safety margin [°C] below T_safe.
    regularisation : float
        Weight on future control actions (keeps horizon actions small).
    """

    def __init__(
        self,
        horizon: int = 5,
        T_safe: float = _T_SAFE,
        soc_min: float = _SOC_MIN,
        soc_max: float = _SOC_MAX,
        margin: float = 0.5,
        regularisation: float = 0.01,
    ) -> None:
        self.H = horizon
        self.T_safe = T_safe
        self.T_safe_margin = T_safe - margin
        self.soc_min = soc_min + 0.02
        self.soc_max = soc_max - 0.02
        self.reg = regularisation
        self.stats = MPCSFStats()

    def reset(self) -> None:
        self.stats = MPCSFStats()

    def _decode_obs(self, obs: NDArray) -> dict:
        return {
            "T_A":    float(obs[_F.TEMP_A]) * self.T_safe,
            "T_B":    float(obs[_F.TEMP_B]) * self.T_safe,
            "SOC":    float(obs[_F.SOC]),
            "p_base": float(obs[_F.P_BASE]) * 250_000.0,
            "p_flex": float(obs[_F.P_FLEX]) * 250_000.0,
            "T_amb":  float(obs[_F.T_AMB]) * 50.0,
            "freq_dev": float(obs[_F.FREQ_DEV]) * 0.5,
            "v_pcc":  float(obs[_F.VPCC]),
        }

    def _predict_state(self, s: dict, u: NDArray) -> dict:
        """One-step forward prediction using simplified thermal + BESS model."""
        throttle = float(u[0])
        pump = float(u[1])
        hvac = float(u[2])
        bess = float(u[3])
        T_amb = s["T_amb"]

        # Zone A
        q_a = throttle * _P_FLEX_MAX * 1000 + s["p_base"] * 0.4 * 1000
        cool_a = pump * _P_PUMP_MAX * 1000
        T_A = s["T_A"] + (q_a - cool_a - _UA_A * (s["T_A"] - T_amb)) / _C_A * _DT

        # Zone B
        q_b = s["p_base"] * 0.6 * 1000
        cool_b = hvac * _P_HVAC_MAX * 1000
        T_B = s["T_B"] + (q_b - cool_b - _UA_B * (s["T_B"] - T_amb)) / _C_B * _DT

        # SOC
        dSOC = -bess * _P_BESS_MAX / (_E_BESS_CAP * 3600.0) * _DT
        SOC = np.clip(s["SOC"] + dSOC, 0.0, 1.0)

        return {
            "T_A": float(np.clip(T_A, 10.0, 50.0)),
            "T_B": float(np.clip(T_B, 10.0, 50.0)),
            "SOC": float(SOC),
            "p_base": s["p_base"],
            "p_flex": s["p_flex"],
            "T_amb": T_amb,
            "freq_dev": s["freq_dev"],
            "v_pcc": s["v_pcc"],
        }

    def _check_needs_filter(self, s: dict) -> bool:
        """Quick check: does the current state need MPC-SF?"""
        T_margin_A = self.T_safe_margin - s["T_A"]
        T_margin_B = self.T_safe_margin - s["T_B"]
        SOC_margin_low = s["SOC"] - self.soc_min
        SOC_margin_high = self.soc_max - s["SOC"]

        # Only solve MPC if some constraint is within concern range
        return (
            T_margin_A < 3.0 or
            T_margin_B < 3.0 or
            SOC_margin_low < 0.10 or
            SOC_margin_high < 0.10
        )

    def _solve_mpc_nlp(
        self, u_ref: NDArray, s: dict
    ) -> tuple[NDArray, bool]:
        """
        Solve the MPC-SF NLP.

        Decision variables: u = [u_0, u_1, …, u_{H-1}] ∈ R^{4H}
        """
        import time as _time
        t0 = _time.perf_counter()
        self.stats.nlp_solves += 1

        H = self.H
        n_u = 4

        # Initial guess: repeat u_ref for all steps
        x0 = np.tile(u_ref, H)

        # Bounds per step
        lb = np.tile([0.0, 0.0, 0.0, -1.0], H)
        ub = np.tile([1.0, 1.0, 1.0, 1.0], H)
        bounds = list(zip(lb, ub))

        # Objective: min ‖u₀ − u_ref‖² + ε Σ ‖u_k‖²
        def objective(x):
            u0 = x[:n_u]
            cost = float(np.sum((u0 - u_ref) ** 2))
            for k in range(1, H):
                uk = x[k * n_u:(k + 1) * n_u]
                cost += self.reg * float(np.sum(uk ** 2))
            return cost

        # Constraints: state constraints at each future step
        def constraints_fn(x):
            state = dict(s)
            cons = []
            for k in range(H):
                uk = x[k * n_u:(k + 1) * n_u]
                state = self._predict_state(state, uk)
                # T_A ≤ T_safe_margin  →  T_safe_margin - T_A ≥ 0
                cons.append(self.T_safe_margin - state["T_A"])
                cons.append(self.T_safe_margin - state["T_B"])
                cons.append(state["SOC"] - self.soc_min)
                cons.append(self.soc_max - state["SOC"])
            return np.array(cons)

        constraint = {"type": "ineq", "fun": constraints_fn}

        if scipy_minimize is None:
            self.stats.nlp_failures += 1
            elapsed = (_time.perf_counter() - t0) * 1000
            self.stats._solve_time_sum += elapsed
            self.stats.mean_solve_time_ms = (
                self.stats._solve_time_sum / self.stats.nlp_solves)
            return u_ref.copy(), False

        try:
            result = scipy_minimize(
                objective,
                x0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraint,
                options={"maxiter": 30, "ftol": 1e-7},
            )
            elapsed = (_time.perf_counter() - t0) * 1000
            self.stats._solve_time_sum += elapsed
            self.stats.mean_solve_time_ms = (
                self.stats._solve_time_sum / self.stats.nlp_solves)

            if result.success:
                u0 = result.x[:n_u].astype(np.float32)
                u0 = np.clip(u0, [0, 0, 0, -1], [1, 1, 1, 1])
                was_modified = float(np.linalg.norm(u0 - u_ref)) > 1e-4
                return u0, was_modified
            else:
                self.stats.nlp_failures += 1
                return self._conservative_fallback(u_ref, s), True
        except Exception:
            elapsed = (_time.perf_counter() - t0) * 1000
            self.stats._solve_time_sum += elapsed
            self.stats.mean_solve_time_ms = (
                self.stats._solve_time_sum / self.stats.nlp_solves)
            self.stats.nlp_failures += 1
            return self._conservative_fallback(u_ref, s), True

    def _conservative_fallback(self, u_ref: NDArray, s: dict) -> NDArray:
        """Conservative fallback if NLP fails."""
        safe = u_ref.copy()
        if s["T_A"] > self.T_safe_margin or s["T_B"] > self.T_safe_margin:
            safe[0] = min(safe[0], 0.3)
            safe[1] = max(safe[1], 0.8)
            safe[2] = max(safe[2], 0.8)
        if s["SOC"] < self.soc_min + 0.03 and safe[3] > 0:
            safe[3] = 0.0
        if s["SOC"] > self.soc_max - 0.03 and safe[3] < 0:
            safe[3] = 0.0
        return np.clip(safe, [0, 0, 0, -1], [1, 1, 1, 1]).astype(np.float32)

    def filter(
        self,
        action: NDArray[np.float32],
        obs: NDArray[np.float32],
    ) -> tuple[NDArray[np.float32], bool, dict]:
        """
        Project action through the MPC safety filter.

        Parameters
        ----------
        action : ndarray, shape (4,)
        obs : ndarray, shape (16,) or (17,)

        Returns
        -------
        safe_action : ndarray, shape (4,)
        was_modified : bool
        info : dict
        """
        self.stats.total_steps += 1
        s = self._decode_obs(obs)

        if not self._check_needs_filter(s):
            info = {
                "mpcsf_modified": False,
                "mpcsf_skipped": True,
                "mpcsf_intervention_rate": self.stats.intervention_rate,
            }
            return action.copy().astype(np.float32), False, info

        safe_action, was_modified = self._solve_mpc_nlp(
            action.copy().astype(np.float32), s)

        if was_modified:
            self.stats.interventions += 1

        info = {
            "mpcsf_modified": was_modified,
            "mpcsf_skipped": False,
            "mpcsf_solve_time_ms": self.stats.mean_solve_time_ms,
            "mpcsf_intervention_rate": self.stats.intervention_rate,
        }
        return safe_action, was_modified, info


class MPCSFShieldedEnv(gym.Wrapper):
    """
    Gymnasium wrapper applying MPC Safety Filter to every action.

    Usage
    -----
      env = MPCSFShieldedEnv(C2GFastEnv(scenario="default"))
      obs, _ = env.reset()
      obs, rew, term, trunc, info = env.step(agent_action)
    """

    def __init__(self, env: gym.Env, shield: MPCSafetyFilter | None = None):
        super().__init__(env)
        self.shield = shield or MPCSafetyFilter()

    def reset(self, **kwargs) -> tuple[NDArray, dict]:
        self.shield.reset()
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        return obs, info

    def step(self, action):
        obs_prev = getattr(self, "_last_obs", np.zeros(18, dtype=np.float32))
        safe_action, was_modified, shield_info = self.shield.filter(
            action, obs_prev)
        obs, reward, terminated, truncated, info = self.env.step(safe_action)
        self._last_obs = obs
        info.update(shield_info)
        info["mpcsf_stats"] = self.shield.stats.as_dict()
        return obs, reward, terminated, truncated, info
