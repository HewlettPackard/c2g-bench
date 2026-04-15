"""
baselines/safety/cbf_shield.py  —  Control Barrier Function Safety Filter
==========================================================================
A runtime safety filter that projects RL actions into the safe set
using Control Barrier Functions (CBFs) [Ames et al., ECC 2019].

For each hard constraint C_i, we define a barrier function h_i(x) such
that h_i(x) > 0 ⟹ constraint C_i is satisfied. The CBF condition
requires:

    ḣ_i(x, u) + α_i · h_i(x) ≥ 0      ∀ i ∈ {1, …, 5}

where α_i > 0 is a class-K parameter controlling the decay rate.
This ensures the safe set {x : h_i(x) ≥ 0} is forward-invariant.

At each step, the filter solves a Quadratic Program (QP):

    min_u  ‖u − u_RL‖²
    s.t.   ḣ_i(x, u) + α_i · h_i(x) ≥ 0   ∀ i
           u ∈ [u_min, u_max]

This is **more permissive** than the Simplex shield because it exploits
the system dynamics model to allow actions that would be blocked by the
conservative analytic bounds.

Hard Safety Constraints (same as Simplex shield)
-------------------------------------------------
  C1. T_A < T_safe (35°C)        → h1(x) = T_safe − T_A
  C2. T_B < T_safe (35°C)        → h2(x) = T_safe − T_B
  C3. SOC ∈ [SOC_min, SOC_max]   → h3(x) = SOC − SOC_min, h4(x) = SOC_max − SOC
  C4. |Δf| < 0.5 Hz              → h5(x) = 0.5 − |Δf|
  C5. V_pcc > 0.90 pu            → h6(x) = V_pcc − 0.90

Barrier Function Derivatives
-----------------------------
We use the thermal ODE model to compute ḣ analytically:
  dT_A/dt ≈ (Q_A − UA_A·(T_A − T_amb) − pump_cooling) / C_A
  dT_B/dt ≈ (Q_B − UA_B·(T_B − T_amb) − hvac_cooling) / C_B
  dSOC/dt ≈ −P_bess / E_cap

where the action u = [throttle, pump, hvac, bess] enters linearly
through the cooling and BESS power terms.

The QP is solved using scipy.optimize.minimize (SLSQP) which handles
the linear inequality constraints efficiently.

References
----------
  [Ames 2019]  A. Ames et al., "Control Barrier Functions: Theory and
               Applications", ECC 2019.
  [Ames 2017]  A. Ames et al., "Control Barrier Function Based
               Quadratic Programs for Safety Critical Systems",
               IEEE TAC 2017.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
import gymnasium as gym

try:
    from scipy.optimize import minimize
except ImportError:
    minimize = None  # graceful fallback


# ─── Physical constants (from C2GFastEnv / config.yaml) ───────────
_T_SAFE     = 35.0       # °C — Silicon thermal limit
_T_AMB_NOM  = 25.0       # °C — Default ambient (overridden by obs)
_SOC_MIN    = 0.10
_SOC_MAX    = 0.95
_FREQ_MAX   = 0.5        # Hz — UFLS threshold
_V_MIN      = 0.90       # pu — UV relay

# Thermal model parameters (simplified from thermal.py)
_C_A        = 4.0e6      # J/°C — Zone A thermal capacitance
_C_B        = 6.0e6      # J/°C — Zone B thermal capacitance
_UA_A       = 1.5e5      # W/°C — Zone A heat-loss coefficient
_UA_B       = 2.0e5      # W/°C — Zone B heat-loss coefficient
_DT         = 5.0        # seconds per tick

# Power scaling
_P_FLEX_MAX = 30_000.0   # kW — max batch power (Zone A)
_P_PUMP_MAX = 5_000.0    # kW — max CDU pump cooling effect
_P_HVAC_MAX = 8_000.0    # kW — max HVAC cooling effect
_E_BESS_CAP = 150_000.0  # kWh — BESS energy capacity
_P_BESS_MAX = 50_000.0   # kW — BESS max power

# Observation indices (C2GFastEnv, 16-D + backlog at 16)
_I_TEMP_A   = 0
_I_TEMP_B   = 1
_I_SOC      = 2
_I_P_BASE   = 3
_I_P_FLEX   = 4
_I_REGD     = 6
_I_T_AMB    = 13
_I_FREQ_DEV = 14
_I_VPCC     = 15

# Default CBF class-K parameters (higher = more conservative)
_DEFAULT_ALPHA = {
    "thermal_A": 0.5,
    "thermal_B": 0.5,
    "soc_low":   1.0,
    "soc_high":  1.0,
    "frequency": 2.0,
    "voltage":   1.5,
}


@dataclass
class CBFStats:
    """Counters for monitoring CBF filter interventions."""
    total_steps: int = 0
    interventions: int = 0
    qp_solves: int = 0
    qp_failures: int = 0
    thermal_active: int = 0
    soc_active: int = 0
    freq_active: int = 0
    voltage_active: int = 0
    mean_modification: float = 0.0
    _modification_sum: float = field(default=0.0, repr=False)

    @property
    def intervention_rate(self) -> float:
        return self.interventions / max(self.total_steps, 1)

    @property
    def qp_failure_rate(self) -> float:
        return self.qp_failures / max(self.qp_solves, 1)

    def as_dict(self) -> dict:
        return {
            "cbf_total_steps": self.total_steps,
            "cbf_interventions": self.interventions,
            "cbf_intervention_rate": self.intervention_rate,
            "cbf_qp_solves": self.qp_solves,
            "cbf_qp_failures": self.qp_failures,
            "cbf_thermal_active": self.thermal_active,
            "cbf_soc_active": self.soc_active,
            "cbf_freq_active": self.freq_active,
            "cbf_voltage_active": self.voltage_active,
            "cbf_mean_modification": self.mean_modification,
        }


class CBFShield:
    """
    Control Barrier Function safety filter for C2GFastEnv.

    Solves a QP at each step to find the closest safe action to the
    RL agent's proposal. More permissive than the Simplex shield
    because it uses the system dynamics model.

    Parameters
    ----------
    T_safe : float
        Silicon thermal limit [°C].
    soc_min, soc_max : float
        BESS SOC operational bounds.
    freq_max : float
        Maximum allowed |Δf| [Hz].
    v_min : float
        Minimum PCC voltage [pu].
    alphas : dict
        Class-K function parameters for each barrier.
        Higher values = more conservative.
    margin : float
        Extra safety margin subtracted from each barrier.
        Accounts for model mismatch.
    """

    def __init__(
        self,
        T_safe: float = _T_SAFE,
        soc_min: float = _SOC_MIN,
        soc_max: float = _SOC_MAX,
        freq_max: float = _FREQ_MAX,
        v_min: float = _V_MIN,
        alphas: dict[str, float] | None = None,
        margin: float = 0.5,
    ) -> None:
        self.T_safe = T_safe
        self.soc_min = soc_min
        self.soc_max = soc_max
        self.freq_max = freq_max
        self.v_min = v_min
        self.alphas = alphas or dict(_DEFAULT_ALPHA)
        self.margin = margin
        self.stats = CBFStats()

    def reset(self) -> None:
        self.stats = CBFStats()

    # ── Barrier functions ─────────────────────────────────────────

    def _decode_obs(self, obs: NDArray) -> dict:
        """Extract physical states from normalised observation."""
        return {
            "T_A":     float(obs[_I_TEMP_A]) * self.T_safe,
            "T_B":     float(obs[_I_TEMP_B]) * self.T_safe,
            "SOC":     float(obs[_I_SOC]),
            "p_base":  float(obs[_I_P_BASE]) * 250_000.0,   # kW
            "p_flex":  float(obs[_I_P_FLEX]) * 250_000.0,   # kW
            "T_amb":   float(obs[_I_T_AMB]) * 50.0,         # approx denorm
            "freq_dev": float(obs[_I_FREQ_DEV]) * 0.5,      # Hz
            "v_pcc":   float(obs[_I_VPCC]),
        }

    def _barrier_values(self, s: dict) -> dict:
        """Compute h_i(x) for each constraint."""
        m = self.margin
        return {
            "thermal_A": self.T_safe - s["T_A"] - m,
            "thermal_B": self.T_safe - s["T_B"] - m,
            "soc_low":   s["SOC"] - self.soc_min - 0.02,
            "soc_high":  self.soc_max - s["SOC"] - 0.02,
            "frequency": self.freq_max - abs(s["freq_dev"]) - 0.05,
            "voltage":   s["v_pcc"] - self.v_min - 0.01,
        }

    def _barrier_derivatives(
        self, s: dict, u: NDArray
    ) -> dict:
        """
        Compute ḣ_i(x, u) — the time derivative of each barrier.

        Action u = [throttle, pump_speed, hvac_effort, bess_dispatch]
        """
        throttle  = float(u[0])
        pump      = float(u[1])
        hvac      = float(u[2])
        bess      = float(u[3])

        T_A, T_B = s["T_A"], s["T_B"]
        T_amb = s["T_amb"]

        # Zone A thermal derivative:
        # dT_A/dt = (Q_IT_A - pump_cooling - natural_loss) / C_A
        # Q_IT_A ∝ throttle * p_flex_max   (batch load drives heat)
        # pump_cooling ∝ pump * P_PUMP_MAX
        q_it_a = throttle * _P_FLEX_MAX + s["p_base"] * 0.4  # Zone A fraction
        cool_a = pump * _P_PUMP_MAX
        natural_a = _UA_A * (T_A - T_amb)
        dT_A_dt = (q_it_a - cool_a - natural_a) / _C_A * 1000.0  # kW to W

        # Zone B thermal derivative:
        q_it_b = s["p_base"] * 0.6  # Zone B fraction (DLRM)
        cool_b = hvac * _P_HVAC_MAX
        natural_b = _UA_B * (T_B - T_amb)
        dT_B_dt = (q_it_b - cool_b - natural_b) / _C_B * 1000.0

        # SOC derivative: dSOC/dt = -P_bess / E_cap
        # bess > 0 means discharge (SOC decreases)
        dSOC_dt = -bess * _P_BESS_MAX / (_E_BESS_CAP * 3600.0)  # per second

        # Frequency derivative: approximated as small perturbation
        # Facility power draw affects grid frequency (simplified model)
        # Higher facility power → more negative freq deviation
        # We approximate dΔf/dt ≈ 0 (frequency changes are exogenous)
        # CBF constraint on freq is therefore: α * h_freq ≥ 0
        # which means we just need h_freq ≥ 0 (the barrier itself)
        d_freq = 0.0

        # Voltage derivative: also largely exogenous
        d_voltage = 0.0

        return {
            "thermal_A": -dT_A_dt * _DT,  # ḣ = -dT/dt (barrier is T_safe - T)
            "thermal_B": -dT_B_dt * _DT,
            "soc_low":   dSOC_dt * _DT,   # ḣ = dSOC/dt
            "soc_high":  -dSOC_dt * _DT,  # ḣ = -dSOC/dt
            "frequency": d_freq,
            "voltage":   d_voltage,
        }

    # ── QP solver ─────────────────────────────────────────────────

    def _solve_cbf_qp(
        self,
        u_ref: NDArray,
        s: dict,
        h_vals: dict,
    ) -> tuple[NDArray, bool, list[str]]:
        """
        Solve:  min ‖u − u_ref‖²
                s.t.  ḣ_i(x,u) + α_i h_i(x) ≥ 0   ∀ i
                      u ∈ [u_min, u_max]

        Returns (safe_action, was_modified, active_constraints).
        """
        active_constraints: list[str] = []

        # Check if any barrier is close to violation
        needs_qp = False
        for name, h in h_vals.items():
            alpha = self.alphas.get(name, 1.0)
            if h < alpha * 2.0:  # barrier is within concern range
                needs_qp = True
                active_constraints.append(name)

        if not needs_qp:
            return u_ref.copy(), False, []

        self.stats.qp_solves += 1

        # SLSQP-based QP
        bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (-1.0, 1.0)]

        def objective(u):
            return float(np.sum((u - u_ref) ** 2))

        def grad_objective(u):
            return 2.0 * (u - u_ref)

        constraints = []
        for name in active_constraints:
            alpha = self.alphas.get(name, 1.0)
            h = h_vals[name]

            def make_cbf_constraint(cname, calpha, ch):
                def cbf_ineq(u):
                    u_arr = np.array(u, dtype=np.float32)
                    hdot = self._barrier_derivatives(s, u_arr)
                    return hdot[cname] + calpha * ch
                return cbf_ineq

            constraints.append({
                "type": "ineq",
                "fun": make_cbf_constraint(name, alpha, h),
            })

        if minimize is None:
            # Fallback: no scipy → use conservative projection
            self.stats.qp_failures += 1
            return self._fallback_projection(u_ref, s, h_vals), True, active_constraints

        try:
            result = minimize(
                objective,
                u_ref.copy(),
                method="SLSQP",
                jac=grad_objective,
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 20, "ftol": 1e-8},
            )
            if result.success:
                safe = np.clip(result.x.astype(np.float32),
                               [0, 0, 0, -1], [1, 1, 1, 1])
                modification = float(np.linalg.norm(safe - u_ref))
                was_modified = modification > 1e-4
                return safe, was_modified, active_constraints
            else:
                self.stats.qp_failures += 1
                return self._fallback_projection(u_ref, s, h_vals), True, active_constraints
        except Exception:
            self.stats.qp_failures += 1
            return self._fallback_projection(u_ref, s, h_vals), True, active_constraints

    def _fallback_projection(
        self, u_ref: NDArray, s: dict, h_vals: dict
    ) -> NDArray:
        """Conservative fallback when QP fails — similar to Simplex."""
        safe = u_ref.copy()
        if h_vals["thermal_A"] < 1.0 or h_vals["thermal_B"] < 1.0:
            safe[0] = min(safe[0], 0.3)   # reduce batch throttle
            safe[1] = max(safe[1], 0.8)   # increase pump
            safe[2] = max(safe[2], 0.8)   # increase HVAC
        if h_vals["soc_low"] < 0.05 and safe[3] > 0:
            safe[3] = 0.0
        if h_vals["soc_high"] < 0.05 and safe[3] < 0:
            safe[3] = 0.0
        return np.clip(safe, [0, 0, 0, -1], [1, 1, 1, 1]).astype(np.float32)

    # ── Main filter API ───────────────────────────────────────────

    def filter(
        self,
        action: NDArray[np.float32],
        obs: NDArray[np.float32],
    ) -> tuple[NDArray[np.float32], bool, dict]:
        """
        Project action into the CBF-safe set.

        Parameters
        ----------
        action : ndarray, shape (4,)
            [throttle_batch, pump_speed_A, hvac_effort, bess_dispatch]
        obs : ndarray, shape (16,) or (17,)
            Current observation from C2GFastEnv.

        Returns
        -------
        safe_action : ndarray, shape (4,)
        was_modified : bool
        info : dict
        """
        self.stats.total_steps += 1

        s = self._decode_obs(obs)
        h_vals = self._barrier_values(s)
        safe_action, was_modified, active = self._solve_cbf_qp(
            action.copy().astype(np.float32), s, h_vals
        )

        if was_modified:
            self.stats.interventions += 1
            mod = float(np.linalg.norm(safe_action - action))
            self.stats._modification_sum += mod
            self.stats.mean_modification = (
                self.stats._modification_sum / self.stats.interventions
            )

            # Track which constraints were active
            for name in active:
                if "thermal" in name:
                    self.stats.thermal_active += 1
                elif "soc" in name:
                    self.stats.soc_active += 1
                elif "freq" in name:
                    self.stats.freq_active += 1
                elif "voltage" in name:
                    self.stats.voltage_active += 1

        info = {
            "cbf_modified": was_modified,
            "cbf_active_constraints": active,
            "cbf_barrier_values": h_vals,
            "cbf_intervention_rate": self.stats.intervention_rate,
        }
        return safe_action, was_modified, info


class CBFShieldedEnv(gym.Wrapper):
    """
    Gymnasium wrapper applying a CBF safety filter to every action.

    Usage
    -----
      env = CBFShieldedEnv(C2GFastEnv(scenario="default"))
      obs, _ = env.reset()
      obs, rew, term, trunc, info = env.step(agent_action)  # CBF-filtered
    """

    def __init__(self, env: gym.Env, shield: CBFShield | None = None):
        super().__init__(env)
        self.shield = shield or CBFShield()

    def reset(self, **kwargs) -> tuple[NDArray, dict]:
        self.shield.reset()
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        return obs, info

    def step(self, action):
        obs_prev = getattr(self, "_last_obs", np.zeros(17, dtype=np.float32))
        safe_action, was_modified, shield_info = self.shield.filter(action, obs_prev)
        obs, reward, terminated, truncated, info = self.env.step(safe_action)
        self._last_obs = obs
        info.update(shield_info)
        info["cbf_stats"] = self.shield.stats.as_dict()
        return obs, reward, terminated, truncated, info


class CBFShieldedAgent:
    """
    Wraps any SB3-compatible agent with a CBF safety filter.

    Usage
    -----
      agent = PPO.load("my_model")
      safe_agent = CBFShieldedAgent(agent)
      action, _ = safe_agent.predict(obs)  # CBF-safe
    """

    def __init__(self, agent, shield: CBFShield | None = None):
        self.agent = agent
        self.shield = shield or CBFShield()

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        raw_action, state = self.agent.predict(
            obs, state=state, episode_start=episode_start,
            deterministic=deterministic,
        )
        single = raw_action.ndim == 1
        if single:
            safe_action, _, _ = self.shield.filter(raw_action, obs)
        else:
            safe_actions = []
            for a, o in zip(raw_action, obs):
                sa, _, _ = self.shield.filter(a, o)
                safe_actions.append(sa)
            safe_action = np.array(safe_actions, dtype=np.float32)
        return safe_action, state
