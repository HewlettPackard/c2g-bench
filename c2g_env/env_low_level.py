"""
Step 2.1 — C2G-FastEnv  (Low-Level / Hardware-Controller Environment)
======================================================================
A ``gymnasium.Env`` that wraps all five C2G-Bench simulators.

Every call to ``step()`` advances the physical simulation by one 5-second
interval.  The RL agent controls four levers:

  Lever 1  ``throttle_batch``  – DVFS factor for schedulable batch jobs  [0, 1]
  Lever 2  ``pump_speed_A``    – Zone-A CDU circulating pump speed       [0, 1]
                                  (modulates liquid-loop heat-transfer;
                                   lower speed stores heat in water-loop
                                   thermal mass for grid regulation)
  Lever 3  ``hvac_effort``     – Zone-B HVAC fan + chiller effort         [0, 1]
  Lever 4  ``bess_dispatch``   – BESS power command (−1 = full charge,
                                  +1 = full discharge / inject to grid)   [-1, 1]

The environment is designed for facility-level frequency regulation
(PJM/NYISO RegD).  The reward incentivises:
  1. Batch throughput            (do not shed batch jobs unnecessarily)
  2. Grid-signal tracking        (match committed ΔP_demand within ±5%)
  3. Thermal safety              (zero tolerance for temp > T_warn)
  4. BESS health                 (avoid deep discharge below SOC_min + 2%)

Observation space  (14-D, all values normalised to approximately [0, 1])
------------------------------------------------------------------------
  [0]  temp_A_norm        = T_A / T_safe
  [1]  temp_B_norm        = T_B / T_safe
  [2]  bess_soc           = BESS state-of-charge                  ∈ [0, 1]
  [3]  p_base_norm        = p_base_kw / FACILITY_CAP_KW
  [4]  p_flex_nom_norm    = p_flex_nom_kw / FACILITY_CAP_KW
  [5]  p_facility_norm    = p_facility_mw × 1000 / FACILITY_CAP_KW
  [6]  regd_signal        = normalised RegD signal                 ∈ [-1, 1]
  [7]  lmp_norm           = LMP / 200  (soft cap at $200/MWh)
  [8]  grid_load_norm     = NYISO zone load / historical max       ∈ [0, 1]
  [9]  is_spike           = GenAI burst flag                       ∈ {0, 1}
  [10] prev_throttle      = throttle from previous step            ∈ [0, 1]
  [11] prev_pump_speed    = pump speed from previous step          ∈ [0, 1]
  [12] pue_norm           = PUE / 2.5

Reward
------
  r = α·throttle
    − β·|ΔP_demanded_kW − ΔP_actual_kW| / (committed_mw × 1000)
    − γ·[max(0, T_A − T_warn_A) + max(0, T_B − T_warn_B)]
    − soc_penalty   if SOC < SOC_min + 0.02

Termination
-----------
  • ``terminated=True``  if T_A > T_safe **or** T_B > T_safe  (hardware fault)
  • ``truncated=True``   after ``episode_ticks`` steps (time limit)

Usage
-----
  from c2g_env import C2GFastEnv
  env = C2GFastEnv(scenario="scenario_a")
  obs, info = env.reset(seed=0)
  obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml
import gymnasium as gym
from gymnasium import spaces

from c2g_env.simulators.workload   import WorkloadOrchestrator
from c2g_env.simulators.thermal    import ThermalTwin
from c2g_env.simulators.electrical import DatacenterElectrical
from c2g_env.simulators.bess       import BESSModel
from c2g_env.simulators.macro_grid import MacroGridSignal
from c2g_env.simulators.renewable  import RenewableGen
from c2g_env.simulators.weather    import WeatherLoader

# ---------------------------------------------------------------------------
# Constants — kept in sync with electrical.py rack parameters
# ---------------------------------------------------------------------------
_FACILITY_CAP_KW = 250_000.0    # Nameplate IT capacity (kW)

# Zone A: 2000 racks (800 GenAI base + 1200 batch flex)
_ZA_N_RACKS  = 2_000
_ZA_P_IDLE   = 8.0              # kW per rack idle
_ZA_P_RANGE  = 75.0 - 8.0      # kW per rack: (p_max - p_idle)
_ZA_ALPHA    = 1.4              # GPU super-linear exponent

# Zone B: 2500 racks (all DLRM / inference)
_ZB_N_RACKS  = 2_500
_ZB_P_IDLE   = 4.0
_ZB_P_RANGE  = 40.0 - 4.0
_ZB_ALPHA    = 1.2

_CONFIG_PATH = Path(__file__).parent / "config.yaml"


def _inverse_rack_util(p_zone_kw: float, n_racks: int,
                       p_idle_kw: float, p_range_kw: float,
                       alpha: float) -> float:
    """
    Invert  P(u) = N × [p_idle + p_range × u^alpha]  to recover u ∈ [0, 1].

    Used to pass the workload-derived IT power back through the electrical
    accounting model (which takes utilisation fractions as input).
    """
    per_rack = p_zone_kw / max(n_racks, 1)
    frac = (per_rack - p_idle_kw) / max(p_range_kw, 1e-9)
    frac = float(np.clip(frac, 0.0, 1.0))
    return frac ** (1.0 / alpha)


class C2GFastEnv(gym.Env):
    """
    C2G-Bench Low-Level Environment (Hardware Controller).

    Parameters
    ----------
    scenario : str
        Key in ``config.yaml``: ``"default"``, ``"scenario_a"``,
        ``"scenario_b"``, or ``"scenario_c"``.
    config_path : str or Path, optional
        Override path to ``config.yaml``.
    """

    metadata = {"render_modes": []}

    # ------------------------------------------------------------------
    # Space definitions
    # ------------------------------------------------------------------
    _ACT_LOW  = np.array([0.0,  0.0,  0.0, -1.0], dtype=np.float32)  # throttle, pump, hvac, bess
    _ACT_HIGH = np.array([1.0,  1.0,  1.0,  1.0], dtype=np.float32)

    _OBS_LOW  = np.array([
        0.0,  # temp_A_norm
        0.0,  # temp_B_norm
        0.0,  # bess_soc
        0.0,  # p_base_norm
        0.0,  # p_flex_nom_norm
        0.0,  # p_facility_norm
       -1.0,  # regd_signal (signed)
        0.0,  # lmp_norm
        0.0,  # grid_load_norm
        0.0,  # is_spike
        0.0,  # prev_throttle
        0.0,  # prev_pump_speed
        0.0,  # pue_norm
        0.0,  # T_amb_norm
    ], dtype=np.float32)

    _OBS_HIGH = np.array([
        2.0,  # temp_A_norm (can temporarily exceed 1)
        2.0,  # temp_B_norm
        1.0,  # bess_soc
        1.0,  # p_base_norm
        1.0,  # p_flex_nom_norm
        2.0,  # p_facility_norm
        1.0,  # regd_signal
        1.0,  # lmp_norm
        1.0,  # grid_load_norm
        1.0,  # is_spike
        1.0,  # prev_throttle
        1.0,  # prev_pump_speed
        2.0,  # pue_norm
        1.0,  # T_amb_norm
    ], dtype=np.float32)

    def __init__(
        self,
        scenario: str = "default",
        config_path: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        cfg_path = Path(config_path) if config_path else _CONFIG_PATH
        with open(cfg_path) as fh:
            full_cfg = yaml.safe_load(fh)

        self._gcfg   = full_cfg["global"]
        self._rcfg   = full_cfg["reward"]
        self._scfg   = full_cfg[scenario]
        self._scenario = scenario

        self.action_space = spaces.Box(
            low=self._ACT_LOW, high=self._ACT_HIGH, dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=self._OBS_LOW, high=self._OBS_HIGH, dtype=np.float32
        )

        # Simulator handles (reset on each reset() call)
        self._workload  = None
        self._thermal   = None
        self._elec      = None
        self._bess      = None
        self._grid      = None
        self._renewable = None
        self._weather   = None

        # Episode state
        self._tick           = 0
        self._prev_throttle  = 1.0
        self._prev_pump_speed = 1.0
        self._committed_mw   = float(self._scfg["committed_mw"])
        self._episode_ticks  = int(self._gcfg["episode_ticks"])
        self._dt             = float(self._gcfg["dt_seconds"])

    # ------------------------------------------------------------------
    # Gymnasium core
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Re-initialise all simulators and return the initial observation.

        Parameters
        ----------
        seed : int, optional
            RNG seed passed to all stochastic simulators.
        options : dict, optional
            Optional dictionary that may contain ``"scenario"`` key to
            override the scenario for this episode.
        """
        super().reset(seed=seed)
        rng_seed = seed if seed is not None else 42

        # Allow per-episode scenario override
        if options and "scenario" in options:
            cfg_path = _CONFIG_PATH
            with open(cfg_path) as fh:
                full_cfg = yaml.safe_load(fh)
            self._scfg     = full_cfg[options["scenario"]]
            self._scenario = options["scenario"]
        scfg = self._scfg
        gcfg = self._gcfg

        # --- Build / reset simulators ------------------------------------
        self._workload = WorkloadOrchestrator(
            trace_dir=gcfg["trace_dir"], seed=rng_seed
        )
        self._thermal = ThermalTwin(dt_seconds=self._dt)
        self._thermal.reset()
        self._elec = DatacenterElectrical()
        self._elec.reset()
        self._bess = BESSModel(dt_seconds=self._dt)
        self._bess.reset()
        self._grid = MacroGridSignal(
            energy_dir=gcfg["energy_dir"],
            zone=gcfg["nyiso_zone"],
            dt_seconds=self._dt,
            committed_mw=float(scfg["committed_mw"]),
            seed=rng_seed,
            market=gcfg.get("grid_market", "nyiso_nyc"),
        )
        self._renewable = RenewableGen(renewable_dir=gcfg["renewable_dir"])
        self._weather_driven = bool(scfg.get("weather_driven", True))
        _weather_market = scfg.get(
            "weather_market",
            scfg.get("grid_market", gcfg.get("grid_market", "nyiso_nyc")),
        )
        self._weather = WeatherLoader(
            weather_dir=gcfg.get("weather_dir", "data/processed/weather"),
            market=_weather_market,
            dt_seconds=self._dt,
            fallback_temp_c=float(scfg["T_amb"]),
        )

        # --- Apply scenario parameters -----------------------------------
        # T_amb: set initial value from scenario config.
        # When weather_driven=True, this is overridden each tick by WeatherLoader.
        # When weather_driven=False (stress scenarios), this fixed value is used throughout.
        self._thermal.T_amb = float(scfg["T_amb"])
        if scfg.get("cooling_fault", False):
            self._thermal.set_cooling_fault(
                active=True,
                fault_factor=float(scfg.get("cooling_fault_factor", 0.4)),
            )
        # Override initial BESS SOC if specified
        bess_soc_init = float(scfg.get("bess_soc_init", 0.5))
        self._bess._soc = bess_soc_init

        self._committed_mw   = float(scfg["committed_mw"])
        self._tick           = 0
        self._prev_throttle  = 1.0
        self._prev_pump_speed = 1.0

        return self._build_obs_at_reset(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Advance simulation by one 5-second timestep.

        Parameters
        ----------
        action : ndarray of shape (4,)
            [throttle_batch, pump_speed_A, hvac_effort, bess_dispatch]
            All values are expected in the bounds defined by ``action_space``.

        Returns
        -------
        obs, reward, terminated, truncated, info
        """
        action         = np.clip(action.astype(np.float32),
                                 self._ACT_LOW, self._ACT_HIGH)
        throttle_batch = float(action[0])
        pump_speed_A   = float(action[1])
        hvac_effort    = float(action[2])
        bess_dispatch  = float(action[3])   # [-1, 1]

        # Scale BESS dispatch to MW.
        # Positive = discharge (DC injects power back to grid side).
        # Negative = charge    (DC draws more from grid).
        bess_mw = bess_dispatch * getattr(BESSModel, "P_MAX_MW", 50.0)

        # -----------------------------------------------------------------
        # 1. Workload  → IT power by zone
        # -----------------------------------------------------------------
        w = self._workload.step(throttle_batch)

        # -----------------------------------------------------------------
        # 2. BESS
        # -----------------------------------------------------------------
        bess_out = self._bess.step(bess_mw)

        # -----------------------------------------------------------------
        # 3. Thermal  ← zone IT powers (MW)
        # -----------------------------------------------------------------
        # Update ambient temperature: real/synthetic weather OR static scenario value
        if self._weather_driven:
            self._thermal.T_amb = self._weather.temp_c(self._tick)
        # else: T_amb remains as set during reset() (static scenario override)

        p_it_A_mw = (w.p_base_a_kw + w.p_flex_kw) / 1_000.0
        p_it_B_mw =  w.p_base_b_kw               / 1_000.0
        (temp_A, temp_B), (p_cool_A_mw, p_hvac_mw, p_pump_mw) = self._thermal.step(
            p_it_A_mw=p_it_A_mw,
            p_it_B_mw=p_it_B_mw,
            hvac_effort=hvac_effort,
            pump_speed=pump_speed_A,
        )

        # -----------------------------------------------------------------
        # 4. Electrical  (full facility accounting: UPS/PDU/XFMR + PUE)
        # -----------------------------------------------------------------
        util_A = _inverse_rack_util(
            w.p_base_a_kw + w.p_flex_kw, _ZA_N_RACKS, _ZA_P_IDLE, _ZA_P_RANGE, _ZA_ALPHA
        )
        util_B = _inverse_rack_util(
            w.p_base_b_kw, _ZB_N_RACKS, _ZB_P_IDLE, _ZB_P_RANGE, _ZB_ALPHA
        )
        # p_pump_mw is a separate facility electrical load (CDU circulating pump)
        elec = self._elec.step(util_A, util_B, p_cool_A_mw, p_hvac_mw + p_pump_mw)

        # -----------------------------------------------------------------
        # 5. Grid regulation signal
        # -----------------------------------------------------------------
        gs = self._grid.step(committed_mw=self._committed_mw)

        # -----------------------------------------------------------------
        # 6. ΔP tracking
        # -----------------------------------------------------------------
        # Positive regd_signal → grid wants DC to REDUCE net draw.
        # The DC can deliver this by:
        #   (a) Reducing batch load (flex shedding)    → flex_reduction_kw
        #   (b) Discharging BESS (inject energy back)  → bess_actual_kw
        bess_actual_kw     = bess_out["actual_power_mw"] * 1_000.0
        flex_reduction_kw  = (1.0 - throttle_batch) * w.p_flex_nom_kw

        # ΔP_actual = how much the DC has actually reduced its net draw
        delta_p_actual_kw  = flex_reduction_kw + bess_actual_kw

        # ΔP_demanded = committed_mw × regd_signal (signed MW → kW)
        delta_p_demanded_kw = (self._committed_mw * 1_000.0
                               * float(gs["regd_signal"]))

        tracking_err_kw = abs(delta_p_demanded_kw - delta_p_actual_kw)

        # -----------------------------------------------------------------
        # 7. Reward
        # -----------------------------------------------------------------
        alpha      = float(self._rcfg["alpha"])
        beta       = float(self._rcfg["beta"])
        gamma      = float(self._rcfg["gamma_thermal"])
        T_warn_A   = float(self._rcfg["T_warn_A"])
        T_warn_B   = float(self._rcfg["T_warn_B"])
        soc_pen_c  = float(self._rcfg["soc_penalty"])
        norm_kw    = max(self._committed_mw * 1_000.0, 1.0)

        thermal_pen = (max(0.0, temp_A - T_warn_A)
                       + max(0.0, temp_B - T_warn_B))
        soc_pen = soc_pen_c if bess_out["soc_fraction"] < 0.12 else 0.0

        reward = float(
            alpha  * throttle_batch
            - beta  * (tracking_err_kw / norm_kw)
            - gamma * thermal_pen
            - soc_pen
        )

        # -----------------------------------------------------------------
        # 8. Termination / truncation
        # -----------------------------------------------------------------
        T_safe     = self._thermal.T_safe
        terminated = bool(temp_A > T_safe or temp_B > T_safe)

        self._tick += 1
        truncated  = self._tick >= self._episode_ticks

        # -----------------------------------------------------------------
        # 9. Observation
        # -----------------------------------------------------------------
        self._prev_throttle  = throttle_batch
        self._prev_pump_speed = pump_speed_A
        obs = self._build_obs(temp_A, temp_B, bess_out, w, elec, gs)

        info = {
            "tick":                  self._tick,
            "temp_A":                temp_A,
            "temp_B":                temp_B,
            "bess_soc":              bess_out["soc_fraction"],
            "p_facility_mw":         elec["p_facility_mw"],
            "p_total_it_mw":         elec["p_total_it_mw"],
            "pue":                   elec["pue_dynamic"],
            "tracking_err_kw":       tracking_err_kw,
            "delta_p_actual_kw":     delta_p_actual_kw,
            "delta_p_demanded_kw":   delta_p_demanded_kw,
            "flex_reduction_kw":     flex_reduction_kw,
            "bess_actual_kw":        bess_actual_kw,
            "lmp":                   gs["lmp_usd_mwh"],
            "regd_signal":           gs["regd_signal"],
            "reward":                reward,
            "is_spike":              w.is_spike_active,
            "pump_speed_A":          pump_speed_A,
            "p_pump_mw":             p_pump_mw,
            "T_amb":                  self._thermal.T_amb,
            "weather_driven":          self._weather_driven,
            "weather_source":          self._weather.source,
            "thermal_terminated":    terminated,
            "scenario":              self._scenario,
        }
        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        pass   # headless benchmarking environment

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_obs(
        self,
        temp_A: float,
        temp_B: float,
        bess_out: dict,
        w,               # WorkloadState
        elec: dict,
        gs: dict,
    ) -> np.ndarray:
        T_safe = self._thermal.T_safe
        return np.array([
            temp_A / T_safe,
            temp_B / T_safe,
            bess_out["soc_fraction"],
            w.p_base_kw        / _FACILITY_CAP_KW,
            w.p_flex_nom_kw    / _FACILITY_CAP_KW,
            min(elec["p_facility_mw"] * 1_000.0 / _FACILITY_CAP_KW, 2.0),
            float(np.clip(gs["regd_signal"], -1.0, 1.0)),
            min(gs["lmp_usd_mwh"] / 200.0, 1.0),
            float(gs["load_norm"]),
            float(w.is_spike_active),
            self._prev_throttle,
            self._prev_pump_speed,
            min(elec["pue_dynamic"] / 2.5, 2.0),
            self._weather.temp_norm(self._tick),
        ], dtype=np.float32)

    def _build_obs_at_reset(self) -> np.ndarray:
        """Approximated observation before any step has been taken."""
        T_safe = self._thermal.T_safe
        bess_soc = float(self._scfg.get("bess_soc_init", 0.5))
        return np.array([
            self._thermal.temp_A / T_safe,
            self._thermal.temp_B / T_safe,
            bess_soc,
            0.5,   # p_base_norm  (estimate)
            0.3,   # p_flex_nom_norm
            0.8,   # p_facility_norm
            0.0,   # regd_signal
            0.2,   # lmp_norm
            0.5,   # grid_load_norm
            0.0,   # is_spike
            1.0,   # prev_throttle
            1.0,   # prev_pump_speed
            0.6,   # pue_norm
            self._weather.temp_norm(0),  # T_amb_norm at tick 0
        ], dtype=np.float32)
