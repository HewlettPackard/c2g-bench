"""
baselines/bang_bang.py  —  Bang-Bang / Hysteresis Controller
============================================================
The simplest reasonable baseline: binary switching with hysteresis bands.

Policy logic
------------
1. **BESS dispatch** = sign(regd_signal) × 0.4 (moderate bang-bang)
   - Binary switching but at ±20 MW instead of ±50 MW to reduce overshoot.
   - SOC guard: disable discharge below 12%, disable charge above 93%.

2. **Throttle** (batch load shedding):
   - 0.7 when regd > 0.05 (grid wants reduced draw → shed batch)
   - 1.0 otherwise (max throughput)

3. **CDU pump** (Zone A):
   - ON  (1.0) when  temp_A_norm > 31/35 ≈ 0.886
   - OFF (0.3) when  temp_A_norm < 29/35 ≈ 0.829
   - Hysteresis holds previous state between thresholds.

4. **HVAC** (Zone B):
   - ON  (1.0) when  temp_B_norm > 31/35 ≈ 0.886
   - OFF (0.0) when  temp_B_norm < 29/35 ≈ 0.829
   - Same hysteresis logic.

The hysteresis bands prevent rapid on/off chattering and represent a
plausible "junior engineer's first controller" — the absolute floor
baseline for the benchmark.

Observation index map (C2GFastEnv, 17-D)
-----------------------------------------
  [0]  temp_A_norm     [1]  temp_B_norm     [2]  bess_soc
  [6]  regd_signal

Usage
-----
  from baselines.bang_bang import BangBangController
  ctrl = BangBangController()
  obs, _ = env.reset(seed=0)
  action, _ = ctrl.predict(obs)
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# Observation indices
_I_TEMP_A = 0
_I_TEMP_B = 1
_I_SOC    = 2
_I_REGD   = 6

# Hysteresis thresholds (normalised: T / T_safe where T_safe = 35°C)
_PUMP_ON  = 31.0 / 35.0   # ≈ 0.886
_PUMP_OFF = 29.0 / 35.0   # ≈ 0.829
_HVAC_ON  = 31.0 / 35.0
_HVAC_OFF = 29.0 / 35.0

# SOC guard bands
_SOC_DISCHARGE_MIN = 0.12
_SOC_CHARGE_MAX    = 0.93


class BangBangController:
    """
    Hysteresis bang-bang controller for ``C2GFastEnv``.

    Implements the SB3 ``predict(obs)`` interface.
    """

    def __init__(self) -> None:
        self._pump_on: bool = False
        self._hvac_on: bool = False

    def predict(
        self,
        obs: NDArray[np.float32],
        state=None,
        episode_start=None,
        deterministic: bool = True,
    ) -> tuple[NDArray[np.float32], None]:
        single = obs.ndim == 1
        if single:
            obs = obs[np.newaxis, :]
        actions = np.array([self._action_for(o) for o in obs],
                           dtype=np.float32)
        return (actions[0] if single else actions), None

    def _action_for(self, obs: NDArray[np.float32]) -> NDArray[np.float32]:
        temp_A_n = float(obs[_I_TEMP_A])
        temp_B_n = float(obs[_I_TEMP_B])
        soc      = float(obs[_I_SOC])
        regd     = float(obs[_I_REGD])

        # ── Throttle: bang-bang based on regd direction ────────────────
        # regd > 0 → grid wants reduced draw → shed some batch load
        # regd < 0 → grid wants increased draw → max throughput
        _THROTTLE_LOW = 0.5
        if regd > 0.05:
            throttle = _THROTTLE_LOW
        else:
            throttle = 1.0

        # ── Pump: hysteresis on Zone A temperature ────────────────────
        if temp_A_n >= _PUMP_ON:
            self._pump_on = True
        elif temp_A_n <= _PUMP_OFF:
            self._pump_on = False
        pump_speed = 1.0 if self._pump_on else 0.3

        # ── HVAC: hysteresis on Zone B temperature ────────────────────
        if temp_B_n >= _HVAC_ON:
            self._hvac_on = True
        elif temp_B_n <= _HVAC_OFF:
            self._hvac_on = False
        hvac_effort = 1.0 if self._hvac_on else 0.0

        # ── BESS: bang-bang with moderate magnitude ────────────────────
        # Binary switching (true bang-bang) but at ±0.4 instead of ±1.0
        # to avoid massive overshoot. 0.4 × 50 MW = 20 MW, closer to
        # typical committed_mw (10-20 MW) than full 50 MW blast.
        _BESS_MAG = 0.5
        if abs(regd) < 0.05:
            bess_dispatch = 0.0
        elif regd > 0:
            # Grid wants DC to reduce draw → discharge
            bess_dispatch = _BESS_MAG if soc > _SOC_DISCHARGE_MIN else 0.0
        else:
            # Grid wants DC to increase draw → charge
            bess_dispatch = -_BESS_MAG if soc < _SOC_CHARGE_MAX else 0.0

        return np.array([throttle, pump_speed, hvac_effort, bess_dispatch],
                        dtype=np.float32)
