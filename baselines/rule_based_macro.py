"""
baselines/rule_based_macro.py  —  Rule-Based Macro Controller
===============================================================
Deterministic macro-level controller for ``C2GMacroEnv``.

Implements simple commitment and dispatch rules that serve as the
classical baseline for the high-level decision problem.

Policy logic
------------
1. **Commitment sizing** — proportional to grid load stress:
   - High load (grid_load > 0.7): commit 80% of capacity → earn high LMP
   - Medium load (0.4–0.7):       commit 50% of capacity
   - Low load (< 0.4):            commit 20% → conserve BESS

2. **BESS target** — price-responsive:
   - High LMP (> 0.5 norm):  discharge (+0.6) to earn revenue
   - Low LMP (< 0.2 norm):   charge (-0.5) when cheap
   - Otherwise:               idle (0.0)

3. **Safety overrides:**
   - If SOC < 0.15: set BESS target = -0.3 (gentle charge)
   - If SOC > 0.90: set BESS target = +0.3 (gentle discharge)
   - If thermal headroom < 0.1: reduce commitment to 30%

Observation index map (C2GMacroEnv, 16-D)
-------------------------------------------
  [0]  temp_A_mean     [1]  temp_B_mean     [2]  bess_soc_end
  [3]  p_base_mean     [4]  p_facility_mean [5]  regd_mean
  [6]  lmp_mean        [7]  grid_load_mean  [8]  tracking_err_mean
  [9]  is_spike_any    [10] headroom_A      [11] headroom_B
  [12] commit_prev     [13] bess_prev       [14] freq_dev_mean
  [15] v_pcc_mean
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# Observation indices
_I_SOC       = 2
_I_LMP       = 6
_I_LOAD      = 7
_I_HEADROOM_A = 10
_I_HEADROOM_B = 11
_I_FREQ_DEV  = 14
_I_VPCC      = 15


class RuleBasedMacroController:
    """
    Deterministic rule-based controller for ``C2GMacroEnv``.

    Implements the ``predict(obs)`` interface compatible with SB3 so it
    can be used interchangeably with trained agents in the evaluation loop.
    """

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
        soc       = float(obs[_I_SOC])
        lmp_norm  = float(obs[_I_LMP])
        load_norm = float(obs[_I_LOAD])
        hroom_A   = float(obs[_I_HEADROOM_A])
        hroom_B   = float(obs[_I_HEADROOM_B])
        freq_dev  = float(obs[_I_FREQ_DEV])
        v_pcc     = float(obs[_I_VPCC])

        # ── Commitment sizing (proportional to grid load) ────────────
        if load_norm > 0.7:
            commit_norm = 0.80
        elif load_norm > 0.4:
            commit_norm = 0.50
        else:
            commit_norm = 0.20

        # ── BESS target (price-responsive) ───────────────────────────
        if lmp_norm > 0.5:
            bess_target = 0.6    # discharge when price is high
        elif lmp_norm < 0.2:
            bess_target = -0.5   # charge when price is low
        else:
            bess_target = 0.0

        # ── Safety overrides ─────────────────────────────────────────
        # SOC protection
        if soc < 0.15:
            bess_target = -0.3   # gentle charge
        elif soc > 0.90:
            bess_target = 0.3    # gentle discharge

        # Thermal protection: reduce commitment
        min_headroom = min(hroom_A, hroom_B)
        if min_headroom < 0.10:
            commit_norm = min(commit_norm, 0.30)
            bess_target = max(bess_target, 0.0)  # don't charge (adds heat)

        # Frequency support: if under-frequency, increase commitment
        # (committing more → more regulation capacity → helps frequency)
        if freq_dev < -0.3:
            commit_norm = min(1.0, commit_norm + 0.2)

        # Voltage support: if low voltage, reduce facility load commitment
        if v_pcc < 0.96:
            commit_norm = min(commit_norm, 0.40)

        return np.array([commit_norm, bess_target], dtype=np.float32)
