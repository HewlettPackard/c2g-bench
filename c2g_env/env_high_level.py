"""
Step 2.2 — C2G-MacroEnv  (High-Level / Grid-Manager Environment)
=================================================================
A ``gymnasium.Env`` that wraps :class:`C2GFastEnv` and operates at a
coarser 15-minute (900 s) timescale — 180 five-second sub-steps per action.

This is the decision layer of the hierarchical control stack:

  MacroEnv action (every 15 min)
      → translated into 180 repeated FastEnv actions (every 5 s)
      → each FastEnv step advances all physics engines

The MacroEnv agent commits to grid-regulation capacity and sets a
facility-level power target.  The inner FastEnv executes fine-grained
BESS + DVFS + HVAC control to track that commitment.

Action space  (Box, 2-D, continuous)
-------------------------------------
  [0] commit_norm    ∈ [0, 1]   → committed regulation capacity,
                                   mapped to [0, committed_max_mw].
  [1] bess_target    ∈ [-1, 1]  → average BESS dispatch to hold over 3
                                   sub-steps.  Passed unchanged to FastEnv.

MacroEnv holds the other two FastEnv levers at fixed "safe defaults"
unless overridden by ``inner_action_fn`` (optional callback for research):
  • throttle_batch = 1.0   (do not throttle batch by default)
  • hvac_effort    = 0.7   (moderate cooling)

Observation space  (Box, 16-D)
---------------------------------
Aggregated over the 180 sub-steps to give the grid-manager a stable view:

  [0]  temp_A_mean        Mean Zone A temperature / T_safe
  [1]  temp_B_mean        Mean Zone B temperature / T_safe
  [2]  bess_soc_end       SOC at end of the macro-step         ∈ [0, 1]
  [3]  p_base_mean        Mean p_base_norm
  [4]  p_facility_mean    Mean p_facility_norm
  [5]  regd_mean          Mean |regd_signal|                   ∈ [0, 1]
  [6]  lmp_mean           Mean lmp_norm                        ∈ [0, 1]
  [7]  grid_load_mean     Mean load_norm                       ∈ [0, 1]
  [8]  tracking_err_mean  Mean |ΔP_demanded − ΔP_actual| / norm
  [9]  is_spike_any       1.0 if any sub-step had a GenAI spike
  [10] thermal_headroom_A (T_safe − T_A_max) / T_safe          ∈ [0, 1]
  [11] thermal_headroom_B (T_safe − T_B_max) / T_safe          ∈ [0, 1]
  [12] commit_prev_norm   Previous macro-action [0] committed
  [13] bess_target_prev   Previous macro-action [1] bess target
  [14] freq_dev_mean      Mean normalised frequency deviation   ∈ [-1, 1]
  [15] v_pcc_mean         Mean PCC voltage (per-unit)           ∈ [0, 1.1]
  [16] backlog_norm_mean  Mean batch queue depth / p_flex_max   ∈ [0, 2]

Reward
------
  R_macro = mean of sub-step rewards (from FastEnv)
           + lmp_bonus × mean_lmp × |bess_actual| / P_max   (export revenue)
           − commit_volatility × |commit_now − commit_prev|  (penalise churning)

Usage
-----
  from c2g_env import C2GMacroEnv
  env  = C2GMacroEnv(scenario="scenario_b")
  obs, info = env.reset(seed=0)
  obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml
import gymnasium as gym
from gymnasium import spaces

from c2g_env.env_low_level import C2GFastEnv

_CONFIG_PATH = Path(__file__).parent / "config.yaml"

# Number of FastEnv steps per MacroEnv step (180 × 5 s = 900 s = 15 min)
_SUBSTEPS = 180

# Safe-default inner levers when MacroEnv has full control
_DEFAULT_THROTTLE = 1.0   # do not shed batch
_DEFAULT_PUMP     = 0.7   # moderate pump speed (thermal inertia reserve)
_DEFAULT_HVAC     = 0.7   # moderate cooling


class C2GMacroEnv(gym.Env):
    """
    C2G-Bench High-Level Environment (Grid Manager / Macro Controller).

    Parameters
    ----------
    scenario : str
        Scenario key in ``config.yaml``.
    config_path : str or Path, optional
        Override path to ``config.yaml``.
    committed_max_mw : float, optional
        Maximum regulation capacity the agent may commit.
        Defaults to 2× the scenario's ``committed_mw``.
    lmp_bonus : float, optional
        Revenue multiplier applied to BESS discharge revenue in the macro
        reward.  Default 0.1 (tunable hyper-parameter).
    commit_volatility : float, optional
        Penalty coefficient for large step-to-step changes in the
        committed capacity.  Default 0.05.
    inner_action_fn : callable, optional
        If provided, called as ``fn(macro_obs, macro_action) → low_action``
        to produce a 3-D FastEnv action for each sub-step.  Allows research
        into HRL decompositions.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario: str = "default",
        config_path: str | Path | None = None,
        committed_max_mw: float | None = None,
        lmp_bonus: float = 0.1,
        commit_volatility: float = 0.05,
        inner_action_fn: Callable | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        cfg_path = Path(config_path) if config_path else _CONFIG_PATH
        with open(cfg_path) as fh:
            full_cfg = yaml.safe_load(fh)

        self._gcfg    = full_cfg["global"]
        self._scfg    = full_cfg[scenario]
        self._scenario = scenario

        self._lmp_bonus         = float(lmp_bonus)
        self._commit_volatility = float(commit_volatility)
        self._inner_action_fn   = inner_action_fn

        # Default committed max = 2× scenario committed
        if committed_max_mw is None:
            committed_max_mw = float(self._scfg["committed_mw"]) * 2.0
        self._committed_max_mw = float(committed_max_mw)

        # --- Spaces ------------------------------------------------------
        self.action_space = spaces.Box(
            low  = np.array([0.0, -1.0], dtype=np.float32),
            high = np.array([1.0,  1.0], dtype=np.float32),
            dtype= np.float32,
        )
        self.observation_space = spaces.Box(
            low  = np.full(17, -1.0, dtype=np.float32),
            high = np.full(17,  2.0, dtype=np.float32),
            dtype= np.float32,
        )
        # Override bounds for strictly non-negative features
        _nonneg = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16]
        self.observation_space.low[_nonneg]  = 0.0
        self.observation_space.high[_nonneg] = np.array([
            2.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.1, 2.0
        ], dtype=np.float32)
        # freq_dev_mean can be negative (index 14)
        self.observation_space.low[14]  = -1.0
        self.observation_space.high[14] =  1.0

        # Inner environment
        self._fast_env = C2GFastEnv(
            scenario=scenario, config_path=cfg_path
        )

        # Episode tracking
        self._macro_tick       = 0
        self._prev_commit_norm = 0.5
        self._prev_bess_target = 0.0
        self._episode_macro_ticks = (
            int(self._gcfg["episode_ticks"]) // _SUBSTEPS
        )

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Reset both the outer macro environment and the inner FastEnv.
        """
        super().reset(seed=seed)
        inner_obs, _ = self._fast_env.reset(seed=seed, options=options)

        self._macro_tick        = 0
        self._prev_commit_norm  = 0.5
        self._prev_bess_target  = 0.0

        # Build macro obs from the reset inner obs
        return self._obs_from_reset(inner_obs), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one 15-minute macro step (= 180 × 5-s FastEnv sub-steps).

        Parameters
        ----------
        action : ndarray of shape (2,)
            [commit_norm, bess_target]
        """
        action         = np.clip(action.astype(np.float32),
                                 self.action_space.low,
                                 self.action_space.high)
        commit_norm    = float(action[0])
        bess_target    = float(action[1])

        # Map commitment fraction to MW and update inner env
        committed_mw = commit_norm * self._committed_max_mw
        self._fast_env.committed_mw = committed_mw

        # Accumulators over sub-steps
        sub_rewards    = []
        sub_obs_list   = []
        temp_As, temp_Bs = [], []
        regd_abs, lmps, load_norms = [], [], []
        track_errs    = []
        bess_actuals  = []
        spike_any      = False
        backlog_norms  = []
        terminated     = False
        truncated      = False
        last_info: dict = {}

        for sub in range(_SUBSTEPS):
            if self._inner_action_fn is not None:
                # Research mode: outer policy generates inner action
                inner_obs = sub_obs_list[-1] if sub_obs_list else np.zeros(17)
                low_action = self._inner_action_fn(inner_obs, action)
            else:
                low_action = np.array([
                    _DEFAULT_THROTTLE,
                    _DEFAULT_PUMP,
                    _DEFAULT_HVAC,
                    bess_target,
                ], dtype=np.float32)

            obs, rew, term, trunc, info = self._fast_env.step(low_action)
            sub_rewards.append(rew)
            sub_obs_list.append(obs)

            temp_As.append(info["temp_A"])
            temp_Bs.append(info["temp_B"])
            regd_abs.append(abs(info["regd_signal"]))
            lmps.append(info["lmp"])
            load_norms.append(obs[8])   # grid_load_norm index
            track_errs.append(info["tracking_err_kw"])
            bess_actuals.append(info["bess_actual_kw"])
            spike_any = spike_any or bool(info["is_spike"])
            backlog_norms.append(min(
                info["backlog_kw"] / self._fast_env._workload.p_flex_max_kw, 2.0
            ))
            last_info = info

            if term or trunc:
                terminated = term
                truncated  = trunc
                break

        self._macro_tick += 1
        if self._macro_tick >= self._episode_macro_ticks and not terminated:
            truncated = True

        # -----------------------------------------------------------------
        # Aggregate observation
        # -----------------------------------------------------------------
        T_safe   = self._fast_env._thermal.T_safe
        last_obs = sub_obs_list[-1]
        mean_lmp = float(np.mean(lmps))
        bess_soc_end = last_obs[2]
        bess_disch_mean = max(float(np.mean(bess_actuals)), 0.0)   # kW discharge

        # Collect freq/voltage from sub-step observations (indices 14, 15)
        freq_devs = [float(o[14]) for o in sub_obs_list]
        v_pccs    = [float(o[15]) for o in sub_obs_list]

        obs = np.array([
            np.mean(temp_As) / T_safe,                     # 0
            np.mean(temp_Bs) / T_safe,                     # 1
            bess_soc_end,                                  # 2
            float(np.mean([o[3] for o in sub_obs_list])),  # 3 p_base_mean
            float(np.mean([o[5] for o in sub_obs_list])),  # 4 p_facility_mean
            float(np.mean(regd_abs)),                      # 5 regd_mean
            min(mean_lmp / 200.0, 1.0),                    # 6 lmp_norm
            float(np.mean(load_norms)),                    # 7 grid_load_mean
            float(np.mean(track_errs)) / max(committed_mw * 1_000, 1),  # 8
            float(spike_any),                              # 9
            max(0.0, (T_safe - max(temp_As)) / T_safe),   # 10 headroom_A
            max(0.0, (T_safe - max(temp_Bs)) / T_safe),   # 11 headroom_B
            commit_norm,                                   # 12
            (bess_target + 1.0) / 2.0,                    # 13 normalise to [0,1]
            float(np.mean(freq_devs)),                     # 14 freq_dev_mean
            float(np.mean(v_pccs)),                        # 15 v_pcc_mean
            float(np.mean(backlog_norms)),                 # 16 backlog_norm_mean
        ], dtype=np.float32)

        # -----------------------------------------------------------------
        # Macro reward
        # -----------------------------------------------------------------
        mean_sub_reward  = float(np.mean(sub_rewards))
        lmp_bonus        = (self._lmp_bonus
                            * mean_lmp / 200.0
                            * bess_disch_mean / 50_000.0)   # 50 MW = P_MAX
        commit_churn_pen = (self._commit_volatility
                            * abs(commit_norm - self._prev_commit_norm))

        reward = float(mean_sub_reward + lmp_bonus - commit_churn_pen)

        self._prev_commit_norm = commit_norm
        self._prev_bess_target = bess_target

        info_out = {
            "macro_tick":         self._macro_tick,
            "committed_mw":       committed_mw,
            "mean_sub_reward":    mean_sub_reward,
            "lmp_bonus":          lmp_bonus,
            "commit_churn_pen":   commit_churn_pen,
            "mean_tracking_err":  float(np.mean(track_errs)),
            "mean_lmp":           mean_lmp,
            "spike_any":          spike_any,
            "temp_A_max":         max(temp_As),
            "temp_B_max":         max(temp_Bs),
            "bess_soc_end":       bess_soc_end,
            "backlog_norm_mean":  float(np.mean(backlog_norms)),
            "sub_steps_run":      len(sub_rewards),
            "scenario":           self._scenario,
            "last_inner_info":    last_info,
        }
        return obs, reward, terminated, truncated, info_out

    def render(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _obs_from_reset(self, inner_obs: np.ndarray) -> np.ndarray:
        """Build macro observation from inner env's reset observation."""
        T_safe = self._fast_env._thermal.T_safe
        bess_soc = inner_obs[2]
        return np.array([
            inner_obs[0],   # temp_A_norm
            inner_obs[1],   # temp_B_norm
            bess_soc,
            inner_obs[3],   # p_base_norm
            inner_obs[5],   # p_facility_norm
            abs(inner_obs[6]),  # regd_abs
            inner_obs[7],   # lmp_norm
            inner_obs[8],   # grid_load_norm
            0.0,            # tracking_err_mean (no steps yet)
            0.0,            # is_spike_any
            max(0.0, (T_safe - self._fast_env._thermal.temp_A) / T_safe),
            max(0.0, (T_safe - self._fast_env._thermal.temp_B) / T_safe),
            self._prev_commit_norm,
            (self._prev_bess_target + 1.0) / 2.0,
            0.0,                        # 14 freq_dev_mean (nominal)
            1.0,                        # 15 v_pcc_mean (nominal)
            0.0,                        # 16 backlog_norm_mean (no deferred work at reset)
        ], dtype=np.float32)
