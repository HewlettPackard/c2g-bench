"""
Step 2.2 — C2G-MacroEnv  (High-Level / Grid-Manager Environment)
=================================================================
A ``gymnasium.Env`` that wraps :class:`C2GFastEnv` and operates at a
coarser 15-minute (900 s) timescale — 180 five-second sub-steps per action.

This is the decision layer of the hierarchical control stack:

  MacroEnv action (every 15 min)
      → translated into 180 repeated FastEnv actions (every 5 s)
      → each FastEnv step advances all physics engines

The MacroEnv agent participates in a 3-phase market handshake each macro
step:
  1. Grid posts offer  — RMCP and regulation need
  2. DC bids           — agent offers MW capacity at a price
  3. Grid clears       — probabilistic acceptance (sigmoid × quantity)

If the bid is accepted the DC commits the accepted MW at the clearing
price.  If rejected it falls back to a standing DR baseline contract.

Action space  (Box, 2-D, continuous)
-------------------------------------
  [0] bid_mw_norm    ∈ [0, 1]   → MW to offer to the grid,
                                   mapped to [0, committed_max_mw].
  [1] bid_price_norm ∈ [0, 1]   → asking price,
                                   mapped to [0, 2 × rmcp_max].

MacroEnv holds all FastEnv levers at fixed "safe defaults"
unless overridden by ``inner_action_fn`` (optional callback for research):
  • throttle_batch = 1.0   (do not throttle batch by default)
  • hvac_effort    = 0.7   (moderate cooling)
  • bess_dispatch  = 0.0   (neutral)

Observation space  (Box, 19-D)
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
  [12] bid_mw_prev_norm   Previous macro-action [0] bid MW
  [13] bid_price_prev_norm Previous macro-action [1] bid price
  [14] freq_dev_mean      Mean normalised frequency deviation   ∈ [-1, 1]
  [15] v_pcc_mean         Mean PCC voltage (per-unit)           ∈ [0, 1.1]
  [16] backlog_norm_mean  Mean batch queue depth / p_flex_max   ∈ [0, 2]
  [17] rmcp_norm          Grid's posted RMCP / rmcp_max         ∈ [0, 5]
  [18] reg_need_norm      Grid's residual need / committed_max  ∈ [0, 5]

Reward
------
  R_macro = λ_rev  × regulation_revenue / 1000
           + mean_sub_reward
           − λ_elec × electricity_cost / 1000
           − λ_churn × |bid_mw_now − bid_mw_prev|

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
_DEFAULT_BESS     = 0.0   # neutral BESS (low-level agent controls dispatch)


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
        Maximum regulation capacity the agent may bid.
        Defaults to 2× the scenario's ``committed_mw``.
    inner_action_fn : callable, optional
        If provided, called as ``fn(macro_obs, macro_action) → low_action``
        to produce a 4-D FastEnv action for each sub-step.  Allows research
        into HRL decompositions.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario: str = "default",
        config_path: str | Path | None = None,
        inner_action_fn: Callable | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        cfg_path = Path(config_path) if config_path else _CONFIG_PATH
        with open(cfg_path, encoding="utf-8") as fh:
            full_cfg = yaml.safe_load(fh)

        self._gcfg    = full_cfg["global"]
        self._scfg    = full_cfg[scenario]
        self._scenario = scenario

        # Handshake config
        hs_cfg = full_cfg.get("handshake", {})
        self._rmcp_max     = float(hs_cfg.get("rmcp_max", 100.0))
        self._lambda_rev   = float(hs_cfg.get("lambda_rev", 1.0))
        self._lambda_elec  = float(hs_cfg.get("lambda_elec", 0.5))
        self._lambda_churn = float(hs_cfg.get("lambda_churn", 0.05))

        # DR baseline (standing contract for rejected bids)
        self._dr_baseline_mw = float(self._scfg.get("dr_baseline_mw", 5.0))
        self._dr_rate_usd_mw = float(self._scfg.get("dr_rate_usd_mw", 5.0))

        self._inner_action_fn = inner_action_fn

        committed_max_mw = float(self._scfg["committed_mw_max"])
        self._committed_max_mw = float(committed_max_mw)

        # --- Spaces ------------------------------------------------------
        # [0] bid_mw_norm, [1] bid_price_norm
        self.action_space = spaces.Box(
            low  = np.array([0.0, 0.0], dtype=np.float32),
            high = np.array([1.0, 1.0], dtype=np.float32),
            dtype= np.float32,
        )
        # 19-D observation (17 original + 2 market signals)
        self.observation_space = spaces.Box(
            low  = np.full(19, -1.0, dtype=np.float32),
            high = np.full(19,  5.0, dtype=np.float32),
            dtype= np.float32,
        )
        # Override bounds for strictly non-negative features
        _nonneg = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18]
        self.observation_space.low[_nonneg] = 0.0
        self.observation_space.high[:17] = np.array([
            2.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.1, 2.0,
        ], dtype=np.float32)
        # freq_dev_mean can be negative (index 14)
        self.observation_space.low[14] = -1.0
        # bess_target_prev at [13] normalised to [0,1]
        self.observation_space.low[13] = 0.0
        # market signals [17, 18] bounded [0, 5]
        self.observation_space.high[17] = 5.0
        self.observation_space.high[18] = 5.0

        # Inner environment
        self._fast_env = C2GFastEnv(
            scenario=scenario, config_path=cfg_path
        )

        # Episode tracking
        self._macro_tick        = 0
        self._prev_bid_mw_norm  = 0.5
        self._prev_bid_price_norm = 0.5
        self._last_rmcp_norm    = 0.0
        self._last_reg_need_norm = 0.0
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

        self._macro_tick         = 0
        self._prev_bid_mw_norm   = 0.5
        self._prev_bid_price_norm = 0.5
        self._last_rmcp_norm     = 0.0
        self._last_reg_need_norm = 0.0

        # Build macro obs from the reset inner obs
        return self._obs_from_reset(inner_obs), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one 15-minute macro step with 3-phase market handshake.

        Parameters
        ----------
        action : ndarray of shape (2,)
            [bid_mw_norm, bid_price_norm]
        """
        action          = np.clip(action.astype(np.float32),
                                  self.action_space.low,
                                  self.action_space.high)
        bid_mw_norm     = float(action[0])
        bid_price_norm  = float(action[1])

        bid_mw    = bid_mw_norm * self._committed_max_mw
        bid_price = bid_price_norm * 2.0 * self._rmcp_max

        # =================================================================
        # Phase 1: Grid posts offer (RMCP + regulation need)
        # =================================================================
        grid = self._fast_env._grid
        offer = grid.step_rmcp()
        rmcp_norm     = offer["rmcp_usd"] / self._rmcp_max
        reg_need_norm = offer["residual_mw"] / max(self._committed_max_mw, 1.0)
        self._last_rmcp_norm     = rmcp_norm
        self._last_reg_need_norm = reg_need_norm

        # =================================================================
        # Phase 2: DC bids (agent action already decoded above)
        # =================================================================

        # =================================================================
        # Phase 3: Grid clears bid
        # =================================================================
        result = grid.clear_bid(bid_price, bid_mw)

        if result["accepted"]:
            committed_mw = result["accepted_mw"]
            rate = result["clearing_price"]
        else:
            committed_mw = self._dr_baseline_mw
            rate = self._dr_rate_usd_mw

        self._fast_env.committed_mw = committed_mw

        # =================================================================
        # Run 180 sub-steps
        # =================================================================
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
                inner_obs = sub_obs_list[-1] if sub_obs_list else np.zeros(18)
                low_action = self._inner_action_fn(inner_obs, action)
            else:
                low_action = np.array([
                    _DEFAULT_THROTTLE,
                    _DEFAULT_PUMP,
                    _DEFAULT_HVAC,
                    _DEFAULT_BESS,
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
        # Aggregate observation (19-D)
        # -----------------------------------------------------------------
        T_safe   = self._fast_env._thermal.T_safe
        last_obs = sub_obs_list[-1]
        mean_lmp = float(np.mean(lmps))
        bess_soc_end = last_obs[2]

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
            bid_mw_norm,                                   # 12 bid_mw_prev
            bid_price_norm,                                # 13 bid_price_prev
            float(np.mean(freq_devs)),                     # 14 freq_dev_mean
            float(np.mean(v_pccs)),                        # 15 v_pcc_mean
            float(np.mean(backlog_norms)),                 # 16 backlog_norm_mean
            float(np.clip(rmcp_norm, 0.0, 5.0)),          # 17 rmcp_norm
            float(np.clip(reg_need_norm, 0.0, 5.0)),      # 18 reg_need_norm
        ], dtype=np.float32)

        # -----------------------------------------------------------------
        # Macro reward (market handshake)
        # -----------------------------------------------------------------
        hours = _SUBSTEPS * self._fast_env._dt / 3600.0  # 0.25 h for 15-min

        # Tracking performance score [0, 1]
        tracking_err_norm = float(np.mean(track_errs)) / max(committed_mw * 1_000, 1)
        perf_score = max(0.0, 1.0 - tracking_err_norm)

        # Regulation revenue: rate × MW × performance × hours
        regulation_revenue = rate * committed_mw * perf_score * hours

        # Electricity cost (always runs, prevents "never bid" exploit)
        mean_p_facility_norm = float(np.mean([o[5] for o in sub_obs_list]))
        electricity_cost = mean_lmp * mean_p_facility_norm * hours

        mean_sub_reward  = float(np.mean(sub_rewards))
        commit_churn_pen = abs(bid_mw_norm - self._prev_bid_mw_norm)

        r_regulation = self._lambda_rev  *  regulation_revenue / 1000.0
        r_sub        = mean_sub_reward
        r_elec       = -self._lambda_elec * electricity_cost   / 1000.0
        r_churn      = -self._lambda_churn * commit_churn_pen
        reward = float(r_regulation + r_sub + r_elec + r_churn)

        self._prev_bid_mw_norm = bid_mw_norm
        self._prev_bid_price_norm = bid_price_norm

        info_out = {
            "macro_tick":            self._macro_tick,
            "committed_mw":          committed_mw,
            "bid_mw":                bid_mw,
            "bid_price":             bid_price,
            "bid_accepted":          result["accepted"],
            "accept_prob":           result["accept_prob"],
            "clearing_price":        result["clearing_price"],
            "regulation_revenue":    regulation_revenue,
            "electricity_cost":      electricity_cost,
            "perf_score":            perf_score,
            "rate":                  rate,
            "mean_sub_reward":       mean_sub_reward,
            "commit_churn_pen":      commit_churn_pen,
            "mean_tracking_err":     float(np.mean(track_errs)),
            "mean_lmp":              mean_lmp,
            "rmcp_usd":             offer["rmcp_usd"],
            "residual_mw":          offer["residual_mw"],
            "spike_any":             spike_any,
            "temp_A_max":            max(temp_As),
            "temp_B_max":            max(temp_Bs),
            "bess_soc_end":          bess_soc_end,
            "backlog_norm_mean":     float(np.mean(backlog_norms)),
            "sub_steps_run":         len(sub_rewards),
            "scenario":              self._scenario,
            "last_inner_info":       last_info,
            "reward_regulation":     r_regulation,
            "reward_sub":            r_sub,
            "reward_elec":           r_elec,
            "reward_churn":          r_churn,
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
            self._prev_bid_mw_norm,
            self._prev_bid_price_norm,
            0.0,                        # 14 freq_dev_mean (nominal)
            1.0,                        # 15 v_pcc_mean (nominal)
            0.0,                        # 16 backlog_norm_mean
            0.0,                        # 17 rmcp_norm (no market signal yet)
            0.0,                        # 18 reg_need_norm (no market signal yet)
        ], dtype=np.float32)
