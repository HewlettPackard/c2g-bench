"""
baselines/metrics_callback.py  —  C2G Metrics Logger Callback
==============================================================
Tracks all environment-level metrics from the info dicts produced by
C2GFastEnv at every step, then logs episode aggregates to:
  - TensorBoard (via SB3's logger)
  - A rolling console summary every `print_freq` episodes
  - An optional CSV file (one row per episode)

Metrics tracked per episode
---------------------------
  ep/mean_reward          mean step reward
  ep/total_reward         sum of step rewards

  thermal/mean_temp_A     mean zone-A temperature (°C)
  thermal/mean_temp_B     mean zone-B temperature (°C)
  thermal/max_temp_A      max  zone-A temperature (°C)
  thermal/max_temp_B      max  zone-B temperature (°C)
  thermal/viol_rate       fraction of ticks with temp_A or temp_B > T_WARN
  thermal/terminated      1 if thermal kill triggered, else 0

  bess/mean_soc           mean BESS state of charge (0–1)
  bess/min_soc            minimum SOC reached
  bess/mean_dispatch_kw   mean absolute BESS dispatch (kW)

  grid/tracking_rmse_kw   RMSE of (demanded − actual) regulation (kW)
  grid/mean_lmp           mean LMP (USD/MWh)
  grid/spike_fraction     fraction of ticks during a workload spike

  facility/mean_pue       mean Power Usage Effectiveness
  facility/mean_p_facility_mw  mean total facility power (MW)
  facility/mean_flex_reduction_kw  mean load-flex reduction applied (kW)

  ep/length               ticks in episode (288 = full; < 288 = early term)
  ep/survived             1 if full episode completed, else 0

Metrics tracked per step (C2GTransitionLoggerCallback)
State / Observation / Action Space
-----------------------------------

Observations (s_i / o_i, 17-D normalised):
  s_0 / o_0               temp_A_norm — Zone A (liquid-cooled GPU) temperature / T_safe
  s_1 / o_1               temp_B_norm — Zone B (air-cooled CPU) temperature / T_safe
  s_2 / o_2               bess_soc — Battery state of charge [0, 1]
  s_3 / o_3               p_base_norm — Rigid IT load fraction (GenAI + DLRM)
  s_4 / o_4               p_flex_nom_norm — Schedulable batch capacity at full throttle
  s_5 / o_5               p_facility_norm — Total facility power draw
  s_6 / o_6               regd_signal — Grid RegD regulation signal [-1, 1]
  s_7 / o_7               lmp_norm — Locational marginal price (normalised)
  s_8 / o_8               grid_load_norm — Regional grid load stress indicator
  s_9 / o_9               is_spike — GenAI serving spike flag {0, 1}
  s_10 / o_10             prev_throttle — Previous DVFS throttle action (action memory)
  s_11 / o_11             prev_pump_speed — Previous CDU pump speed action (action memory)
  s_12 / o_12             pue_norm — Power Usage Effectiveness / 2
  s_13 / o_13             T_amb_norm — Ambient temperature / 50
  s_14 / o_14             freq_dev_norm — Grid frequency deviation / 0.5 Hz
  s_15 / o_15             v_pcc_pu — PCC voltage in per-unit (Thévenin model)
  s_16 / o_16             backlog_norm — Deferred batch FIFO queue depth / p_flex_max

Actions (a_i, 4-D continuous):
  a_0                     throttle_batch — DVFS throttle [0=full speed, 1=fully throttled]
  a_1                     pump_speed_A — CDU pump speed Zone A [0, 1]
  a_2                     hvac_effort — Zone B HVAC fan effort [0, 1]
  a_3                     bess_dispatch — BESS dispatch [−1=full charge, +1=full discharge]
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

T_WARN = 33.0   # °C — thermal warning threshold (mirrors config.yaml)

STATE_COLUMNS = [
    "s_temp_A_norm",
    "s_temp_B_norm",
    "s_bess_soc",
    "s_p_base_norm",
    "s_p_flex_nom_norm",
    "s_p_facility_norm",
    "s_regd_signal",
    "s_lmp_norm",
    "s_grid_load_norm",
    "s_is_spike",
    "s_prev_throttle",
    "s_prev_pump_speed",
    "s_pue_norm",
    "s_T_amb_norm",
    "s_freq_dev_norm",
    "s_v_pcc_pu",
    "s_backlog_norm",
]

# Macro-level state: 19-D (indices 0-16 as above, plus 2 market signals)
STATE_COLUMNS_MACRO = STATE_COLUMNS + [
    "s_rmcp_norm",      # obs[17]: RMCP ÷ rmcp_max
    "s_reg_need_norm",  # obs[18]: reg need ÷ cap_max
]

REWARD_COLUMNS = [
    "total_reward",
    "r_throughput",
    "r_tracking",
    "r_thermal",
    "r_soc",
    "r_freq",
    "r_volt",
    "r_backlog",
]

# Macro-level reward components
REWARD_COLUMNS_MACRO = [
    "total_reward",
    "r_regulation",
    "r_sub",
    "r_elec",
    "r_churn",
]

ACTION_NAMES = [
    "throttle_batch",
    "pump_speed_A",
    "hvac_effort",
    "bess_dispatch",
]

# Macro-level actions (2-D)
ACTION_NAMES_MACRO = [
    "commit_norm",
    "bid_price",
]

ACTION_SHORT_NAMES = {
    "throttle_batch": "THROTTLE",
    "pump_speed_A": "PUMP",
    "hvac_effort": "HVAC",
    "bess_dispatch": "BESS",
}

ACTION_SHORT_NAMES_MACRO = {
    "commit_norm": "COMMIT",
    "bid_price": "PRICE",
}

VALID_ACTIONS = tuple(ACTION_SHORT_NAMES.keys())
STATE_NAMES = [col.removeprefix("s_") for col in STATE_COLUMNS]
STATE_NAMES_MACRO = [col.removeprefix("s_") for col in STATE_COLUMNS_MACRO]


def _format_action_value(value: float) -> str:
    formatted = f"{value:.3f}".rstrip("0").rstrip(".")
    return formatted if formatted else "0"


def _sorted_ablation_actions(
    unavailable_actions: tuple[str, ...],
    fixed_action_values: dict[str, float],
) -> list[str]:
    # Sort by short action tag so suffixes are stable and human-readable
    # (for example BESS before THROTTLE).
    action_names = set(unavailable_actions) | set(fixed_action_values.keys())
    return sorted(
        action_names,
        key=lambda name: ACTION_SHORT_NAMES.get(name, name.upper()),
    )


def build_ablation_suffix(
    unavailable_actions: tuple[str, ...] | None,
    fixed_action_values: dict[str, float] | None,
) -> str:
    unavailable = tuple(unavailable_actions or ())
    fixed = dict(fixed_action_values or {})

    tags: list[str] = []
    for action_name in _sorted_ablation_actions(unavailable, fixed):
        short = ACTION_SHORT_NAMES.get(action_name, action_name.upper())
        if action_name in unavailable:
            tags.append(f"{short}_disabled")
        if action_name in fixed:
            value = _format_action_value(float(fixed[action_name]))
            tags.append(f"{short}_{value}")

    return f"__{'_'.join(tags)}" if tags else ""


class C2GMetricsCallback(BaseCallback):
    """
    SB3 callback that aggregates per-step info dicts into episode metrics
    and writes them to TensorBoard + optional CSV.

    Parameters
    ----------
    print_freq : int
        Log a console summary every N *episodes*.
    csv_path : str | Path | None
        If given, append one CSV row per episode to this file.
    verbose : int
        0 = silent (TensorBoard only); 1 = console summaries too.
    """

    def __init__(
        self,
        print_freq: int = 20,
        csv_path: str | Path | None = None,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self._print_freq = print_freq
        self._csv_path   = Path(csv_path) if csv_path else None
        self._csv_writer : csv.DictWriter | None = None
        self._csv_file   = None

        # Per-step accumulators (one list per parallel env, keyed by env idx)
        self._reset_buffers()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _on_training_start(self) -> None:
        if self._csv_path:
            self._csv_path.parent.mkdir(parents=True, exist_ok=True)
            write_header = not self._csv_path.exists()
            self._csv_file   = open(self._csv_path, "a", newline="")  # noqa: SIM115
            self._csv_writer = csv.DictWriter(
                self._csv_file, fieldnames=self._csv_columns()
            )
            if write_header:
                self._csv_writer.writeheader()

    def _on_training_end(self) -> None:
        if self._csv_file:
            self._csv_file.flush()
            self._csv_file.close()

    # ------------------------------------------------------------------
    # Step hook
    # ------------------------------------------------------------------

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [False] * len(infos))

        for env_idx, (info, done) in enumerate(zip(infos, dones)):
            if env_idx not in self._bufs:
                self._reset_buf(env_idx)
            buf = self._bufs[env_idx]

            # Accumulate step-level quantities
            buf["rewards"].append(float(self.locals["rewards"][env_idx]))
            for key in ("temp_A", "temp_B", "bess_soc", "pue", "lmp",
                        "tracking_err_kw", "delta_p_actual_kw",
                        "delta_p_demanded_kw", "flex_reduction_kw",
                        "bess_actual_kw", "p_facility_mw"):
                if key in info:
                    buf[key].append(float(info[key]))

            buf["spike"].append(1.0 if info.get("is_spike", False) else 0.0)
            buf["thermal_viol"].append(
                1.0 if (info.get("temp_A", 0) > T_WARN or
                         info.get("temp_B", 0) > T_WARN) else 0.0
            )
            buf["tick"] = info.get("tick", 0)
            if info.get("thermal_fault", False):
                buf["terminated"] = 1

            # Episode ended → log aggregates
            if done:
                self._log_episode(env_idx, info)
                self._reset_buf(env_idx)

        return True

    # ------------------------------------------------------------------
    # Episode logging
    # ------------------------------------------------------------------

    def _log_episode(self, env_idx: int, last_info: dict[str, Any]) -> None:
        buf  = self._bufs[env_idx]
        self._ep_count += 1
        n    = self.num_timesteps

        def _mean(lst):
            return float(np.mean(lst)) if lst else 0.0

        def _max(lst):
            return float(np.max(lst)) if lst else 0.0

        def _rmse(lst):
            return float(np.sqrt(np.mean(np.square(lst)))) if lst else 0.0

        ep_len      = buf["tick"]
        survived    = 1.0 if ep_len >= 287 else 0.0   # 288 ticks = full ep
        total_r     = float(np.sum(buf["rewards"]))
        mean_r      = _mean(buf["rewards"])

        metrics: dict[str, float] = {
            # episode
            "ep/mean_reward"            : mean_r,
            "ep/total_reward"           : total_r,
            "ep/length"                 : float(ep_len),
            "ep/survived"               : survived,
            # thermal
            "thermal/mean_temp_A"       : _mean(buf["temp_A"]),
            "thermal/mean_temp_B"       : _mean(buf["temp_B"]),
            "thermal/max_temp_A"        : _max(buf["temp_A"]),
            "thermal/max_temp_B"        : _max(buf["temp_B"]),
            "thermal/viol_rate"         : _mean(buf["thermal_viol"]),
            "thermal/terminated"        : float(buf["terminated"]),
            # BESS
            "bess/mean_soc"             : _mean(buf["bess_soc"]),
            "bess/min_soc"              : float(np.min(buf["bess_soc"])) if buf["bess_soc"] else 0.0,
            "bess/mean_dispatch_kw"     : _mean([abs(v) for v in buf["bess_actual_kw"]]),
            # grid
            "grid/tracking_rmse_kw"     : _rmse(buf["tracking_err_kw"]),
            "grid/mean_lmp"             : _mean(buf["lmp"]),
            "grid/spike_fraction"       : _mean(buf["spike"]),
            # facility
            "facility/mean_pue"         : _mean(buf["pue"]),
            "facility/mean_p_facility_mw": _mean(buf["p_facility_mw"]),
            "facility/mean_flex_reduction_kw": _mean(buf["flex_reduction_kw"]),
        }

        # Write to TensorBoard via SB3 logger
        for tag, val in metrics.items():
            self.logger.record(tag, val)
        self.logger.dump(step=n)

        # CSV row
        if self._csv_writer:
            row = {"timestep": n, "episode": self._ep_count,
                   "env_idx": env_idx, **metrics}
            self._csv_writer.writerow(row)
            self._csv_file.flush()

        # Console summary
        if self.verbose >= 1 and self._ep_count % self._print_freq == 0:
            print(
                f"  ep {self._ep_count:5d} | steps {n:8d} | "
                f"r={mean_r:8.2f} | "
                f"T_A={_mean(buf['temp_A']):5.1f}°C "
                f"T_B={_mean(buf['temp_B']):5.1f}°C | "
                f"viol={_mean(buf['thermal_viol']):.3f} | "
                f"SOC={_mean(buf['bess_soc']):.2f} | "
                f"RMSE={_rmse(buf['tracking_err_kw']):7.0f}kW | "
                f"PUE={_mean(buf['pue']):.3f}"
            )

    # ------------------------------------------------------------------
    # Buffer helpers
    # ------------------------------------------------------------------

    def _reset_buffers(self) -> None:
        self._ep_count = 0
        self._bufs: dict[int, dict] = {}
        # Initialised lazily on first step; we don't know n_envs here yet

    def _reset_buf(self, env_idx: int) -> None:
        self._bufs[env_idx] = {
            "rewards": [], "temp_A": [], "temp_B": [], "bess_soc": [],
            "pue": [], "lmp": [], "tracking_err_kw": [], "delta_p_actual_kw": [],
            "delta_p_demanded_kw": [], "flex_reduction_kw": [],
            "bess_actual_kw": [], "p_facility_mw": [],
            "spike": [], "thermal_viol": [],
            "tick": 0, "terminated": 0,
        }

    def __getitem__(self, env_idx: int):
        if env_idx not in self._bufs:
            self._reset_buf(env_idx)
        return self._bufs[env_idx]

    @property
    def _bufs(self) -> dict:
        if not hasattr(self, "_buf_store"):
            self._buf_store: dict[int, dict] = {}
        return self._buf_store

    @_bufs.setter
    def _bufs(self, val: dict) -> None:
        self._buf_store = val

    def _csv_columns(self) -> list[str]:
        return [
            "timestep", "episode", "env_idx",
            "ep/mean_reward", "ep/total_reward", "ep/length", "ep/survived",
            "thermal/mean_temp_A", "thermal/mean_temp_B",
            "thermal/max_temp_A", "thermal/max_temp_B",
            "thermal/viol_rate", "thermal/terminated",
            "bess/mean_soc", "bess/min_soc", "bess/mean_dispatch_kw",
            "grid/tracking_rmse_kw", "grid/mean_lmp", "grid/spike_fraction",
            "facility/mean_pue", "facility/mean_p_facility_mw",
            "facility/mean_flex_reduction_kw",
        ]


class C2GTransitionLoggerCallback(BaseCallback):
    """
    Logs step-wise transitions (state, action, observation, reward) to CSV.

    Supports two modes:
      1) SB3 callback mode via ``_on_step``.
      2) Manual mode via ``record_transition`` for evaluation loops.
    """

    # Semantic lookup tables for state, observation, and action indices
    STATE_NAMES = STATE_NAMES
    STATE_NAMES_MACRO = STATE_NAMES_MACRO
    ACTION_NAMES = ACTION_NAMES
    ACTION_NAMES_MACRO = ACTION_NAMES_MACRO
    ACTION_SHORT_NAMES = ACTION_SHORT_NAMES
    ACTION_SHORT_NAMES_MACRO = ACTION_SHORT_NAMES_MACRO

    def __init__(
        self,
        output_dir: Path | str = "runs",
        algorithm_name: str | None = None,
        scenario_name: str | None = None,
        agent_type: str = "hardware",
        episode_number: int | None = None,
        unavailable_actions: tuple[str, ...] | None = None,
        fixed_action_values: dict[str, float] | None = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self._algorithm_name = algorithm_name
        self._scenario_name = scenario_name
        self._agent_type = agent_type
        self._episode_number = episode_number
        self._unavailable_actions = tuple(unavailable_actions or ())
        self._fixed_action_values = dict(fixed_action_values or {})
        self._active = True

        self._csv_file: Any | None = None
        self._csv_writer: csv.DictWriter | None = None

        self._n_state: int | None = None
        self._n_obs: int | None = None
        self._n_act: int | None = None
        self._reward_component_keys: list[str] = []

        # Agent-aware schemas for transition logging.
        if agent_type is None:
            raise ValueError("agent_type must be either 'hardware' or 'macro'.")
        agent_type = agent_type.lower()
        if agent_type not in ["macro", "hardware"]:
            raise ValueError(
                f"Invalid agent_type '{agent_type}'. Expected one of ['macro', 'hardware']."
            )
        self._state_names = self.STATE_NAMES_MACRO if agent_type == "macro" else self.STATE_NAMES
        self._action_names = self.ACTION_NAMES_MACRO if agent_type == "macro" else self.ACTION_NAMES

        self._global_step = 0

        # Transition logs are only written under <project_root>/runs.
        project_root_runs = Path(__file__).resolve().parent.parent / output_dir
        if not project_root_runs.exists() or not project_root_runs.is_dir():
            self._output_dir = project_root_runs
            self._active = False
            if self.verbose >= 1:
                print(f"[transition_logger] Skipping logging: missing {project_root_runs}")
            return

        if algorithm_name is not None and scenario_name is not None and agent_type is not None:
            self._output_dir = project_root_runs / f"{algorithm_name}_{scenario_name}_{agent_type}"
        else:
            self._output_dir = project_root_runs
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def _build_ablation_suffix(self) -> str:
        return build_ablation_suffix(self._unavailable_actions, self._fixed_action_values)

    @staticmethod
    def _safe_name(value: str | None, default: str) -> str:
        if not value:
            return default
        return str(value).replace(" ", "-")

    def _ensure_writer(
        self,
        n_state: int,
        n_act: int,
        n_obs: int,
        reward_components: dict[str, float] | None,
    ) -> None:
        if self._csv_writer is not None:
            return

        self._n_state = n_state
        self._n_act = n_act
        self._n_obs = n_obs

        algo_tag = self._safe_name(self._algorithm_name, "algo")
        scenario_tag = self._safe_name(self._scenario_name, "scenario")
        ablation_suffix = self._build_ablation_suffix()
        if self._episode_number is not None:
            csv_name = f"episode{self._episode_number}_{ablation_suffix}.csv"
        else:
            csv_name = f"episode_{ablation_suffix}.csv"
        csv_path = self._output_dir / csv_name
        if csv_path.exists():
            csv_path.unlink()
        write_header = True
        self._csv_file = open(csv_path, "w", newline="")

        self._reward_component_keys = sorted((reward_components or {}).keys())

        cols = ["timestep"]
        
        # Map state indices to semantic names
        for i in range(min(n_state, len(self._state_names))):
            cols.append(f"s_{self._state_names[i]}")
        # Fall back to generic names if more states than documented
        for i in range(len(self._state_names), n_state):
            cols.append(f"s_{i}")
        
        # Map action indices to semantic names
        for i in range(min(n_act, len(self._action_names))):
            cols.append(f"a_{self._action_names[i]}")
        # Fall back to generic names if more actions than documented
        for i in range(len(self._action_names), n_act):
            cols.append(f"a_{i}")
        
        # Map observation indices to semantic names
        for i in range(min(n_obs, len(self._state_names))):
            cols.append(f"o_{self._state_names[i]}")
        # Fall back to generic names if more observations than documented
        for i in range(len(self._state_names), n_obs):
            cols.append(f"o_{i}")
        
        cols += ["r"]
        cols += [f"r_{k}" for k in self._reward_component_keys]

        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=cols)
        if write_header:
            self._csv_writer.writeheader()

    def record_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        observation: np.ndarray,
        reward: float,
        done: bool,
        reward_components: dict[str, float] | None = None,
    ) -> None:
        """Manually log one transition tuple for evaluation loops."""
        if not self._active:
            return

        if reward_components is not None:
            reward_components = {
                k.removeprefix("reward_"): v for k, v in reward_components.items()
            }

        s = np.asarray(state, dtype=np.float32).reshape(-1)
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        o = np.asarray(observation, dtype=np.float32).reshape(-1)
        d = bool(done)

        self._ensure_writer(len(s), len(a), len(o), reward_components)
        if self._csv_writer is None:
            return

        row: dict[str, float | int] = {
            "timestep": self._global_step,
            "r": float(reward),
        }
        
        # Map state values to semantic names
        for i, val in enumerate(s):
            if i < len(self._state_names):
                row[f"s_{self._state_names[i]}"] = float(val)
            else:
                row[f"s_{i}"] = float(val)
        
        # Map action values to semantic names
        for i, val in enumerate(a):
            if i < len(self._action_names):
                row[f"a_{self._action_names[i]}"] = float(val)
            else:
                row[f"a_{i}"] = float(val)
        
        # Map observation values to semantic names
        for i, val in enumerate(o):
            if i < len(self._state_names):
                row[f"o_{self._state_names[i]}"] = float(val)
            else:
                row[f"o_{i}"] = float(val)

        for k in self._reward_component_keys:
            row[f"r_{k}"] = float((reward_components or {}).get(k, 0.0))

        self._csv_writer.writerow(row)
        self._global_step += 1

        if d: self.close()

    def close(self) -> None:
        """Flush and close CSV file."""
        if self._csv_file is not None and not self._csv_file.closed:
            self._csv_file.flush()
            self._csv_file.close()
        self._active = False

    ## To be used if we enable logging transitions during training
    def _on_step(self) -> bool:
        pass
        # """Called every SB3 training step; logs available transitions."""
        # obs = self.locals.get("observations", None)
        # actions = self.locals.get("actions", None)
        # rewards = self.locals.get("rewards", None)
        # dones = self.locals.get("dones", [False])

        # if obs is None or actions is None or rewards is None:
        #     return True

        # obs_batch = np.asarray(obs)
        # actions_batch = np.asarray(actions)
        # rewards_batch = np.asarray(rewards)
        # dones_batch = np.asarray(dones)

        # if obs_batch.ndim == 1:
        #     obs_batch = obs_batch[np.newaxis, :]
        # if actions_batch.ndim == 1:
        #     actions_batch = actions_batch[np.newaxis, :]
        # if rewards_batch.ndim == 0:
        #     rewards_batch = rewards_batch.reshape(1)
        # if dones_batch.ndim == 0:
        #     dones_batch = dones_batch.reshape(1)

        # # In callback mode, previous-state is not exposed via stable key.
        # # We duplicate current obs into state/observation columns.
        # for o, a, r, d in zip(obs_batch, actions_batch, rewards_batch, dones_batch):
        #     self.record_transition(
        #         state=o,
        #         action=a,
        #         observation=o,
        #         reward=float(r),
        #         done=bool(d),
        #         reward_components=None,
        #     )

        # if self._csv_file is not None and not self._csv_file.closed and self._global_step % 100 == 0:
        #     self._csv_file.flush()

    #     return True

    # def _on_training_end(self) -> None:
        pass
        # """Flush and close CSV files on training end."""
        # self.close()
