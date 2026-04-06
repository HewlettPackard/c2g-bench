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
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

T_WARN = 33.0   # °C — thermal warning threshold (mirrors config.yaml)


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
            if info.get("thermal_terminated", False):
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
