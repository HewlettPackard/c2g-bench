"""
Step 1.1 — Workload Orchestrator
=================================
Ingests all four Alibaba trace datasets and converts raw workload metrics
into facility-level IT power demand at 5-minute (300 s) timesteps.

Trace → Power mapping
---------------------
Traces are converted to rack-level GPU utilisation fractions, then to power via
the non-linear server model from DatacenterElectrical (duplicated here as
constants to keep simulators independent and avoid circular imports):

    P_server(u) = N_racks × [P_idle + (P_max - P_idle) × u^alpha]   (kW)

Zone assignment and flex/base classification
--------------------------------------------
+--------------+--------+--------+---------+----------------------------+
| Trace        | Zone   | Class  | Racks   | Notes                      |
+--------------+--------+--------+---------+----------------------------+
| batch_v2023  |   A    | P_flex |  1 200  | Schedulable; DVFS applies  |
| genai_v2026  |   A    | P_base |    800  | Rigid; SLA-protected       |
| dlrm_v2025   |   B    | P_base |  2 500  | Rigid inference serving    |
+--------------+--------+--------+---------+----------------------------+

spot_v2026 is excluded from this release (arrival-based trace requires a
separate queue simulation; reserved for a future extension).

Timestep
--------
All traces are at 5-minute (300 s) resolution which matches ThermalTwin and
BESSModel defaults.  The low-level RL env runs at dt=300 s for Phase 1.
References
----------
[1] Weng, Q., Xiao, W., Yu, Y., et al. (2022) "MLaaS in the Wild: Workload
    Analysis and Scheduling in Large-Scale Heterogeneous GPU Clusters,"
    USENIX NSDI 2022.
    https://www.usenix.org/conference/nsdi22/presentation/weng
    — Alibaba production GPU trace (batch_v2023 & genai_v2026 modelled here).
[2] Guo, J., Chang, Z., Wang, S., et al. (2019) "Who Limits the Resource
    Efficiency of My Datacenter: An Analysis of Alibaba Datacenter Traces,"
    ACM IWQoS 2019.  DOI: 10.1145/3326285.3329074
    — GPU utilization distributions and flex/rigid workload calibration.
[3] Fan, X., Weber, W., Barroso, L.A. (2007) "Power Provisioning for a
    Warehouse-sized Computer," ACM ISCA 2007, pp. 13–23.
    DOI: 10.1145/1273440.1250665
    — P_server(u) = P_idle + (P_max-P_idle)×u^α; α=1.4 GPU superlinear.
[4] Wierman, A., Liu, Z., Liu, I., Mohsenian-Rad, H. (2014) "Opportunities
    and Challenges for Data Center Demand Response," IEEE IGCC 2014.
    DOI: 10.1109/IGCC.2014.7039172
    — deferrable (P_flex) vs. rigid (P_base) workload classification.
[5] Narayanan, D., Shoeybi, M., Casper, J., et al. (2021) "Efficient
    Large-Scale Language Model Training on GPU Clusters Using Megatron-LM,"
    SC 2021.  DOI: 10.1145/3458817.3476209
    — GPU cluster power (rack-level P_max, burst) for genai_v2026 trace.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Server power model constants — mirrors DatacenterElectrical to avoid
# circular imports.  Keep in sync with electrical.py if parameters change.
# ---------------------------------------------------------------------------
_ZONE_A_N_RACKS_FLEX = 1_200         # Batch: 60% of Zone A (2000 racks)
_ZONE_A_N_RACKS_BASE = 800           # GenAI: 40% of Zone A
_ZONE_B_N_RACKS      = 2_500         # DLRM:  all of Zone B
_P_IDLE_A_KW   = 8.0                 # Idle power per rack, Zone A (kW)
_P_MAX_A_KW    = 75.0                # Peak power per rack, Zone A (kW)
_ALPHA_A       = 1.4                 # GPU superlinear exponent
_P_IDLE_B_KW   = 4.0                 # Idle power per rack, Zone B (kW)
_P_MAX_B_KW    = 40.0                # Peak power per rack, Zone B (kW)
_ALPHA_B       = 1.2                 # CPU/inference exponent

# Normalisation denominators — 99th-percentile values from trace stats to
# cap utilisation at 1.0 while preserving the shape of heavy-tail bursts.
_BATCH_GPU_MILLI_MAX = 12_620.0      # batch: max observed gpu_milli_request
_DLRM_GPU_MAX        = 227.0         # dlrm:  max observed active_gpu_count
_GENAI_DUTY_MAX      = 100.0         # genai: duty_cycle already in [0, 100]
_GENAI_SPIKE_PCT75   = 12.19         # genai: P75 duty cycle → spike threshold


def _rack_power_kw(
    util: np.ndarray | float,
    n_racks: int,
    p_idle_kw: float,
    p_max_kw: float,
    alpha: float,
) -> np.ndarray | float:
    """Non-linear server power model (vectorised)."""
    u = np.clip(util, 0.0, 1.0)
    return n_racks * (p_idle_kw + (p_max_kw - p_idle_kw) * u ** alpha)


@dataclass(frozen=True)
class WorkloadState:
    """IT power snapshot at a single 300 s timestep."""
    p_base_kw:      float   # Rigid IT load: GenAI (Zone A) + DLRM (Zone B)  [kW]
    p_base_a_kw:    float   # Zone A rigid load: GenAI inference racks         [kW]
    p_base_b_kw:    float   # Zone B rigid load: DLRM serving racks            [kW]
    p_flex_nom_kw:  float   # Schedulable batch load at full throttle          [kW]
    p_flex_kw:      float   # Batch load after DVFS throttle                   [kW]
    p_total_it_kw:  float   # p_base + p_flex                                  [kW]
    throttle:       float   # DVFS factor applied                           [0, 1]
    is_spike_active: bool   # True when GenAI duty cycle > P75 threshold
    tick:           int     # Current trace tick index


class WorkloadOrchestrator:
    """
    Manages IT power demand for the C2G-Bench 250 MW hyperscale facility.

    Fuses three Alibaba trace datasets (batch 2023, DLRM 2025, GenAI 2026)
    into synchronised P_base and P_flex time-series at 300 s resolution.

    Parameters
    ----------
    trace_dir:
        Directory containing the preprocessed trace CSVs.  Must contain:
        ``batch_v2023.csv``, ``dlrm_v2025.csv``, ``genai_v2026.csv``.
        Defaults to ``data/processed/workload_traces`` relative to cwd.
    dt_seconds:
        Simulation timestep in seconds.  Must match trace resolution (300 s).
    seed:
        Random seed for stochastic fallback (used only if trace files are
        unavailable).
    """

    _TRACE_FILES = {
        "batch": "batch_v2023.csv",
        "dlrm":  "dlrm_v2025.csv",
        "genai": "genai_v2026.csv",
    }

    def __init__(
        self,
        trace_dir: str | Path = "data/processed/workload_traces",
        dt_seconds: float = 300.0,
        seed: int = 42,
    ) -> None:
        self.dt = dt_seconds
        self._rng = np.random.default_rng(seed)
        self._tick: int = 0
        # Traces are at 300-s resolution; this many env steps share each tick.
        self._steps_per_trace_tick: int = max(1, round(300.0 / dt_seconds))

        trace_dir = Path(trace_dir)
        traces = self._load_traces(trace_dir)

        # Build power series aligned to the shortest trace (dlrm = 8640 ticks)
        n = min(len(traces["batch"]), len(traces["dlrm"]))

        batch = traces["batch"].reindex(range(n), fill_value=0)
        dlrm  = traces["dlrm"].reindex(range(n),  fill_value=traces["dlrm"].iloc[-1])

        # GenAI has only 288 ticks (1 day) → tile across full horizon
        genai_raw = traces["genai"]
        repeats   = int(np.ceil(n / len(genai_raw)))
        genai     = pd.concat([genai_raw] * repeats, ignore_index=True).iloc[:n]

        # --- Utilisation fractions ----------------------------------------
        util_batch = (batch["gpu_milli_request"].values / _BATCH_GPU_MILLI_MAX
                     ).clip(0.0, 1.0)
        util_dlrm  = (dlrm["active_gpu_count"].values  / _DLRM_GPU_MAX
                     ).clip(0.0, 1.0)
        util_genai = (genai["avg_gpu_duty_cycle"].values / _GENAI_DUTY_MAX
                     ).clip(0.0, 1.0)

        # --- Convert to power (kW) ----------------------------------------
        self._p_flex_arr   = _rack_power_kw(util_batch, _ZONE_A_N_RACKS_FLEX,
                                             _P_IDLE_A_KW, _P_MAX_A_KW, _ALPHA_A)
        self._p_base_a_arr = _rack_power_kw(util_genai, _ZONE_A_N_RACKS_BASE,
                                             _P_IDLE_A_KW, _P_MAX_A_KW, _ALPHA_A)
        self._p_base_b_arr = _rack_power_kw(util_dlrm,  _ZONE_B_N_RACKS,
                                             _P_IDLE_B_KW, _P_MAX_B_KW, _ALPHA_B)
        self._p_base_arr   = self._p_base_a_arr + self._p_base_b_arr

        # Spike flag: GenAI duty cycle above P75
        self._spike_arr = (genai["avg_gpu_duty_cycle"].values > _GENAI_SPIKE_PCT75)

        self._n = n

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, throttle_batch: float) -> WorkloadState:
        """
        Return the IT power state for the current tick and advance by one step.

        Parameters
        ----------
        throttle_batch:
            DVFS factor for schedulable batch jobs in [0, 1].
            1.0 = full speed (max power); 0.0 = fully suspended.
        """
        throttle_batch = float(np.clip(throttle_batch, 0.0, 1.0))
        # Trace is at 300-s resolution; hold constant across sub-ticks.
        trace_tick = self._tick // self._steps_per_trace_tick
        idx = trace_tick % self._n

        p_base_a  = float(self._p_base_a_arr[idx])
        p_base_b  = float(self._p_base_b_arr[idx])
        p_base    = p_base_a + p_base_b
        p_flex_n  = float(self._p_flex_arr[idx])
        p_flex    = p_flex_n * throttle_batch
        is_spike  = bool(self._spike_arr[idx])

        self._tick += 1
        return WorkloadState(
            p_base_kw      = p_base,
            p_base_a_kw    = p_base_a,
            p_base_b_kw    = p_base_b,
            p_flex_nom_kw  = p_flex_n,
            p_flex_kw      = p_flex,
            p_total_it_kw  = p_base + p_flex,
            throttle       = throttle_batch,
            is_spike_active= is_spike,
            tick           = idx,
        )

    def reset(self, seed: int | None = None) -> None:
        """Reset trace pointer to the beginning."""
        self._tick = 0
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    @property
    def horizon_ticks(self) -> int:
        """Total number of ticks before the trace loops."""
        return self._n

    @property
    def p_flex_max_kw(self) -> float:
        """Theoretical maximum flexible load (all batch racks at peak)."""
        return float(_rack_power_kw(1.0, _ZONE_A_N_RACKS_FLEX,
                                    _P_IDLE_A_KW, _P_MAX_A_KW, _ALPHA_A))

    @property
    def p_base_range_kw(self) -> tuple[float, float]:
        """(min, max) of P_base across the full trace horizon."""
        return float(self._p_base_arr.min()), float(self._p_base_arr.max())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_traces(self, trace_dir: Path) -> dict[str, pd.DataFrame]:
        """Load and validate the three trace CSVs."""
        required_cols = {
            "batch": {"tick", "gpu_milli_request"},
            "dlrm":  {"tick", "active_gpu_count"},
            "genai": {"tick", "avg_gpu_duty_cycle"},
        }
        frames: dict[str, pd.DataFrame] = {}
        for key, fname in self._TRACE_FILES.items():
            path = trace_dir / fname
            if not path.exists():
                raise FileNotFoundError(
                    f"Trace file not found: {path}\n"
                    f"Run preprocessing/workload_traces/process_{key}*.py first."
                )
            df = pd.read_csv(path).set_index("tick").sort_index()
            missing = required_cols[key] - (set(df.columns) | {"tick"})
            if missing:
                raise ValueError(f"{fname} missing columns: {missing}")
            frames[key] = df
        return frames
