# C2G-Bench: Hierarchical AI Orchestration for Grid-Interactive Hyperscale Data Centers

**Target Venue:** NeurIPS 2026 — Datasets and Benchmarks Track  
**Strategic Alignment:** HPE Edge-to-Cloud, US DOE Genesis Mission, EU Horizon Europe (Cluster 5)

---

## 1. Executive Summary

This project addresses the **"AI-Energy Paradox"** by transforming 250 MW+ hyperscale data centers from passive power consumers into active, grid-balancing assets. By establishing a formal **Energy System Handshake**, we enable data centers to provide wholesale Frequency Regulation, stabilizing the regional transmission grid in exchange for significant revenue and faster deployment permits.

We solve this using a **Hierarchical AI Orchestration** framework that bridges long-term energy market bidding (minutes/hours) and sub-second hardware physics. The framework evaluates the synergy between three critical control levers: **Throttling Batch Workloads (DVFS)**, **Modulating Cooling Thermal Inertia (CDU pump)**, and **Dispatching Battery Energy Storage (BESS)**. This project delivers a high-fidelity cyber-physical benchmark for NeurIPS 2026, positioning HPE at the frontier of autonomous, grid-interactive infrastructure.

---

## 2. Problem Statement: The "Handshake" Gap

Current data center management systems are "grid-blind": they optimize internal efficiency (PUE) while ignoring the real-time needs of the regional energy system.

- **The Grid Need:** Modern grids require large loads to respond to Frequency Regulation signals (e.g., PJM RegD) every 2–4 seconds to balance renewable energy volatility.
- **The Datacenter Barrier:** Standard AI controllers cannot track these high-speed signals because they do not account for the non-linear physics of liquid cooling, battery degradation, and the bursty nature of GenAI workloads.
- **The Objective:** Create a synergy where the data center matches the grid's power signal perfectly without violating hardware safety limits or AI training SLAs.

---

## 3. State-of-the-Art and Our Contribution

| SOTA | Gap | Our Step Further |
|------|-----|-----------------|
| **Wang et al., 2019** — Proved DCs can follow grid signals using DVFS. | Used "dummy loads" to intentionally waste power to meet the signal. | We use **BESS + thermal storage synergy** — no wasted power. |
| **Fu et al., 2021** — Demonstrated cooling systems have "thermal inertia" for grid services. | Relies on classical MPC, which fails under unpredictable GenAI serving spikes. | We replace MPC with **Hierarchical RL** to handle extreme, non-linear volatility of Alibaba GenAI traces. |
| **Li et al., 2026** — Identifies the need for intelligent VPP aggregation. | Lacks a standardized, high-fidelity physical testbed for datacenters. | We provide the **first 250 MW-scale evaluation testbed** with real data across 6 global energy markets. |

---

## 4. Technical Solution: Hierarchical AI Orchestration

### 4.1. Upper-Level Agent: The Market Orchestrator (15-min ticks)

Manages the "Business Handshake." Observes regional market prices, weather forecasts, and the Alibaba batch job queue.

> **Decision:** *"How much flexible MW capacity should I commit to the grid operator for the next 15 minutes?"*

- **Action Space (2-D):** `[commit_norm ∈ [0,1], bess_target ∈ [-1,1]]` — MW commitment and average BESS dispatch.
- **Observation Space (16-D):** Aggregated over 180 sub-steps — mean temps, SOC, tracking error, spike flag, thermal headroom, LMP, previous action, mean frequency deviation, mean PCC voltage.
- **Reward:** mean of sub-step rewards + LMP dispatch revenue − commitment-churn penalty.

### 4.2. Lower-Level Agent: The Hardware Controller (5 s ticks)

Executes the physical "Handshake." Receives the real-time frequency regulation signal and uses **four physical levers**:

| Lever | Action dim | Range | Effect |
|-------|-----------|-------|--------|
| **IT (DVFS)** | `action[0]` | [0, 1] | Throttles schedulable Alibaba batch jobs; GenAI/DLRM rigid loads unaffected |
| **Cooling (CDU pump)** | `action[1]` | [0, 1] | Modulates liquid cooling pump speed, exploiting thermal inertia |
| **HVAC** | `action[2]` | [0, 1] | Zone B air-side fan speed |
| **BESS** | `action[3]` | [-1, 1] | Charge (−) / discharge (+) the 150 MWh battery |

- **Action Space (4-D, continuous):** `[throttle_batch, pump_speed_A, hvac_effort, bess_dispatch]`
- **Observation Space (16-D, normalised):**
  | Index | Name | Range | Description |
  |-------|------|-------|-------------|
  | 0 | `temp_A_norm` | [0, 2] | Zone A (liquid-cooled GPU) temperature / T_safe |
  | 1 | `temp_B_norm` | [0, 2] | Zone B (air-cooled CPU) temperature / T_safe |
  | 2 | `bess_soc` | [0, 1] | Battery state of charge |
  | 3 | `p_base_norm` | [0, 1] | Rigid IT load (GenAI + DLRM) |
  | 4 | `p_flex_nom_norm` | [0, 1] | Schedulable batch load at full throttle |
  | 5 | `p_facility_norm` | [0, 2] | Total facility power |
  | 6 | `regd_signal` | [-1, 1] | Grid regulation signal (signed) |
  | 7 | `lmp_norm` | [0, 1] | Locational marginal price |
  | 8 | `grid_load_norm` | [0, 1] | Regional grid load stress indicator |
  | 9 | `is_spike` | {0, 1} | GenAI serving spike flag |
  | 10 | `prev_throttle` | [0, 1] | Previous DVFS throttle |
  | 11 | `prev_pump_speed` | [0, 1] | Previous pump speed |
  | 12 | `pue_norm` | [0, 2] | Current Power Usage Effectiveness |
  | 13 | `T_amb_norm` | [0, 1] | Ambient temperature |
  | 14 | `freq_dev_norm` | [-1, 1] | Normalised grid frequency deviation (swing equation) |
  | 15 | `v_pcc_pu` | [0, 1.1] | PCC voltage in per-unit (Thévenin model) |

### 4.3. The NeurIPS Evaluation Metric: The Tracking Reward

$$\mathcal{R} = \alpha \cdot u_{\text{thr}} - \beta \cdot \frac{|\Delta P_{\text{demand}} - \Delta P_{\text{actual}}|}{P_{\text{norm}}} - \gamma \cdot (T - T_{\text{warn}})^{+} - \delta_{\text{soc}} \cdot \mathbf{1}_{\text{soc}} - \delta_f \cdot (|\Delta f| - 0.2)^{+} - \delta_v \cdot \varepsilon_v$$

where $(x)^{+} = \max(0, x)$, $\varepsilon_v = (0.95 - v_{\text{pcc}})^{+} + (v_{\text{pcc}} - 1.05)^{+}$, and the coefficients are $\alpha{=}1.0$, $\beta{=}2.0$, $\gamma{=}5.0$, $\delta_f{=}2.0$, $\delta_v{=}5.0$.

The tracking loop: $\Delta P_{\text{actual}} = (1 - \text{throttle}) \times P_{\text{flex,nom}} + P_{\text{BESS,actual}}$

**Termination** (episode ends immediately):
- Thermal fault: $T_A > 35°$C or $T_B > 35°$C
- Frequency fault: $|f - f_{\text{nom}}| > 0.5$ Hz (UFLS / over-frequency trip)
- Voltage fault: $v_{\text{pcc}} < 0.90$ pu (under-voltage relay)

Episode truncates at 17,280 ticks (24 hours at 5 s).

---

## 5. Simulators

Seven independent physics/data modules, all with exact-exponential or analytical solutions (unconditionally stable):

| Simulator | File | Description |
|-----------|------|-------------|
| **Workload Orchestrator** | `workload.py` | Fuses Alibaba batch (2023), DLRM (2025), and GenAI (2026) traces into P_base + P_flex at 5-min resolution |
| **Thermal Twin** | `thermal.py` | Exact exponential ODE integration for dual-zone cooling (Zone A: HPE Cray EX liquid, Zone B: HPE ProLiant air) |
| **Electrical Chain** | `electrical.py` | Non-linear UPS/PDU/XFMR loss curves + PUE calculation |
| **BESS** | `bess.py` | 150 MWh / 50 MW Li-ion NMC (pure-Python backend + optional PySAM) with C-rate η, SOC derating, capacity fade |
| **Macro-Grid** | `macro_grid.py` | AR(1) RegD signal + LMP proxy; calibrated for 6 global markets |
| **Renewable** | `renewable.py` | IEC wind power curve (100 MW) + solar PV (75 MW) with degradation |
| **Weather** | `weather.py` | NOAA ISD-Lite real data or calibrated synthetic (6 climate profiles) |

---

## 6. Data

### Real Datasets

| Dataset | Source | Markets/Zones | Resolution | Files |
|---------|--------|--------------|------------|-------|
| **Workload traces** | Alibaba cluster traces | batch, DLRM, GenAI, spot | 5-min | 4 CSVs |
| **Energy load** | EIA, SMARD.de, AEMO | NYISO (11 zones), PJM, CAISO, ERCOT, ENTSO-E DE, AEMO NSW | 5-min (resampled) | 16 CSVs |
| **Weather** | NOAA ISD-Lite | NYC, DCA, SJC, DFW, FRA, BKT | Hourly | 7 CSVs |
| **Renewable** | Synthetic (IEC/PVUSA calibrated) | Wind + Solar | 5-min | 4 CSVs |

### 6 Global Energy Markets

| Market Key | Region | Grid Operator | Energy Source | Weather Station |
|-----------|--------|---------------|---------------|-----------------|
| `nyiso_nyc` | New York City | NYISO | NYISO OASIS | NYC (Central Park) |
| `pjm_dom` | Northern Virginia | PJM | EIA API | DCA (Reagan Natl) |
| `caiso_pgae` | Bay Area / San Jose | CAISO | EIA API | SJC (Mineta Intl) |
| `ercot_north` | Dallas–Fort Worth | ERCOT | EIA API | DFW (DFW Intl) |
| `entso_de` | Frankfurt, Germany | ENTSO-E / EPEX | SMARD.de | FRA (Frankfurt) |
| `aemo_nsw` | Sydney, Australia | AEMO / NEM | AEMO CSVs | BKT (Bankstown) |

---

## 7. Evaluation Scenarios

C2G-Bench ships four progressively harder 24-hour scenarios (17,280 ticks at 5 s each). Every scenario is fully deterministic when a fixed seed is set and can be combined with any of the six energy markets via a single Hydra override.

```bash
# Run any scenario × any market
uv run python baselines/train_ppo.py scenario=scenario_b market=ercot_north
```

### 7.1 Scene-setting: shared physics

All scenarios share the same underlying simulator stack and reward weights:

| Parameter | Value | Meaning |
|-----------|-------|---------|
| Episode length | 17,280 ticks | 24 h × 3,600 s h⁻¹ ÷ 5 s tick⁻¹ |
| IT capacity | 250 MW | Rigid (GenAI/DLRM) + flexible (Alibaba batch) |
| BESS | 150 MWh / 50 MW | NMC Li-ion, C-rate derating + capacity fade |
| Cooling zones | Zone A (liquid, HPE Cray EX) · Zone B (air, HPE ProLiant) | |
| $T_{\text{safe}}$ | 35 °C | Silicon hard limit → immediate termination |
| $T_{\text{warn}}$ | 33 °C | Soft threshold → thermal penalty begins |
| Frequency UFLS | ±0.5 Hz | Under/over-frequency relay → termination |
| Voltage UV relay | 0.90 pu | Under-voltage → termination |

---

### 7.2 `default` — Baseline Operations

> *"Can the agent learn to coordinate four physical levers under normal grid conditions?"*

The entry-level scenario. Ambient temperature is comfortable (25 °C, NYISO NYC summer), BESS starts at 50 % SOC, and the regulation signal has standard amplitude. No faults are injected. This is the recommended starting point for algorithm development and ablation studies.

| Parameter | Value |
|-----------|-------|
| Market | NYISO NYC |
| Ambient $T_{\text{amb}}$ | 25 °C (weather-driven) |
| Committed MW | 15 MW |
| BESS SOC₀ | 50 % |
| GenAI spike scale | 1.0× (nominal) |
| Grid stress scale | 1.0× (nominal) |
| Cooling fault | None |

**Primary challenge:** Learning the basic DVFS ↔ cooling ↔ BESS synergy to track the regulation signal while keeping temperatures below $T_{\text{warn}}$.

**Termination risk:** Low. An untrained random agent survives ≈ 40 % of the episode on average.

---

### 7.3 `scenario_a` — GenAI Crisis

> *"A viral model launch + a grid under-frequency event hit simultaneously. The agent must shed flexible load without starving the BESS."*

This scenario models a **Northern Virginia (PJM DOM)** summer day when a new GPT-class model goes viral. GenAI serving load spikes to **1.8× nominal**, consuming headroom that the agent would otherwise use for regulation. At the same time, the grid issues a sustained under-frequency signal, demanding active discharge. The agent must resolve the conflict between IT throughput and grid support.

| Parameter | Value |
|-----------|-------|
| Market | PJM DOM |
| Ambient $T_{\text{amb}}$ | 30 °C (static) |
| Committed MW | 20 MW |
| BESS SOC₀ | 55 % |
| GenAI spike scale | **1.8×** |
| Grid stress scale | **1.5×** |
| Cooling fault | None |

**Primary challenge:** IT vs. grid conflict. The GenAI rigid load is non-throttleable, so the agent must use BESS discharge and batch-job throttling simultaneously — but throttling reduces throughput reward $\alpha \cdot u_{\text{thr}}$, and over-discharging depletes the BESS.

**Termination risk:** Medium–High. Frequency faults are likely if the agent ignores the regulation signal. Thermal faults are possible if cooling is under-prioritised during spikes.

---

### 7.4 `scenario_b` — Thermal Squeeze

> *"Dallas in August: 40 °C ambient, a 30 MW commitment, and a cooling system pushed to its physical limits."*

This scenario targets **ERCOT North (DFW)** during a peak-summer heat wave. The 40 °C ambient temperature drives the cooling COP down by ≈ 30 %, meaning the pump must work harder to achieve the same heat rejection. The committed MW is raised to 30 MW, increasing the power swings the agent must track. GenAI load is nominal, but the thermal margin to $T_{\text{safe}}$ is extremely thin.

| Parameter | Value |
|-----------|-------|
| Market | ERCOT North |
| Ambient $T_{\text{amb}}$ | **40 °C** (static) |
| Committed MW | **30 MW** |
| BESS SOC₀ | 60 % |
| GenAI spike scale | 1.0× (nominal) |
| Grid stress scale | 1.3× |
| Cooling fault | None |

**Primary challenge:** Thermal constraint binding. The thermal penalty $\gamma \cdot (T - T_{\text{warn}})^{+}$ dominates the reward signal. The agent must learn aggressive pump-speed scheduling and accept reduced throughput to keep temperatures in the safe band.

**Termination risk:** Very High. A naive agent that ignores the pump lever will hit $T_{\text{safe}} = 35$ °C within the first hour. This scenario is the primary driver of thermal-safety research.

---

### 7.5 `scenario_c` — Battery Drain

> *"Western Sydney summer: the BESS starts nearly empty, the pump is failing, and the grid is stressed."*

This scenario represents a compounding failure in **AEMO NSW**. The BESS begins at only **15 % SOC** (near the 10 % hard floor), leaving almost no discharge capacity for regulation. A simulated CDU pump degradation reduces cooling efficiency to **60 % of nominal**, tightening the thermal margin. GenAI and grid stress are both elevated. The agent must simultaneously ration the BESS, compensate for degraded cooling, and track the regulation signal — with essentially no buffer.

| Parameter | Value |
|-----------|-------|
| Market | AEMO NSW |
| Ambient $T_{\text{amb}}$ | 32 °C (static) |
| Committed MW | 20 MW |
| BESS SOC₀ | **15 %** |
| GenAI spike scale | 1.2× |
| Grid stress scale | 1.2× |
| Cooling fault | **Pump degradation (60 % efficiency)** |

**Primary challenge:** Resource scarcity under compound failure. The BESS SOC penalty $\delta_{\text{soc}}$ activates immediately. The agent must switch to DVFS-only regulation while the pump fault is active, and carefully trickle-charge the BESS when the regulation signal allows.

**Termination risk:** Extreme. This is the hardest scenario in the benchmark. A random agent terminates within ≈ 5 % of the episode on average.

---

### 7.6 Scenario × Market grid

All four scenarios can be combined with all six markets, yielding **24 distinct evaluation configurations**. Market selection changes the LMP profile, weather driver, and grid-stress statistics, while scenario selection changes the hardware stress and initial conditions:

| | `nyiso_nyc` | `pjm_dom` | `caiso_pgae` | `ercot_north` | `entso_de` | `aemo_nsw` |
|-|:-----------:|:---------:|:------------:|:-------------:|:----------:|:----------:|
| `default` | ★ default | | | | | |
| `scenario_a` | | ★ default | | | | |
| `scenario_b` | | | | ★ default | | |
| `scenario_c` | | | | | | ★ default |

★ = default market for that scenario. Any other cell is a valid cross-market stress test.

```bash
# Example: Thermal Squeeze under European low-carbon prices
uv run python baselines/train_ppo.py scenario=scenario_b market=entso_de experiment.seed=1
```

---

## 8. Repository Structure

```
C2G-Macro/
├── pyproject.toml                       # uv/hatchling build + all dependencies
├── uv.lock                              # Reproducible dependency lock
├── README.md
│
├── c2g_env/                             # The Core RL Environment
│   ├── __init__.py                      # Exports C2GFastEnv, C2GMacroEnv
│   ├── env_low_level.py                 # 5 s physics step — C2GFastEnv (16-D obs, 4-D act)
│   ├── env_high_level.py                # 15-min market step — C2GMacroEnv (16-D obs, 2-D act)
│   ├── config.yaml                      # Centralised env configuration
│   └── simulators/
│       ├── workload.py                  # Alibaba trace fusion (batch/DLRM/GenAI)
│       ├── thermal.py                   # Exact-exponential ODEs, dual-zone cooling
│       ├── electrical.py                # Non-linear UPS/PDU/XFMR loss + PUE
│       ├── bess.py                      # 150 MWh NMC BESS (pure-Python + PySAM)
│       ├── macro_grid.py                # AR(1) RegD + LMP proxy, 6 market presets
│       ├── renewable.py                 # IEC wind + solar PV (100 MW + 75 MW)
│       └── weather.py                   # NOAA ISD real data + synthetic climate, 6 presets
│
├── data/
│   ├── raw/                             # Original trace files
│   └── processed/
│       ├── workload_traces/             # batch_v2023, dlrm_v2025, genai_v2026, spot_v2026
│       ├── energy/                      # 16 CSVs: 11 NYISO zones + PJM/CAISO/ERCOT/ENTSO-E/AEMO
│       ├── weather/                     # 7 CSVs: NYC, DCA, SJC, DFW, FRA, BKT + merged
│       └── renewable/                   # wind_5min, solar_5min, wind_hourly, solar_hourly
│
├── conf/                                # Hydra configuration tree
│   ├── config.yaml                      # Top-level defaults
│   ├── algo/                            # ppo.yaml, sac.yaml, ppo_macro.yaml
│   ├── scenario/                        # default, scenario_a, scenario_b, scenario_c
│   ├── market/                          # nyiso_nyc, pjm_dom, caiso_pgae, ercot_north, entso_de, aemo_nsw
│   └── logging/                         # tensorboard.yaml
│
├── baselines/                           # NeurIPS Evaluation Agents
│   ├── train_ppo.py                     # SB3 PPO + Hydra + VecNormalize + callbacks
│   ├── train_sac.py                     # SB3 SAC (off-policy, auto entropy)
│   ├── train_ppo_macro.py               # PPO on C2GMacroEnv (optional inner policy)
│   ├── train_hierarchical.py            # Two-phase sequential HRL pipeline
│   ├── train_shielded_ppo.py            # PPO inside ShieldedEnv (safety-filtered)
│   ├── rule_based_mpc.py                # Classical threshold controller (SB3-compatible API)
│   ├── rule_based_macro.py              # Macro-level rule-based controller
│   ├── safety_shield.py                 # Simplex safety filter (5 hard constraints)
│   └── metrics_callback.py              # C2GMetricsCallback — per-episode CSV + TensorBoard
│
├── evaluation/                          # Benchmark auditing
│   ├── run_benchmark.py                 # Runs agents on all 4 scenarios
│   └── generate_plots.py                # Publication-ready PDF/PNG figures
│
├── scripts/                             # Data download & training utilities
│   ├── download_weather.py              # Open-Meteo ERA5 → 6 weather CSVs
│   ├── download_energy.py               # EIA + SMARD + AEMO → 5 energy CSVs
│   └── run_sweep.sh                     # Full training sweep (4 scenarios × 2 algos × 3 seeds)
│
├── preprocessing/                       # Raw → processed data pipelines
│   ├── workload_traces/                 # process_v2023.py, process_v2025.py, process_v2026_genai.py
│   ├── energy/                          # process_energy.py (NYISO zone load)
│   ├── renewable/                       # process_renewable.py, download_renewable.py
│   └── weather/                         # download_noaa_isd.py
│
├── notebooks/                           # 8 Jupyter notebooks for exploration & visualisation
│   ├── 01_workload.ipynb                # Alibaba trace analysis
│   ├── 02_thermal.ipynb                 # Thermal model step response & steady-state
│   ├── 03_electrical_bess.ipynb         # Electrical chain + BESS cycling
│   ├── 04_macro_grid.ipynb              # RegD signal + LMP proxy
│   ├── 05_renewable.ipynb               # Wind/solar generation profiles
│   ├── 06_environments.ipynb            # Gym API demo, scenario comparison
│   ├── 07_weather.ipynb                 # Weather data: 6 markets, real vs. synthetic
│   ├── 08_energy_markets.ipynb          # Energy load: 6 markets, LDC, diurnal patterns
│   ├── 09_frequency_voltage.ipynb       # Grid frequency & PCC voltage safety signals
│   └── 10_evaluation_scenarios.ipynb    # Scenario deep dive: params, rollouts, risk, reward
│
├── paper/                               # NeurIPS 2026 manuscript
│   ├── main.tex                         # 13-page paper (NeurIPS format)
│   ├── references.bib
│   ├── neurips2026.sty
│   └── figures/                         # fig1–fig4 (architecture, simulators, curves, trajectory)
│
├── tests/                               # 371 tests (pytest)
│   ├── test_workload.py                 # 26 tests
│   ├── test_thermal.py                  # 32 tests
│   ├── test_electrical.py               # 25 tests
│   ├── test_macro_grid.py               # 33 tests
│   ├── test_renewable.py                # 26 tests
│   ├── test_weather.py                  # 18 tests
│   ├── test_gym_api.py                  # 73 tests (API compliance both envs)
│   ├── test_baselines.py                # 26 tests
│   ├── test_frequency_voltage.py        # 35 tests (freq/voltage safety signals)
│   ├── test_hierarchical.py             # 17 tests (HRL, macro agents)
│   └── test_safety_shield.py            # 27 tests (Simplex shield, wrappers)
│
└── trained_models/                      # Saved checkpoints from training runs
    └── ppo_default_s42/                 # PPO on default scenario, seed=42
```

---

## 9. Quick Start

### Prerequisites

- **Python ≥ 3.11**
- **[uv](https://docs.astral.sh/uv/)** — fast Python package manager

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Clone & install

```bash
git clone <repo-url>
cd C2G-Macro
uv sync
uv sync --extra dev   # pytest, ruff, mypy
```

### Run the tests

```bash
uv run pytest tests/ -q
# 371 passed
```

### Train a single agent

```bash
# PPO — default scenario, 300k steps
uv run python baselines/train_ppo.py

# PPO — GenAI Crisis + PJM market
uv run python baselines/train_ppo.py scenario=scenario_a market=pjm_dom

# SAC — Thermal Squeeze
uv run python baselines/train_sac.py algo=sac scenario=scenario_b

# Hydra multirun — all scenarios × 3 seeds
uv run python baselines/train_ppo.py --multirun \
    scenario=default,scenario_a,scenario_b,scenario_c \
    experiment.seed=1,2,3

# Hierarchical RL — sequential two-phase pipeline
uv run python baselines/train_hierarchical.py

# Safety-shielded PPO (provable constraint satisfaction)
uv run python baselines/train_shielded_ppo.py scenario=default
```

### Run the full benchmark sweep

```bash
# Dry-run first — prints all 48 jobs without executing anything:
bash scripts/run_sweep.sh --dry-run

# Full sweep (default: 4 parallel jobs):
bash scripts/run_sweep.sh

# Use more parallelism (208 cores available — 16 is safe):
MAX_PARALLEL=16 bash scripts/run_sweep.sh
```

The sweep runs in 4 phases:

| Phase | Jobs | What runs |
|-------|------|-----------|
| 1 | 24 | Rule-Based + Random evaluation only (no training, ~5 min) |
| 2 | 12 | PPO training (300k steps) + evaluation |
| 3 | 12 | SAC training (200k steps) + evaluation |
| 4 | 4  | Macro-level Rule-Based evaluation (4 scenarios) |
| 5 | 4  | PPO-Macro training (100k steps) + evaluation |
| 6 | 4  | Hierarchical RL training (Phase 1 low-level → Phase 2 macro) |
| 7 | 1  | Summary table + LaTeX rows for Table 5 |

Results are written to `results/sweep_results.csv` (one row per run, upserted on re-runs) and `results/sweep_summary.csv` (mean ± std across seeds).

### Download real-world data (optional — CSVs are bundled)

```bash
uv run python scripts/download_weather.py --year 2024
uv run python scripts/download_energy.py  --year 2024
```

### Explore interactively

```bash
uv run jupyter lab notebooks/
```

> **Note:** The optional `nrel-pysam` BESS backend requires `uv pip install nrel-pysam`.
> The environment automatically falls back to the pure-Python `_SimpleBESSModel` if absent.

---

## 9.5. High-Assurance Safety Controllers

C2G-Bench includes a **Simplex-architecture safety shield** [Sha 2001] that provides **provable hard-constraint satisfaction** for any RL agent, without retraining.

### Safety Shield Design

The shield intercepts every agent action and projects it into the safe subset of the action space in **O(1) time** using analytic worst-case bounds:

| ID | Constraint | Threshold | Shield Response |
|----|-----------|-----------|-----------------|
| C1 | $T_A < T_{\text{safe}}$ | 35 °C (margin 1 °C) | Progressive throttle reduction + forced cooling |
| C2 | $T_B < T_{\text{safe}}$ | 35 °C (margin 1 °C) | Progressive HVAC increase |
| C3 | SOC ∈ [SOC_min, SOC_max] | [0.10, 0.95] (guard 0.03) | Block discharge at low SOC, block charge at high SOC |
| C4 | $|\Delta f| < 0.5$ Hz | ±0.4 Hz trigger | Force discharge on under-frequency, charge on over-frequency |
| C5 | $V_{\text{pcc}} > 0.90$ pu | 0.92 pu trigger | Proportional throttle reduction |

### Three Usage Modes

```python
# 1. Standalone filter — works with ANY agent
from baselines.safety_shield import SafetyShield
shield = SafetyShield()
safe_action, was_modified, info = shield.filter(raw_action, obs)

# 2. Gymnasium wrapper — agent trains inside safe manifold
from baselines.safety_shield import ShieldedEnv
env = ShieldedEnv(C2GFastEnv(scenario="default"))

# 3. SB3-compatible agent wrapper — for evaluation
from baselines.safety_shield import ShieldedAgent
safe_agent = ShieldedAgent(trained_agent, env)
```

### Training with the Shield

```bash
# PPO inside ShieldedEnv (agent learns within safe manifold)
uv run python baselines/train_shielded_ppo.py scenario=default experiment.seed=42
```

### Research Challenges for the Community

The built-in shield is deliberately conservative (O(1), no solver). Researchers are invited to develop more permissive shields using:

- **Control Barrier Functions (CBFs)** — continuous-time safety certificates [Ames 2019]
- **Hamilton-Jacobi reachability** — compute maximal safe sets offline
- **Model-Predictive Safety Filters (MPC-SF)** — receding-horizon constrained optimisation
- **Neural Lyapunov / barrier networks** — learned certificates with formal verification
- **Constrained RL** — PPO-Lagrangian, CPO, PCPO for soft-constraint satisfaction

The benchmark tracks shield intervention rate (`ShieldStats.intervention_rate`) as a key metric: a lower rate indicates better alignment between agent policy and safety constraints.

---

## 10. Strategic Value

### For the Energy System
- **Renewable Integration:** Data centers absorb excess wind/solar, preventing curtailment.
- **Grid Stability:** The DC acts as a "shock absorber" for the transmission grid, reducing reliance on fossil-fuel peaker plants.

### For HPE and Industry Partners
- **New Revenue Streams:** Grid operators (PJM, CAISO, ERCOT) pay for frequency regulation tracking — turning DC energy flexibility into direct profit.
- **Faster Deployment:** Demonstrating the "handshake" proves HPE-powered DCs help stabilize the grid, accelerating construction permits.
- **HPE Hardware Differentiation:** Explicit HPE Cray EX (liquid-cooled Zone A) and ProLiant (air-cooled Zone B) with CDU pump as a thermal-battery lever — unique to this benchmark.

### For AI Research (NeurIPS 2026)
- **Cyber-Physical Benchmark:** The first high-fidelity, multi-market testbed for hierarchical RL on real infrastructure physics.
- **Six Global Markets:** NYISO, PJM, CAISO, ERCOT, ENTSO-E, AEMO — largest DC hubs on Earth.
- **DOE Genesis Alignment:** 250 MW–1 GW scale matches the US national AI infrastructure program.

---

## 11. Citation

```bibtex
@inproceedings{c2gbench2026,
  title     = {{C2G-Bench}: A Cyber-Physical Benchmark for Grid-Interactive
               Hyperscale Data Centres},
  author    = {Anonymous},
  booktitle = {NeurIPS 2026 Datasets and Benchmarks Track},
  year      = {2026},
}
```
