Project Description Document
Project Title: Hierarchical AI Orchestration for Grid-Interactive Hyperscale Data Centers
Target Venue: NeurIPS 2026 (Evaluations & Datasets Track)
Strategic Alignment: HPE Edge-to-Cloud, US DOE Genesis Mission, EU Horizon Europe (Cluster 5)
________________________________________
1. Executive Summary
This project addresses the "AI-Energy Paradox" by transforming 250MW+ hyperscale data centers from passive power consumers into active, grid-balancing assets. By establishing a formal Energy System Handshake, we enable data centers to provide wholesale Frequency Regulation, stabilizing the regional transmission grid in exchange for significant revenue and faster deployment permits.
We solve this using a Hierarchical AI Orchestration framework. This architecture bridges the gap between long-term energy market bidding (minutes/hours) and sub-second hardware physics. The framework evaluates the synergy between three critical control levers: Throttling Batch Workloads, Modulating Cooling Thermal Inertia, and Dispatching Battery Energy Storage (BESS). This project will deliver a high-fidelity cyber-physical benchmark for NeurIPS 2026, positioning HPE at the frontier of autonomous, grid-interactive infrastructure.
________________________________________
2. Problem Statement: The "Handshake" Gap
Current data center management systems are "grid-blind." They focus on internal efficiency (PUE) while ignoring the real-time needs of the regional energy system.
	The Grid Need: Modern grids require large loads to respond to Frequency Regulation signals (e.g., PJM RegD) every 2–4 seconds to balance renewable energy volatility.
	The Datacenter Barrier: Standard AI controllers cannot track these high-speed signals because they do not account for the non-linear physics of liquid cooling, battery degradation, and the bursty nature of GenAI workloads.
	The Objective: Create a synergy where the data center matches the grid's power signal perfectly without violating hardware safety limits or AI training SLAs.
________________________________________
3. State-of-the-Art (SOTA) and Our "Step Further"
We analyzed the latest SOTA in energy systems to ensure this project advances the frontier:
	SOTA 1: Computational Flexibility (Wang et al., 2019): Proved datacenters can follow grid signals using Dynamic Voltage and Frequency Scaling (DVFS).
	The Gap: They used "dummy loads" to intentionally waste power to meet the signal.
	Our Step Further: We do not waste power. We use the battery (BESS) and thermal storage synergy to track the signal efficiently.
	SOTA 2: Thermal Energy Storage (Fu et al., 2021): Demonstrated that cooling systems have "thermal inertia," allowing datacenters to pause cooling temporarily to help the grid.
	The Gap: Relies on classical Model Predictive Control (MPC), which fails when faced with unpredictable GenAI serving spikes.
	Our Step Further: We replace MPC with Hierarchical Reinforcement Learning to handle the extreme, non-linear volatility of modern Alibaba GenAI traces that MPC cannot predict.
	SOTA 3: AI-Driven VPPs (Li et al., 2026): Identifies the need for intelligent aggregation of distributed energy resources.
	The Gap: Lacks a standardized, high-fidelity physical testbed for datacenters.
	Our Step Further: We provide the first high-assurance, 250MW-scale evaluation testbed to operationalize these concepts for hyperscale supercomputing.
Wang et al., 2019: Frequency regulation service provision in data center with computational flexibility. https://intra.ece.ucr.edu/~nyu/papers/2019-Data-Center-Frequency-Regulation.pdf
Fu et al., 2021: Multi-stage Power Scheduling Framework for Data Center with Chilled Water Storage in Energy and Regulation Markets. https://arxiv.org/abs/2007.09770
Li et al., 2026: AI-Driven Virtual Power Plants: A Comprehensive Review. https://www.mdpi.com/1996-1073/19/4/1084
________________________________________
4. Technical Solution: Hierarchical AI Orchestration
The project implements a two-tier AI architecture to manage the different time scales of the energy-compute synergy.
4.1. Upper-Level Agent: The Market Orchestrator (15–60 Minute Ticks)
This agent manages the "Business Handshake." It observes regional market prices (NYISO/PJM/CAISO), weather forecasts, and the Alibaba batch job queue.
	The Decision: It decides the "Macro Strategy." For example: "How much flexible MW capacity should I commit to the grid operator for the next hour?"
4.2. Lower-Level Agent: The Hardware Controller (2–4 Second Ticks)
This agent executes the physical "Handshake." It receives the real-time frequency regulation signal (e.g., drop 10MW now) from the grid. It uses three ultra-fast physical levers to match the signal:
	The IT Lever: Uses DVFS to instantly throttle down flexible Alibaba batch jobs (while leaving GenAI spikes untouched).
	The Cooling Lever: Modulates the liquid cooling pumps, exploiting the thermal inertia of the water to reduce power without overheating the GPUs.
	The BESS Lever: Instantly charges or discharges the PySAM battery to smooth out any remaining power gaps.
4.3. The NeurIPS Evaluation Metric: The Tracking Reward
The primary metric to evaluate the AI in our NeurIPS benchmark is the Grid Tracking Score.
Reward=α×"Throughput"-β×∣"Grid Signal"-ΔP_datacenter∣-"Safety Penalties" 
If the grid asks the datacenter to drop 10MW, and the datacenter drops exactly 10MW using its 3 levers, it earns a massive reward. If it violates a thermal limit (>35^∘ C), the episode terminates with a heavy penalty.
________________________________________
5. Strategic Value and Outcomes
By achieving this "Energy System Handshake," we unlock massive value, positioning HPE and National Laboratories at the forefront of the energy transition:
1. Value to the Energy System (The Grid)
	Renewable Integration: Datacenters absorb excess wind/solar, preventing energy waste.
	Grid Stability: The datacenter acts as a massive "shock absorber" for the regional transmission grid, eliminating the need for utilities to turn on dirty fossil-fuel "peaker plants" during emergencies.
2. Value to Datacenters and HPE (The Business)
	New Revenue Streams: Grid operators (like PJM or CAISO) pay millions of dollars to facilities that can accurately track Frequency Regulation signals. This turns datacenter energy flexibility into a direct profit center.
	Faster Deployment: Utilities are currently blocking new datacenter construction due to grid strain. Demonstrating this "handshake" proves that HPE-powered datacenters help stabilize the grid, accelerating construction permits.
3. Value to AI Research (NeurIPS 2026)
	High-Impact Evaluation: Perfect fit for the 2026 Evaluations & Datasets track. It provides researchers with a highly realistic, cyber-physical environment to stress-test Multi-Agent and Hierarchical RL algorithms.
	Solving the AI-Energy Paradox: It moves the AI community away from isolated software benchmarks and connects machine learning directly to global infrastructure sustainability.
________________________________________
6. Technical Execution Plan: From Zero to NeurIPS Submission
This section outlines the rigorous, milestone-driven engineering plan required to build, test, and publish the benchmark.
Phase 1: Core Simulator Development
Goal: Build the independent physics and data simulators that represent the "Macro-Grid" environment.
	Step 1.1: Workload Orchestrator (Alibaba Traces)
	Data Used: Alibaba 2023 (Batch), 2025 (DLRM), and 2026 (GenAI) traces.
	Action: Write Python parsers to convert raw CSVs into a unified, synchronized time-series generator. Ensure the generator outputs P_base  (rigid load) and P_flex (schedulable load) at 2-second and 5-minute resolutions.
	Step 1.2: Thermal Twin (Liquid/Air Physics)
	Action: Implement the Exact Exponential ODEs for dual-zone cooling (Zone A: Liquid GPU, Zone B: Air CPU).
	Action: Calibrate the thermal capacitance (C) and heat rejection (K) coefficients to represent a 250MW facility. Ensure the equations remain numerically stable at 2-second integration steps.
	Step 1.3: Electrical & BESS Integration (NREL PySAM)
	Action: Wrap the nrel-pysam Li-ion NMC battery model to expose a simple charge/discharge API.
	Action: Implement the non-linear UPS and PDU loss curves to calculate total facility power (P_dc).
	Step 1.4: Macro-Grid Signal Generator (The "Handshake")
	Data Used: Historical PJM/NYISO Frequency Regulation signals (e.g., RegD) and Day-Ahead Locational Marginal Prices (LMP).
	Action: Build a stochastic signal generator that outputs the target ΔP_grid  the datacenter must match every 4 seconds.
Phase 2: Gymnasium Environment & HDRL Architecture
Goal: Fuse the simulators into a Gymnasium-compliant Reinforcement Learning environment and build the Hierarchical RL (HDRL) framework.
	Step 2.1: The Low-Level Environment (C2G-FastEnv)
	Action: Wrap the thermal, electrical, BESS, and workload simulators into a standard gymnasium.Env.
	Action: Define the 3D continuous Action Space: [throttle_batch, pump_speed, bess_dispatch].
	Action: Define the Observation Space: [current_temp, bess_soc, current_load, grid_signal_t].
	Action: Implement the High-Assurance Reward Function (Tracking error penalty + Thermal violation termination).
	Step 2.2: The High-Level Environment (C2G-MacroEnv)
	Action: Build the upper-level wrapper that steps every 15 minutes.
	Action: Define the Action Space: [commit_regulation_mw, baseline_power_target].
	Step 2.3: Baseline Agent Implementation (Stable Baselines 3)
	Action: Implement standard PPO and SAC agents for the low-level environment to prove the environment is solvable and mathematically sound.
Phase 3: Benchmark Design & Evaluation Tiers
Goal: Design the standardized "Challenges" that will define the NeurIPS E&D submission.
	Step 3.1: Define the Evaluation Scenarios (YAML Driven)
	Scenario A (The GenAI Crisis): A massive v2026 GenAI spike occurs simultaneously with a grid regulation signal asking the DC to drop power.
	Scenario B (The Thermal Squeeze): High ambient summer temperatures severely degrade cooling efficiency while the grid demands maximum Frequency Regulation participation.
	Scenario C (The Battery Drain): A long-duration grid anomaly forces the BESS to operate near its 10% minimum SOC limit.
	Step 3.2: Implement the Evaluation Metrics
	Action: Code the auditing scripts that calculate: Mean Time Between Failures (MTBF), Signal Tracking RMSE, and Total SLA Violations.
Phase 4: Experiments & Paper Writing
Goal: Generate the empirical results and draft the NeurIPS manuscript.
	Step 4.1: Execute Benchmark Baselines
	Action: Train PPO, SAC, and a classical Rule-Based Controller (MPC baseline) on the 3 Evaluation Scenarios across 10 random seeds.
	Action: Generate TensorBoard learning curves and evaluation tables.
	Step 4.2: Draft the Manuscript (NeurIPS LaTeX Template)
	Introduction: Define the AI-Energy Paradox and the Macro-Grid Handshake.
	Related Work: Contrast against Wang et al. (2019) and Fu et al. (2021) to highlight the HDRL and GenAI advancements.
	Methodology: Detail the ODE physics, PySAM integration, and the HDRL architecture.
	Experiments: Present the baseline results, highlighting where and why standard agents fail under catastrophic stress (The E&D "Audit" requirement).
	Step 4.3: Metadata and Reproducibility (The Croissant File)
	Action: Generate the mandatory metadata.json using the Croissant format, explicitly filling out the Responsible AI (RAI) fields regarding critical infrastructure simulation.
Phase 5: Open-Source Packaging & Submission
Goal: Prepare the codebase for double-blind review and submit to OpenReview.
	Step 5.1: Code Cleanup & Documentation
	Action: Finalize docstrings, type hinting, and the README.md.
	Action: Ensure uv sync and pytest run flawlessly on a fresh Linux/Mac environment.
	Step 5.2: Anonymization
	Action: Remove all references to HPE Labs, specific authors, and internal URLs from the code, logs, and LaTeX PDF to comply with the double-blind review policy.
	Step 5.3: Submission (May 6, 2026)
	Action: Upload the anonymized PDF, the Croissant metadata file, and the zipped codebase/dataset to the NeurIPS OpenReview portal.
________________________________________
7. Proposed Repository Folder Structure
To ensure the project meets the rigorous engineering standards expected by NeurIPS reviewers, the codebase will follow this modular architecture:

C2G-Macro/
├── pyproject.toml                       # uv/pip dependencies
├── README.md                            # Comprehensive setup and API guide
├── metadata.json                        # Mandatory Croissant format for NeurIPS
│
├── c2g_env/                             # The Core RL Environment
│   ├── env_low_level.py                 # 2-second physics step (Hardware Controller)
│   ├── env_high_level.py                # 15-minute market step (Macro Orchestrator)
│   ├── config.yaml                      # Scenario definitions (GenAI Crisis, Thermal Squeeze)
│   └── simulators/                      # The Physics Engines
│       ├── workload.py                  # Alibaba trace fusion & DVFS logic
│       ├── thermal.py                   # Exponential ODEs for liquid/air zones
│       ├── electrical.py                # Non-linear PUE and loss curves
│       ├── bess.py                      # NREL PySAM wrapper
│       └── macro_grid.py                # PJM/NYISO signal and price generator
│
├── data/                                # Curated Datasets
│   ├── raw/                             # Original Alibaba, PJM, and NOAA files
│   └── processed/                       # 2-second and 15-minute synchronized CSVs
│
├── preprocessing/                       # Data ingestion pipelines
│   ├── process_alibaba.py               # Trace synchronization and scaling
│   └── process_iso_signals.py           # PJM RegD signal extraction
│
├── baselines/                           # NeurIPS Evaluation Agents
│   ├── train_ppo.py                     # Stable Baselines 3 implementation
│   ├── train_sac.py                     # Continuous control baseline
│   └── rule_based_mpc.py                # Classical control baseline (for comparison)
│
├── evaluation/                          # Auditing and Metrics (The E&D Hook)
│   ├── run_benchmark.py                 # Executes the 3 stress-test scenarios
│   └── generate_plots.py                # Outputs publication-ready PDFs
│
└── tests/                               # High-Assurance Validation
    ├── test_physics_conservation.py     # Ensures energy/heat equations balance
    ├── test_bess_limits.py              # Validates PySAM SOC constraints
    └── test_gym_api.py                  # Validates RL environment compliance

In total there are 7 sections added.