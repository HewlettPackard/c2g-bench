# High-Assurance (HA) Benchmark — C2G-Bench

> **61 / 61 tests passing** (`tests/test_ha_safety.py`)

This module adds a comprehensive **High-Assurance safety benchmark** to C2G-Bench, organised into three tiers of increasing sophistication.  Every method enforces the same five hard constraints (C1–C5) defined by the C2G data-centre environment, but differs in *how* safety is achieved and what guarantees it provides.

---

## Hard Constraints (C1–C5)

All benchmarks protect against these non-negotiable physical limits:

| ID | Constraint | Threshold | Physical Meaning |
|----|-----------|-----------|-----------------|
| C1 | T_A < T_safe | 35 °C | Server-room-A silicon thermal limit |
| C2 | T_B < T_safe | 35 °C | Server-room-B silicon thermal limit |
| C3 | SOC ∈ [SOC_min, SOC_max] | [0.10, 0.95] | BESS operational envelope |
| C4 | \|Δf\| < 0.5 Hz | 0.5 Hz | Under/Over Frequency Load Shedding protection |
| C5 | V_pcc > V_min | 0.90 pu | Under-voltage relay threshold |

---

## Tier 1 — Hard-Guarantee Methods (Provable Safety)

These methods provide **formal or mathematically grounded** safety guarantees.  At runtime they solve an optimisation problem (or look up a precomputed solution) to find the closest safe action to the RL agent's proposal.

### 1.1  Simplex Architecture Safety Shield

| | |
|---|---|
| **File** | `baselines/safety_shield.py` |
| **Training script** | `baselines/train_shielded_ppo.py` |
| **Config** | `conf/algo/shielded_ppo.yaml` |
| **Sweep phase** | Phase 2 (existing PPO) + runtime shield at eval |
| **Reference** | Sha, L. "Using Simplicity to Control Complexity." IEEE Software, 2001 |

**Description:**  A Simplex-style runtime safety filter.  The RL agent (complex controller) proposes an action; the shield checks whether it could violate any hard constraint within a worst-case one-step lookahead using analytic thermal-ODE bounds.  If unsafe, it overrides with the closest safe action (baseline controller).

**Key properties:**
- O(1) per step — no optimisation solver, no rollouts
- Works with ANY agent (PPO, SAC, rule-based, random)
- Zero training required for the shield itself
- Provable safety guarantee under model assumptions (thermal time-constant bounds)

**Verification checklist:**
- [ ] `SafetyShield.filter(action, obs)` returns `(safe_action, was_modified, info)`
- [ ] All 5 constraints (C1–C5) are checked independently
- [ ] Analytic worst-case bounds use thermal time constants `τ_A`, `τ_B`
- [ ] SOC guard bands prevent boundary crossing
- [ ] Stats track `total_steps`, `overrides`, `intervention_rate`

---

### 1.2  Control Barrier Function (CBF) Shield

| | |
|---|---|
| **File** | `baselines/safety/cbf_shield.py` |
| **Training script** | `baselines/train_cbf_ppo.py` |
| **Config** | `conf/algo/cbf_ppo.yaml` |
| **Sweep phase** | Phase 12 |
| **Reference** | Ames, A. et al. "Control Barrier Functions: Theory and Applications." ECC 2019 |

**Description:**  Defines 6 barrier functions (one per constraint, with SOC split into upper/lower) of the form `h(x) ≥ 0 ⟹ safe`.  At each step, solves a Quadratic Program (QP) via scipy SLSQP to find the closest action to the RL proposal that satisfies the CBF decay condition `ḣ(x,u) + α·h(x) ≥ 0` for all barriers simultaneously.

**Barrier functions:**
| Barrier | Formula | Constraint |
|---------|---------|------------|
| `h_thermal_A` | `(T_safe − margin)² − T_A²` | C1 |
| `h_thermal_B` | `(T_safe − margin)² − T_B²` | C2 |
| `h_soc_low` | `SOC − SOC_min − margin` | C3 lower |
| `h_soc_high` | `SOC_max + margin − SOC` | C3 upper |
| `h_frequency` | `freq_max² − Δf²` | C4 |
| `h_voltage` | `V_pcc − V_min − margin` | C5 |

**Key properties:**
- QP solved at every step (scipy SLSQP, ~1–5 ms)
- Configurable margin (`cbf_margin`, default 0.5)
- Class-K function α(h) = α·h enforces exponential decay towards safe set
- Falls back to greedy projection if QP fails to converge

**Verification checklist:**
- [ ] `CBFShield.__init__` sets 6 barrier functions matching C1–C5
- [ ] `CBFShield._barrier_values(state)` returns dict with all 6 barrier values
- [ ] `CBFShield.filter(action, obs)` solves QP via `scipy.optimize.minimize(method='SLSQP')`
- [ ] QP objective: `‖u − u_proposed‖²` (minimal intervention)
- [ ] QP constraints: `ḣ(x,u) + α·h(x) ≥ 0` for each barrier
- [ ] `CBFShieldedEnv` wraps gymnasium env, applies filter in `step()`
- [ ] `CBFStats` tracks `total_steps`, `interventions`, `qp_failures`, `mean_barrier_min`

---

### 1.3  Hamilton-Jacobi (HJ) Reachability Shield

| | |
|---|---|
| **File** | `baselines/safety/hj_shield.py` |
| **Training script** | `baselines/train_hj_ppo.py` |
| **Config** | `conf/algo/hj_ppo.yaml` |
| **Sweep phase** | Phase 13 |
| **Reference** | Bansal, S. et al. "Hamilton-Jacobi Reachability: A Brief Overview and Recent Advances." CDC 2017 |

**Description:**  Precomputes a backward-reachable set (BRS) offline via dynamic programming on a grid, producing a value function `V(x)` where `V(x) ≥ 0` means safe.  At runtime, if the current state has `V(x) < δ` (near the safety boundary), the shield overrides the RL action with the precomputed optimal safe control `u*(x)` from the BRS computation.

**Offline computation (two projected subsystems):**
- **Thermal BRS:** 2D grid (T_A or T_B × pump_speed), `n_grid × n_grid` points.  Value iteration with `V(x) = min(ℓ(x), max_u min_d V_next)` where `ℓ(x) = T_safe − T − margin`.
- **SOC BRS:** 1D grid (SOC × bess_dispatch).  Constraint function `ℓ(x) = min(SOC − SOC_min, SOC_max − SOC)`.

**Key properties:**
- Offline DP: O(n_grid² · n_actions) one-time cost
- Runtime: O(1) grid lookup + linear interpolation
- Smooth blending: `α = max(0, 1 − V(x)/δ)`, safe action = `α·u_safe + (1−α)·u_proposed`
- Configurable `delta` (safety margin), `n_grid` (resolution)

**Verification checklist:**
- [ ] `HJShield.__init__` with `precompute=True` populates `V_thermal`, `V_soc`, `U_thermal`, `U_soc`
- [ ] `_compute_thermal_brs()` runs value iteration on thermal grid
- [ ] `_compute_soc_brs()` runs value iteration on SOC grid
- [ ] `_lookup_thermal(T, T_amb)` returns `(value, safe_pump_speed)` via grid interpolation
- [ ] `_lookup_soc(soc)` returns `(value, safe_bess_dispatch)`
- [ ] `filter()` blends using α when V < δ, passes through when V ≥ δ
- [ ] `HJStats` tracks `total_steps`, `thermal_overrides`, `soc_overrides`, `min_value_function`

---

### 1.4  Model-Predictive Safety Filter (MPC-SF)

| | |
|---|---|
| **File** | `baselines/safety/mpc_safety_filter.py` |
| **Training script** | `baselines/train_mpcsf_ppo.py` |
| **Config** | `conf/algo/mpcsf_ppo.yaml` |
| **Sweep phase** | Phase 14 |
| **Reference** | Wabersich, K. and Zeilinger, M. "Linear Model Predictive Safety Certification for Learning-Based Control." CDC 2018 |

**Description:**  Solves a receding-horizon nonlinear program (NLP) at each step.  The NLP finds the action sequence `u_0, …, u_{H−1}` closest to the RL proposal that keeps the predicted state trajectory within the safe set over the full horizon H.  Uses a simplified thermal + BESS dynamics model for forward prediction.

**NLP formulation:**
- **Objective:** `min ‖u_0 − u_proposed‖² + Σ_{t=1}^{H-1} 0.01·‖u_t‖²`
- **Constraints (∀t):** `T_A(t) ≤ T_safe − margin`, `T_B(t) ≤ T_safe − margin`, `SOC_min + margin ≤ SOC(t) ≤ SOC_max − margin`
- **State prediction:** Simplified thermal ODE (`T_next = T + dt·(Q_gen − Q_cool)/C_th`) + linear BESS model

**Key properties:**
- Skip-when-safe: If `_check_needs_filter()` returns False (all states far from limits), NLP is not solved → O(1)
- Horizon H configurable (default 5 steps = 25 minutes lookahead)
- scipy SLSQP solver, ~5–50 ms when active
- Falls back to per-dimension clamping if NLP fails

**Verification checklist:**
- [ ] `MPCSafetyFilter.__init__` sets `horizon`, `margin`, thermal parameters
- [ ] `_check_needs_filter(obs)` returns True only when state is near constraint boundary
- [ ] `_predict_state(state, action)` implements simplified thermal + BESS dynamics
- [ ] `_solve_mpc_nlp(obs, proposed)` solves H-step NLP via scipy
- [ ] `filter()` skips NLP when safe, solves when needed
- [ ] `MPCSFStats` tracks `total_steps`, `nlp_solves`, `nlp_failures`, `mean_solve_time_ms`

---

## Tier 2 — Soft-Guarantee Methods (Statistical Safety)

These methods enforce safety **during training** through constrained optimisation or reward shaping.  They do NOT provide hard runtime guarantees — a trained policy may still occasionally violate constraints — but they statistically reduce violation rates.

### 2.1  PPO-Lagrangian

| | |
|---|---|
| **File** | `baselines/train_ppo_lagrangian.py` |
| **Config** | `conf/algo/ppo_lagrangian.yaml` |
| **Sweep phase** | Phase 11 |
| **Reference** | Ray, A. et al. "Benchmarking Safe Exploration in Deep Reinforcement Learning." 2019 |

**Description:**  Standard PPO augmented with a Lagrangian relaxation of per-constraint cost budgets.  Each constraint `Cᵢ` has a cost function `cᵢ(s,a)` (1 if violated, 0 otherwise) and a budget `dᵢ`.  Dual variables `λᵢ` are updated via gradient ascent on the constraint violation.  The augmented objective is `max_π min_λ E[R − Σᵢ λᵢ·(cᵢ − dᵢ)]`.

**Key properties:**
- Separate cost heads for thermal, SOC, frequency, voltage constraints
- Lagrange multipliers updated every rollout with configurable `lr_lambda`
- Cost budgets per constraint (e.g., `thermal_budget=0.05` allows 5% violation rate)
- No runtime shield — relies entirely on learned policy

**Verification checklist:**
- [ ] `CostWrapper` gymnasium wrapper computes per-constraint cost signals in `info`
- [ ] `LagrangianCallback` maintains and updates `λ` per constraint
- [ ] Augmented reward: `r_aug = r − Σ λᵢ·cᵢ` passed to PPO
- [ ] λ update: `λᵢ ← max(0, λᵢ + lr_λ·(mean_cᵢ − dᵢ))`

---

### 2.2  Constrained Policy Optimisation (CPO)

| | |
|---|---|
| **File** | `baselines/train_cpo.py` |
| **Config** | `conf/algo/cpo.yaml` |
| **Sweep phase** | Phase 15 |
| **Reference** | Achiam, J. et al. "Constrained Policy Optimization." ICML 2017 |

**Description:**  Approximates CPO on top of SB3's PPO.  A `CPOCostWrapper` computes per-step costs for each constraint.  A `CPOUpdateCallback` adjusts the Lagrange multipliers with adaptive step sizing: aggressive correction when violation exceeds 5%, gradual decay when within budget.  Multipliers are capped at `λ_max=5.0`.

**Key properties:**
- Tighter budgets than PPO-Lagrangian (thermal: 0.02, soc: 0.05, freq: 0.02)
- Adaptive λ step: `Δλ = lr · (violation − budget)`, with 2× acceleration when over budget
- λ exponential decay (×0.995) when within budget → avoids over-conservatism
- `CPOCostWrapper` computes binary cost per constraint per step

**Verification checklist:**
- [ ] `CPOCostWrapper` adds `cost_thermal`, `cost_soc`, `cost_freq`, `cost_voltage` to info
- [ ] `CPOUpdateCallback._on_rollout_end()` updates λ from rollout buffer costs
- [ ] Adaptive correction: `lr × 2` when `violation > budget + 0.05`
- [ ] Decay: `λ *= 0.995` when `violation < budget`
- [ ] Cap: `λ = min(λ, λ_max)` with `λ_max=5.0`
- [ ] Augmented reward: `r − Σ λᵢ·cᵢ`

---

### 2.3  Shield Reward Shaping

| | |
|---|---|
| **File** | `baselines/train_shield_reward_shaping.py` |
| **Config** | `conf/algo/shield_reward_shaping.yaml` |
| **Sweep phase** | Phase 16 |
| **Reference** | Inspired by Fulton, N. and Platzer, A. "Safe Reinforcement Learning via Formal Methods." AAAI 2018 |

**Description:**  Combines a fixed Simplex safety shield (applied during training) with continuous reward penalties that shape the policy towards inherently safe behaviour.  The shield prevents catastrophic failures, while the shaped rewards teach the policy to avoid needing the shield.

**Reward shaping components:**
| Component | Formula | Weight |
|-----------|---------|--------|
| Thermal penalty | `−w_th · max(0, T − T_warn)²` | `w_thermal=2.0` |
| SOC penalty | `−w_soc · max(0, SOC_min+g − SOC, SOC − SOC_max+g)²` | `w_soc=1.0` |
| Frequency penalty | `−w_freq · max(0, \|Δf\| − 0.3)²` | `w_freq=1.0` |
| Voltage penalty | `−w_volt · max(0, V_warn − V)²` | `w_volt=0.5` |
| Shield intervention | `−w_shield` (discrete, per override) | `w_shield=0.5` |

**Key properties:**
- Quadratic penalties → smooth gradients near constraint boundaries
- Shield intervention penalty teaches agent to self-avoid needing the shield
- Shield is active during training (prevents unsafe exploration)
- At evaluation, shield can be removed to test learned safety

**Verification checklist:**
- [ ] `ShieldRewardShapingWrapper` applies Simplex shield in `step()`
- [ ] Shaped reward = `original_reward + thermal_pen + soc_pen + freq_pen + volt_pen + shield_pen`
- [ ] Quadratic penalty functions with configurable weights
- [ ] Shield penalty: `−w_shield` if shield modified the action
- [ ] Penalty info logged in `info` dict for debugging

---

## Tier 3 — Neuro-Symbolic / Interpretable High-Assurance (HA-C2G)

This is the flagship method, porting the 3-layer architecture from our SC'26 paper (HA-CompOpt) to C2G-Bench.

### 3.1  HA-C2G: Concept Bottleneck + Safe Projection + Physics Shield

| | |
|---|---|
| **Files** | `baselines/safety/concept_bottleneck.py` (Layer 1) |
| | `baselines/safety/safe_projection.py` (Layer 2) |
| | `baselines/safety/proof_tree.py` (Audit) |
| | `baselines/train_ha_c2g.py` (Training) |
| **Config** | `conf/algo/ha_c2g.yaml` |
| **Sweep phase** | Phase 17 |
| **Reference** | Guillant, G. et al. "HA-CompOpt: High-Assurance Neuro-Symbolic Framework." SC 2026 |

---

#### Layer 1 — Concept Bottleneck Model (CBM)

**File:** `baselines/safety/concept_bottleneck.py`

**Description:**  A differentiable neural encoder maps the raw 17-D observation to 10 interpretable safety concepts in [0,1].  Ground-truth concept labels are derived from the observation for supervised concept loss (decaying weight: 1.0 → 0.1 over training).

**10 C2G Concepts:**

| # | Concept | Formula | Interpretation |
|---|---------|---------|----------------|
| 1 | `thermal_margin_A` | `clip((T_safe − T_A) / (T_safe − 20), 0, 1)` | How far T_A is from limit |
| 2 | `thermal_margin_B` | `clip((T_safe − T_B) / (T_safe − 20), 0, 1)` | How far T_B is from limit |
| 3 | `soc_health` | `clip(min(SOC − SOC_min, SOC_max − SOC) / 0.4, 0, 1)` | SOC distance from nearest bound |
| 4 | `freq_stability` | `clip(1 − \|Δf\| / 0.5, 0, 1)` | Distance from frequency limit |
| 5 | `voltage_margin` | `clip((V_pcc − 0.90) / 0.10, 0, 1)` | Voltage headroom above UV relay |
| 6 | `cooling_demand_A` | `clip((T_A − (T_warn−2)) / 4, 0, 1) + 0.3·spike` | Cooling urgency for room A |
| 7 | `cooling_demand_B` | `clip((T_B − (T_warn−2)) / 4, 0, 1)` | Cooling urgency for room B |
| 8 | `grid_urgency` | `clip(\|regd_signal\|, 0, 1)` | Grid regulation demand |
| 9 | `batch_pressure` | `clip(backlog / 1.5, 0, 1)` | Compute workload pressure |
| 10 | `bess_headroom` | `clip(min(SOC − SOC_min, SOC_max − SOC) / 0.3, 0, 1)` | Bidirectional BESS room |

**Neural encoder architecture:**
- Input: 17-D observation
- Hidden: 64 → 64 (ReLU)
- Output: 10-D (Sigmoid) → concepts ∈ [0,1]

**Verification checklist:**
- [ ] `C2GConcepts.from_obs(obs)` computes all 10 ground-truth concepts
- [ ] `C2GConcepts.to_vector()` returns float32 array of shape (10,)
- [ ] All concepts ∈ [0, 1] for any valid observation
- [ ] `C2GConceptEncoder(obs_dim=17, n_concepts=10)` is an `nn.Module`
- [ ] Encoder output uses Sigmoid activation → bounded [0,1]
- [ ] `C2GConceptFeatureExtractor` outputs `[obs; concepts]` (dim=27)
- [ ] `C2GGatedConceptFeatureExtractor` outputs `[obs; concepts; gate_values]` (dim=31)

---

#### Layer 2 — Safe Projection Gate (**Actively Trained**)

**File:** `baselines/safety/safe_projection.py`

**Description:**  A concept-conditioned gate that attenuates each action dimension based on the current safety concepts.  **This gate is actively trained** with an auxiliary supervision loss — it is NOT a pass-through or hand-coded rule.

**Gate architecture:**
- Input: 10 concepts
- Hidden: 32 (ReLU)
- Output: 4 gate values ∈ (0, 1) via Sigmoid
- Init bias = 2.0 → Sigmoid(2.0) ≈ 0.88 at init (near pass-through, avoids early training collapse)

**Gate supervision targets:**
| Action | Gate target `g*` | Meaning |
|--------|------------------|---------|
| `throttle_batch` | `1 − α · max(cooling_demand_A, cooling_demand_B)` | Reduce batch when cooling is strained |
| `pump_speed_A` | `1.0` (always allow) | Pumps are always safe |
| `hvac_effort` | `1.0` (always allow) | HVAC is always safe |
| `bess_dispatch` | `1 − β · (1 − bess_headroom)` | Restrict BESS when near SOC bounds |

Where `α=0.5` (gate_alpha) and `β=0.3` (gate_beta) are configurable.

**Action attenuation:** `gated_action = raw_action × gate(concepts)` (applied in `HAC2GShieldWrapper._apply_gate()` before shielding)

**Joint training:** `ConceptAndGateSupervisionCallback` runs every `train_freq` steps:
1. Sample mini-batch from rollout buffer
2. Forward pass: obs → encoder → concepts → gate → gate_values
3. Compute concept loss: `MSE(predicted_concepts, ground_truth_concepts)` × decaying weight
4. Compute gate loss: `MSE(gate_values, gate_targets)`
5. Total auxiliary loss: `concept_loss + gate_loss`
6. Update encoder + gate parameters with dedicated Adam optimizer

**Verification checklist:**
- [ ] `SafeProjectionGate(concept_dim=10, action_dim=4)` outputs gate ∈ (0,1)⁴
- [ ] Gate init bias = 2.0 → near pass-through at start of training
- [ ] `GateSupervisionLoss.compute(gate_values, concept_targets)` returns scalar loss
- [ ] Gate targets for throttle use `α·max(cooling_demand_A, cooling_demand_B)`
- [ ] Gate targets for BESS use `β·(1 − bess_headroom)`
- [ ] `HAC2GShieldWrapper._apply_gate()` multiplies raw action by gate values
- [ ] `ConceptAndGateSupervisionCallback` jointly trains encoder AND gate
- [ ] Auxiliary loss = concept_loss × decaying_weight + gate_loss

---

#### Layer 3 — Physics Rule Shield (with Reward Shaping)

**File:** `baselines/train_ha_c2g.py` (`HAC2GShieldWrapper`)

**Description:**  The Simplex safety shield is applied as the final layer, providing hard guarantees.  Critically, **the shield also shapes the reward** during training: each time the shield overrides an action, a penalty of `−shield_penalty` (default 0.5) is added to the reward.  This teaches the concept encoder and gate to produce actions that don't need the shield.

**Shield-in-the-loop training flow:**
```
obs → encoder → concepts → gate(concepts) × raw_action → shield.filter() → env.step()
                                                               ↓
                                                       if modified: reward -= 0.5
```

**Verification checklist:**
- [ ] `HAC2GShieldWrapper` applies `SafetyShield.filter()` in `step()`
- [ ] Shield penalty: `reward -= shield_penalty` when shield modifies action
- [ ] `info["shield_active"]` is `True` when shield modified the action
- [ ] `info["gate_applied"]` is `True` when concept gate is active
- [ ] `info["proof_tree"]` contains serialised proof tree for auditability
- [ ] Training uses `C2GGatedConceptFeatureExtractor` as policy feature extractor
- [ ] `ConceptGateSupervisionCallback` trains encoder+gate every `train_freq` steps
- [ ] Concept supervision weight decays from 1.0 → 0.1 over training
- [ ] Saves `concept_encoder.pt` and `safety_gate.pt` alongside the PPO model

---

#### Audit — Hierarchical Proof Trees

**File:** `baselines/safety/proof_tree.py`

**Description:**  Every step produces a hierarchical proof tree documenting the safety status.  The tree mirrors the 3-layer architecture: system-level → subsystem constraints → individual checks.

**Tree structure:**
```
SYSTEM_SAFE
├── THERMAL_OK
│   ├── C1: T_A < 35°C  [PASS/FAIL, severity]
│   └── C2: T_B < 35°C  [PASS/FAIL, severity]
├── BESS_OK
│   └── C3: SOC ∈ [0.10, 0.95]  [PASS/FAIL, severity]
├── GRID_OK
│   ├── C4: |Δf| < 0.5Hz  [PASS/FAIL, severity]
│   └── C5: V_pcc > 0.90  [PASS/FAIL, severity]
└── CONCEPT_OK  (optional, when concepts provided)
    ├── thermal_margin_A: 0.82  [OK/WARN]
    ├── soc_health: 0.91  [OK/WARN]
    └── ...
```

**Verification checklist:**
- [ ] `ProofTree.from_step(obs, raw_action, safe_action, concepts=...)` builds tree
- [ ] `ProofNode` has `name`, `status` (PASS/FAIL/WARN), `severity`, `evidence`, `children`
- [ ] `RuleStatus` enum: PASS, FAIL, WARN
- [ ] `tree.is_safe` → True iff root status is PASS
- [ ] `tree.n_failures` → count of FAIL nodes
- [ ] `tree.to_dict()` → JSON-serialisable dict
- [ ] `tree.summary()` → human-readable multi-line string

---

## Evaluation Metrics (11 total)

The HA benchmark evaluates all methods on 11 metrics:

| # | Metric | Column Name | Direction | Description |
|---|--------|-------------|-----------|-------------|
| 1 | Mean Reward | `mean_reward` | ↑ higher is better | Average step reward over episode |
| 2 | Total Reward | `total_reward` | ↑ | Cumulative reward |
| 3 | Tracking RMSE | `tracking_rmse` | ↓ lower is better | RMSE of grid regulation tracking error (kW) |
| 4 | Thermal Violation Rate | `thermal_viol_rate` | ↓ | Fraction of steps with T > T_warn (33°C) |
| 5 | Throughput Ratio | `throughput_ratio` | ↑ | Mean flex workload / nominal capacity |
| 6 | Survival Rate | `survival_rate` | ↑ | Fraction of episodes that reach 288 ticks |
| 7 | **Hard Violation Rate** | `hard_violation_rate` | ↓ | Fraction of steps violating ANY of C1–C5 |
| 8 | **Shield Intervention Rate** | `shield_intervention_rate` | ↓ | Fraction of steps where shield modified action |
| 9 | **Constraint Margin** | `constraint_margin` | ↑ | Mean distance to nearest constraint boundary |
| 10 | **Worst-Case Margin** | `worst_case_margin` | ↑ | Minimum constraint margin over entire episode |
| 11 | **Computational Overhead** | `computational_overhead_ms` | ↓ | Mean wall-clock time per shield decision (ms) |

Metrics 7–11 (bold) are new HA-specific metrics not in the base C2G-Bench.

---

## Tier 3 Ablations

These ablation baselines isolate the contribution of each layer in the HA-C2G stack.  They are essential for NeurIPS reviewers to assess the neuro-symbolic claim.

| Ablation | Layers | Safety Guarantee | Research Question |
|----------|--------|-----------------|-------------------|
| `cbm_only` | CBM only | None | Does interpretability alone improve safety? |
| `cbm_gate` | CBM + Trained Gate | Soft (learned) | Does the trained gate reduce violations without a hard shield? |
| `cbm_shield` | CBM + Shield | Hard (Simplex) | Does the gate add value when the hard shield is already present? |
| `ha_c2g` | CBM + Gate + Shield | Hard (Simplex) | Full stack — does the combination outperform each part? |

### A.1  CBM-Only (Concept Bottleneck without Gate or Shield)

| | |
|---|---|
| **File** | `baselines/train_cbm_only.py` |
| **Config** | `conf/algo/cbm_only.yaml` |
| **Sweep phase** | Phase 18 |

**Description:**  Standard PPO with a concept bottleneck feature extractor.  The policy receives `[obs; concepts]` (dim=27) as features, where concepts are predicted by a supervised neural encoder.  No gate attenuates actions; no shield filters them.  This tests whether interpretable features alone help the policy learn safer behaviour.

**Verification checklist:**
- [ ] Uses `C2GConceptFeatureExtractor` (NOT gated), output dim = 27
- [ ] `ConceptSupervisionCallback` trains encoder with decaying MSE loss
- [ ] No `SafetyShield` wrapper, no `HAC2GShieldWrapper`
- [ ] No `SafeProjectionGate` anywhere in the pipeline
- [ ] Same PPO hyperparameters as `ha_c2g` for fair comparison

### A.2  CBM+Gate (Concept Bottleneck + Trained Gate, no Shield)

| | |
|---|---|
| **File** | `baselines/train_cbm_gate.py` |
| **Config** | `conf/algo/cbm_gate.yaml` |
| **Sweep phase** | Phase 19 |

**Description:**  PPO with concept bottleneck AND the actively trained safe projection gate, but NO physics shield.  The gate attenuates actions based on safety concepts (throttle reduced when cooling demand is high, BESS restricted when headroom is low).  This tests whether the learned gate alone can statistically reduce violations.

**Verification checklist:**
- [ ] Uses `C2GGatedConceptFeatureExtractor`, output dim = 31
- [ ] `ConceptGateSupervisionCallback` jointly trains encoder AND gate
- [ ] Gate targets match HA-C2G: throttle uses α·cooling_demand, BESS uses β·(1−headroom)
- [ ] No `SafetyShield` wrapper, no `HAC2GShieldWrapper`
- [ ] No shield penalty in reward
- [ ] Same PPO hyperparameters as `ha_c2g` for fair comparison

---

### A.3  CBM+Shield (Concept Bottleneck + Physics Shield, no Gate)

| | |
|---|---|
| **File** | `baselines/train_cbm_shield.py` |
| **Config** | `conf/algo/cbm_shield.yaml` |
| **Sweep phase** | Phase 20 |

**Description:**  PPO with concept bottleneck feature extractor AND the Simplex physics shield (shield-in-the-loop with reward penalty), but NO trained safety gate.  This isolates whether the gate contributes beyond what the hard shield already provides.

**Verification checklist:**
- [ ] Uses `C2GConceptFeatureExtractor` (NOT gated), output dim = 27
- [ ] `HAC2GShieldWrapper` wraps the env (shield active during training)
- [ ] Shield penalty `−0.5` per override shapes reward
- [ ] `ConceptSupervisionCallback` trains encoder with decaying MSE loss
- [ ] No `SafeProjectionGate` anywhere in the pipeline
- [ ] Same PPO hyperparameters as `ha_c2g` for fair comparison

---

## File Map

```
baselines/
├── safety/
│   ├── __init__.py                  # Package init, conditional torch imports
│   ├── cbf_shield.py                # Tier 1: Control Barrier Function
│   ├── hj_shield.py                 # Tier 1: Hamilton-Jacobi Reachability
│   ├── mpc_safety_filter.py         # Tier 1: MPC Safety Filter
│   ├── concept_bottleneck.py        # Tier 3, Layer 1: Concept encoder
│   ├── safe_projection.py           # Tier 3, Layer 2: Trained gate
│   ├── proof_tree.py                # Tier 3: Audit proof trees
│   └── README.md                    # ← This file
├── safety_shield.py                 # Tier 1: Simplex shield (pre-existing)
├── train_shielded_ppo.py            # Simplex-shielded PPO (pre-existing)
├── train_ppo_lagrangian.py          # Tier 2: PPO-Lagrangian (pre-existing)
├── train_cbf_ppo.py                 # Tier 1: CBF-PPO training
├── train_hj_ppo.py                  # Tier 1: HJ-PPO training
├── train_mpcsf_ppo.py               # Tier 1: MPC-SF-PPO training
├── train_cpo.py                     # Tier 2: CPO training
├── train_shield_reward_shaping.py   # Tier 2: Shield reward shaping
└── train_ha_c2g.py                  # Tier 3: HA-C2G full stack
└── train_cbm_only.py                # Tier 3 Ablation: CBM only
└── train_cbm_gate.py                # Tier 3 Ablation: CBM + gate (no shield)
└── train_cbm_shield.py              # Tier 3 Ablation: CBM + shield (no gate)

conf/algo/
├── cbf_ppo.yaml
├── hj_ppo.yaml
├── mpcsf_ppo.yaml
├── cpo.yaml
├── shield_reward_shaping.yaml
└── ha_c2g.yaml

evaluation/
├── run_benchmark.py                 # Updated: +7 HA agent loaders
├── run_ha_benchmark.py              # HA-specific benchmark (11 metrics)
└── generate_ha_plots.py             # Pareto, radar, violin, bars, LaTeX

scripts/
└── run_sweep.sh                     # Updated: +Phases 12–18 for HA

tests/
└── test_ha_safety.py                # 61 tests (CBF, HJ, MPC, concepts,
                                     #   gate, proof tree, cross-shield)
```

---

## Common API

All Tier 1 shields implement the same interface for drop-in comparison:

```python
class Shield:
    def filter(self, action: np.ndarray, obs: np.ndarray
              ) -> tuple[np.ndarray, bool, dict]:
        """
        Returns: (safe_action, was_modified, info_dict)
        """
        ...

    def reset(self) -> None:
        """Reset per-episode statistics."""
        ...
```

This is verified by the `TestCrossShieldConsistency` parametrised test suite, which runs all 4 shields (Simplex, CBF, HJ, MPC-SF) through identical API contract tests.

---

## Running

```bash
# Run tests
export PATH="/tmp/compopt_venv/bin:$PATH"
python3 -m pytest tests/test_ha_safety.py -v

# Train a single HA method
python3 baselines/train_cbf_ppo.py scenario=default experiment.seed=42

# Run HA benchmark evaluation
python3 evaluation/run_ha_benchmark.py --agents simplex cbf hj mpcsf --n_episodes 5

# Full sweep (all methods × all scenarios × 3 seeds)
bash scripts/run_sweep.sh

# Generate plots
python3 evaluation/generate_ha_plots.py --csv results/ha_benchmark_results.csv
```
