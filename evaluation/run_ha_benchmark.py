"""
evaluation/run_ha_benchmark.py  —  High-Assurance Benchmark Evaluation
=======================================================================
Evaluates all high-assurance safety controllers on all scenarios,
collecting the extended 11-metric set designed for the HA benchmark.

Standard metrics (from run_benchmark.py):
  1. mean_reward           — mean step reward
  2. tracking_rmse         — RMSE of regulation signal tracking error
  3. thermal_viol_rate     — fraction of ticks with T > T_warn
  4. throughput_ratio      — mean batch compute utilisation
  5. bess_degradation      — cumulative battery ageing
  6. survival_rate         — fraction of full-length episodes

HA-specific metrics (new):
  7.  hard_violation_rate  — fraction of steps with ANY C1-C5 breach
  8.  shield_intervention_rate — fraction of steps where shield modified action
  9.  constraint_margin    — mean min distance to nearest constraint boundary
  10. worst_case_margin    — global minimum margin across all constraints
  11. computational_overhead_ms — wall-clock time per filter() call

HA Agents
---------
  simplex_ppo     — Simplex shield + PPO (existing baseline)
  cbf_ppo         — CBF safety filter + PPO
  hj_ppo          — Hamilton-Jacobi reachability + PPO
  mpcsf_ppo       — MPC Safety Filter + PPO
  ppo_lagrangian  — PPO-Lagrangian (constrained RL)
  cpo             — Constrained Policy Optimisation
  reward_shaping  — Shield reward shaping + PPO
  ha_c2g          — Full neuro-symbolic HA (CBM + Gate + Shield)

Usage
-----
  uv run python evaluation/run_ha_benchmark.py
  uv run python evaluation/run_ha_benchmark.py --agents cbf_ppo ha_c2g
  uv run python evaluation/run_ha_benchmark.py --scenarios default scenario_b
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from c2g_env import C2GFastEnv
from baselines.metrics_callback import C2GTransitionLoggerCallback, build_ablation_suffix
from baselines.safety_shield import SafetyShield
from baselines.safety.cbf_shield import CBFShield
from baselines.safety.hj_shield import HJShield
from baselines.safety.mpc_safety_filter import MPCSafetyFilter
from baselines.safety.safe_projection import compute_layer2_action

SCENARIOS = ["default", "scenario_a", "scenario_b", "scenario_c"]
T_WARN_NORM = 33.0 / 35.0
T_SAFE = 35.0
SOC_MIN = 0.10
SOC_MAX = 0.95
_PPO_LIKE_AGENT_KEYS = {
    "shielded_ppo",
    "cbf_ppo",
    "hj_ppo",
    "mpcsf_ppo",
    "ppo_lagrangian",
    "cpo",
    "shield_reward_shaping",
    "ha_c2g",
    "cbm_only",
    "cbm_gate",
    "cbm_shield",
}
_VALID_ACTIONS = ("throttle_batch", "pump_speed_A", "hvac_effort", "bess_dispatch")
_ACTION_BOUNDS: dict[str, tuple[float, float]] = {
    "throttle_batch": (0.0, 1.0),
    "pump_speed_A": (0.0, 1.0),
    "hvac_effort": (0.0, 1.0),
    "bess_dispatch": (-1.0, 1.0),
}


def _make_env(
    scenario: str,
    fixed_action_values: dict[str, float] | None = None,
    **kwargs,
) -> C2GFastEnv:
    """Return ActionAblationFastEnv when fixed-action overrides are active."""
    if fixed_action_values:
        from c2g_env.experiments.action_ablation_env import ActionAblationFastEnv
        return ActionAblationFastEnv(
            scenario=scenario,
            fixed_action_values=fixed_action_values,
            **kwargs,
        )
    return C2GFastEnv(scenario=scenario, **kwargs)

# Obs indices
_I_TEMP_A   = 0
_I_TEMP_B   = 1
_I_SOC      = 2
_I_FREQ_DEV = 14
_I_VPCC     = 15


def _parse_fixed_action_args(values: list[str] | None) -> dict[str, float]:
    if not values:
        return {}

    fixed_action_values: dict[str, float] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(
                f"Invalid --fixed-action '{item}'. Expected format action=value."
            )
        action_name, raw_value = item.split("=", 1)
        action_name = action_name.strip()
        if action_name not in _ACTION_BOUNDS:
            valid = ", ".join(_VALID_ACTIONS)
            raise ValueError(
                f"Invalid action '{action_name}' in --fixed-action. Valid actions: {valid}"
            )
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise ValueError(
                f"Invalid fixed value for action '{action_name}': {raw_value}"
            ) from exc
        if not np.isfinite(value):
            raise ValueError(
                f"Invalid fixed value for action '{action_name}': {raw_value}. Value must be finite."
            )
        lo, hi = _ACTION_BOUNDS[action_name]
        if value < lo or value > hi:
            raise ValueError(
                f"Invalid fixed value for action '{action_name}': {value}. "
                f"Expected range [{lo}, {hi}]."
            )
        fixed_action_values[action_name] = value
    return fixed_action_values


# ── Shield wrappers for evaluation ────────────────────────────────

class ShieldEvaluator:
    """Wraps a shield for evaluation, collecting per-step metrics."""

    def __init__(self, shield_type: str, shield):
        self.shield_type = shield_type
        self.shield = shield
        self._filter_times: list[float] = []

    def filter(self, action, obs):
        t0 = time.perf_counter()
        safe_action, was_modified, info = self.shield.filter(action, obs)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._filter_times.append(elapsed_ms)
        return safe_action, was_modified, info

    @property
    def mean_filter_time_ms(self) -> float:
        return float(np.mean(self._filter_times)) if self._filter_times else 0.0

    def reset(self):
        if hasattr(self.shield, "reset"):
            self.shield.reset()
        self._filter_times = []


class NoShield:
    """Dummy shield that passes actions through unchanged."""
    def filter(self, action, obs):
        return action.copy(), False, {}
    def reset(self):
        pass


def get_shield(agent_name: str) -> ShieldEvaluator:
    """Get the appropriate shield for the agent type."""
    if agent_name == "simplex_ppo":
        return ShieldEvaluator("simplex", SafetyShield())
    elif agent_name == "cbf_ppo":
        return ShieldEvaluator("cbf", CBFShield())
    elif agent_name == "hj_ppo":
        return ShieldEvaluator("hj", HJShield(precompute=True))
    elif agent_name == "mpcsf_ppo":
        return ShieldEvaluator("mpcsf", MPCSafetyFilter(horizon=5))
    elif agent_name in ("ppo_lagrangian", "cpo", "reward_shaping"):
        # Soft-guarantee methods: use Simplex shield at eval for fair comparison
        return ShieldEvaluator("simplex_eval", SafetyShield())
    elif agent_name == "ha_c2g":
        return ShieldEvaluator("simplex_ha", SafetyShield())
    elif agent_name == "cbm_shield":
        # CBM + shield (no gate) — shield is active
        return ShieldEvaluator("simplex_ablation", SafetyShield())
    elif agent_name in ("cbm_only", "cbm_gate"):
        # CBM-only and CBM+gate ablations: no shield
        return ShieldEvaluator("none", NoShield())
    else:
        return ShieldEvaluator("none", NoShield())


def compute_constraint_margin(obs: np.ndarray) -> float:
    """Compute minimum distance to any constraint boundary."""
    T_A = float(obs[_I_TEMP_A]) * T_SAFE
    T_B = float(obs[_I_TEMP_B]) * T_SAFE
    soc = float(obs[_I_SOC])
    freq_dev = abs(float(obs[_I_FREQ_DEV]) * 0.5)
    v_pcc = float(obs[_I_VPCC])

    margins = [
        T_SAFE - T_A,                      # C1
        T_SAFE - T_B,                      # C2
        soc - SOC_MIN,                     # C3 low
        SOC_MAX - soc,                     # C3 high
        0.5 - freq_dev,                    # C4
        v_pcc - 0.90,                      # C5
    ]
    return float(min(margins))


def check_hard_violation(obs: np.ndarray) -> bool:
    """Check if ANY hard constraint (C1-C5) is violated."""
    T_A = float(obs[_I_TEMP_A]) * T_SAFE
    T_B = float(obs[_I_TEMP_B]) * T_SAFE
    soc = float(obs[_I_SOC])
    freq_dev = abs(float(obs[_I_FREQ_DEV]) * 0.5)
    v_pcc = float(obs[_I_VPCC])

    return (
        T_A >= T_SAFE or
        T_B >= T_SAFE or
        soc < SOC_MIN or
        soc > SOC_MAX or
        freq_dev >= 0.5 or
        v_pcc < 0.90
    )


# ── Agent loader ──────────────────────────────────────────────────

class RandomAgent:
    def predict(self, obs, deterministic=True):
        action = np.array([
            np.random.uniform(0, 1),
            np.random.uniform(0, 1),
            np.random.uniform(0, 1),
            np.random.uniform(-1, 1),
        ], dtype=np.float32)
        return action, None


class SB3Agent:
    def __init__(self, model, algo_name: str, obs_normalizer=None):
        self._m = model
        self.algo_name = algo_name
        self._obs_normalizer = obs_normalizer

    def predict(self, obs, deterministic=True):
        if self._obs_normalizer is not None:
            obs = self._obs_normalizer.normalize_obs(np.asarray(obs, dtype=np.float32))
        return self._m.predict(obs, deterministic=deterministic)


class HAC2GAgent(SB3Agent):
    def __init__(self, model, algo_name: str, concept_encoder, safety_gate, obs_normalizer=None):
        super().__init__(model, algo_name=algo_name, obs_normalizer=obs_normalizer)
        self._concept_encoder = concept_encoder
        self._safety_gate = safety_gate

    def predict(self, obs, deterministic=True):
        action, state = super().predict(obs, deterministic=deterministic)
        raw_obs = np.asarray(obs, dtype=np.float32)
        with torch.no_grad():
            device = next(self._concept_encoder.parameters()).device
            obs_t = torch.as_tensor(raw_obs, dtype=torch.float32, device=device).unsqueeze(0)
            action_t = torch.as_tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
            concepts = self._concept_encoder(obs_t)
            gated_action, _, _ = compute_layer2_action(
                action_t,
                concepts,
                obs=obs_t,
                safety_gate=self._safety_gate,
            )
        return gated_action.squeeze(0).detach().cpu().numpy(), state


def _maybe_load_obs_normalizer(stats_path: Path, scenario: str):
    if not stats_path.exists():
        return None

    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    vec_env = DummyVecEnv([lambda: C2GFastEnv(scenario=scenario)])
    vec_norm = VecNormalize.load(str(stats_path), vec_env)
    vec_norm.training = False
    vec_norm.norm_reward = False
    return vec_norm


def load_agent(agent_name: str, scenario: str, seed: int, model_dir: str | None):
    """Load a trained agent. Returns (agent, needs_shield) tuple."""
    if agent_name == "random":
        return RandomAgent(), False

    # For RL-based agents, try to load from trained_models/
    from stable_baselines3 import PPO
    algo_map = {
        "simplex_ppo": "shielded_ppo",
        "cbf_ppo": "cbf_ppo",
        "hj_ppo": "hj_ppo",
        "mpcsf_ppo": "mpcsf_ppo",
        "ppo_lagrangian": "ppo_lagrangian",
        "cpo": "cpo",
        "reward_shaping": "shield_reward_shaping",
        "ha_c2g": "ha_c2g",
    }
    algo_key = algo_map.get(agent_name, agent_name)
    if model_dir:
        path = Path(model_dir) / "final_model"
    else:
        path = Path("trained_models") / f"{algo_key}_{scenario}_s{seed}" / "final_model"

    if path.with_suffix(".zip").exists():
        model = PPO.load(str(path))
        obs_normalizer = None
        if algo_key in _PPO_LIKE_AGENT_KEYS:
            obs_normalizer = _maybe_load_obs_normalizer(path.parent / "vec_normalize.pkl", scenario)
        if agent_name in ("ha_c2g", "cbm_gate"):
            fe = model.policy.features_extractor
            return HAC2GAgent(
                model,
                algo_name=agent_name,
                concept_encoder=fe.concept_encoder,
                safety_gate=fe.safety_gate,
                obs_normalizer=obs_normalizer,
            ), True
        return SB3Agent(model, algo_name=agent_name, obs_normalizer=obs_normalizer), True
    else:
        print(f"    SKIP: No model at {path}.zip — using random agent")
        return RandomAgent(), True


# ── Episode runner ────────────────────────────────────────────────

def run_ha_episode(
    agent,
    shield_eval: ShieldEvaluator,
    scenario: str,
    seed: int,
    agent_name: str,
    episode_number: int,
    record_transitions: bool = False,
    fixed_action_values: dict[str, float] | None = None,
) -> dict[str, float]:
    """Run one episode and return all 11 HA metrics."""
    env = _make_env(
        scenario=scenario,
        fixed_action_values=fixed_action_values,
    )
    obs, _ = env.reset(seed=seed)
    shield_eval.reset()

    transition_logger = None
    if record_transitions:
        transition_logger = C2GTransitionLoggerCallback(
            output_dir="runs",
            algorithm_name=agent_name,
            scenario_name=scenario,
            agent_type="hardware_ha",
            episode_number=episode_number,
            fixed_action_values=fixed_action_values,
            verbose=0,
        )

    rewards: list[float] = []
    tracking_errs: list[float] = []
    thermal_viols = 0
    hard_violations = 0
    interventions = 0
    throughputs: list[float] = []
    margins: list[float] = []
    worst_margin = float("inf")
    bess_init_age = None
    # Cumulative power metrics
    cumul_p_pump_mw     : float = 0.0
    cumul_p_hvac_mw     : float = 0.0
    cumul_flex_reduction_kw : float = 0.0
    cumul_bess_actual_kw : float = 0.0

    done = False
    n_steps = 0
    while not done:
        state = obs.copy()
        action, _ = agent.predict(obs, deterministic=True)

        # Apply safety shield
        safe_action, was_modified, shield_info = shield_eval.filter(action, obs)
        if was_modified:
            interventions += 1

        obs, reward, terminated, truncated, info = env.step(safe_action)
        done = terminated or truncated
        n_steps += 1

        if transition_logger is not None:
            reward_components = {
                k: info[k] for k in (
                    "reward_throughput", "reward_tracking", "reward_thermal",
                    "reward_soc", "reward_freq", "reward_volt", "reward_backlog",
                )
            }
            transition_logger.record_transition(
                state=state,
                action=safe_action,
                observation=obs,
                reward=float(reward),
                done=done,
                reward_components=reward_components,
            )

        rewards.append(float(reward))

        # Thermal violations (soft, T > T_warn)
        if obs[_I_TEMP_A] >= T_WARN_NORM or obs[_I_TEMP_B] >= T_WARN_NORM:
            thermal_viols += 1

        # Hard violations (any C1-C5 breach)
        if check_hard_violation(obs):
            hard_violations += 1

        # Constraint margin
        margin = compute_constraint_margin(obs)
        margins.append(margin)
        worst_margin = min(worst_margin, margin)

        # Tracking error
        tracking_errs.append(info.get("tracking_err_kw", 0.0) ** 2)

        # Throughput
        tp = info.get("throughput_ratio", None)
        if tp is not None:
            throughputs.append(float(tp))

        # Accumulate power metrics
        cumul_p_pump_mw += float(info.get("p_pump_mw", 0.0))
        cumul_p_hvac_mw += float(info.get("p_hvac_mw", 0.0))
        cumul_flex_reduction_kw += float(info.get("flex_reduction_kw", 0.0))
        cumul_bess_actual_kw += float(info.get("bess_actual_kw", 0.0))

        # BESS
        if bess_init_age is None:
            bess_init_age = float(info.get("bess_age_frac", 0.0))

    if transition_logger is not None:
        transition_logger.close()

    bess_final_age = float(info.get("bess_age_frac", bess_init_age or 0.0))
    bess_degradation = (bess_final_age - (bess_init_age or 0.0)) * 1e4
    survived = 1.0 if n_steps >= 288 else 0.0

    return {
        # Standard metrics
        "mean_reward": float(np.mean(rewards)),
        "total_reward": float(np.sum(rewards)),
        "tracking_rmse": float(np.sqrt(np.mean(tracking_errs))) if tracking_errs else 0.0,
        "thermal_viol_rate": thermal_viols / max(n_steps, 1),
        "throughput_ratio": float(np.mean(throughputs)) if throughputs else 0.0,
        "bess_degradation": bess_degradation,
        "episode_length": n_steps,
        "survived": survived,
        # HA-specific metrics
        "hard_violation_rate": hard_violations / max(n_steps, 1),
        "shield_intervention_rate": interventions / max(n_steps, 1),
        "constraint_margin": float(np.mean(margins)) if margins else 0.0,
        "worst_case_margin": worst_margin if worst_margin != float("inf") else 0.0,
        "computational_overhead_ms": shield_eval.mean_filter_time_ms,
        # Cumulative power metrics
        "cumul_p_pump_mw": cumul_p_pump_mw,
        "cumul_p_hvac_mw": cumul_p_hvac_mw,
        "cumul_flex_reduction_kw": cumul_flex_reduction_kw,
        "cumul_bess_actual_kw": cumul_bess_actual_kw,
    }


# ── Main benchmark loop ──────────────────────────────────────────

HA_AGENTS = [
    "simplex_ppo",
    "cbf_ppo",
    "hj_ppo",
    "mpcsf_ppo",
    "ppo_lagrangian",
    "cpo",
    "reward_shaping",
    "ha_c2g",
    "cbm_only",
    "cbm_gate",
    "cbm_shield",
    "random",
]


def benchmark(
    agents: list[str],
    scenarios: list[str],
    n_episodes: int,
    seed_start: int,
    model_dir: str | None,
    record_transitions: bool = False,
    fixed_action_values: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    if record_transitions:
        project_root = Path(__file__).resolve().parent.parent
        (project_root / "runs").mkdir(parents=True, exist_ok=True)

    for scenario in tqdm(scenarios, desc="Scenarios"):
        for agent_name in tqdm(agents, desc="Agents", leave=False):
            agent, needs_shield = load_agent(
                agent_name, scenario, seed_start, model_dir)
            shield_eval = get_shield(agent_name)

            ep_metrics: list[dict] = []
            t0 = time.perf_counter()
            for ep in range(n_episodes):
                m = run_ha_episode(
                    agent,
                    shield_eval,
                    scenario,
                    seed=seed_start + ep,
                    agent_name=agent_name,
                    episode_number=ep,
                    record_transitions=record_transitions,
                    fixed_action_values=fixed_action_values,
                )
                ep_metrics.append(m)
            elapsed = time.perf_counter() - t0

            # Aggregate
            keys = list(ep_metrics[0].keys())
            agg = {k: float(np.mean([m[k] for m in ep_metrics])) for k in keys}
            agg["survival_rate"] = float(np.mean([m["survived"] for m in ep_metrics]))

            row = {
                "scenario": scenario,
                "agent": agent_name,
                "shield_type": shield_eval.shield_type,
                "n_episodes": n_episodes,
                "wall_time_s": round(elapsed, 2),
                **{k: round(v, 6) for k, v in agg.items() if k != "survived"},
            }
            rows.append(row)
            tqdm.write(
                f"  {agent_name:20s}/{scenario:12s}  "
                f"reward={agg['mean_reward']:7.2f}  "
                f"hard_viol={agg['hard_violation_rate']:.4f}  "
                f"shield={agg['shield_intervention_rate']:.4f}  "
                f"margin={agg['constraint_margin']:.2f}  "
                f"survive={agg['survival_rate']:.2f}  "
                f"ms/step={agg['computational_overhead_ms']:.2f}"
            )

    return rows


def _default_output_path(
    agents: list[str],
    scenarios: list[str],
    fixed_action_values: dict[str, float] | None = None,
) -> Path:
    """Build a deterministic output path when --output is not provided."""
    unique_agents = list(dict.fromkeys(agents))
    unique_scenarios = list(dict.fromkeys(scenarios))

    algo_tag = unique_agents[0] if len(unique_agents) == 1 else "multi"
    scenario_tag = unique_scenarios[0] if len(unique_scenarios) == 1 else "multi"

    ablation_suffix = build_ablation_suffix(fixed_action_values)
    ablation_tag = ablation_suffix.lstrip("_") if ablation_suffix else "base"

    return Path("evaluation") / "results" / f"{algo_tag}_{scenario_tag}_hardware_ha_{ablation_tag}.csv"


def print_results_table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    cols = list(rows[0].keys())
    widths = {c: len(c) for c in cols}
    str_rows = []
    for row in rows:
        sr = {}
        for c in cols:
            v = row[c]
            sr[c] = f"{v:.4f}" if isinstance(v, float) else str(v)
            widths[c] = max(widths[c], len(sr[c]))
        str_rows.append(sr)

    header = " | ".join(c.rjust(widths[c]) for c in cols)
    sep = "-+-".join("-" * widths[c] for c in cols)
    print(f"\n{header}")
    print(sep)
    for sr in str_rows:
        print(" | ".join(sr[c].rjust(widths[c]) for c in cols))
    print()


def save_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[HA-Bench] Results saved -> {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="C2G-Bench High-Assurance evaluation runner")
    parser.add_argument(
        "--agents", nargs="+", default=HA_AGENTS,
        help="Agents to evaluate")
    parser.add_argument(
        "--scenarios", nargs="+", default=SCENARIOS,
        choices=SCENARIOS)
    parser.add_argument("--n_episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--n_seeds", type=int, default=1,
                        help="Number of independent seeds. If >1, runs "
                             "n_episodes per seed and saves per-seed rows "
                             "for statistical analysis.")
    parser.add_argument("--model_dir", default=None)
    parser.add_argument(
        "--record_transitions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable/disable per-step transition logging under runs/<algo>_<scenario>_ha/episode*.csv",
    )
    parser.add_argument(
        "--fixed-action",
        action="append",
        default=[],
        help="Optional fixed value for an action, e.g. --fixed-action hvac_effort=0.8",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Path to write the results CSV. If omitted, defaults to "
            "evaluation/results/<algo>_<scenario>_HA_hardware_<ablation>.csv"
        ),
    )
    args = parser.parse_args()

    try:
        fixed_action_values = _parse_fixed_action_args(args.fixed_action)
    except ValueError as exc:
        parser.error(str(exc))

    if args.n_seeds > 1:
        # Multi-seed mode: run full benchmark per seed, tag each row
        all_rows = []
        for s_idx in range(args.n_seeds):
            seed = args.seed + s_idx * 1000
            print(f"\n=== Seed {s_idx+1}/{args.n_seeds} (seed={seed}) ===")
            rows = benchmark(
                agents=args.agents,
                scenarios=args.scenarios,
                n_episodes=args.n_episodes,
                seed_start=seed,
                model_dir=args.model_dir,
                record_transitions=args.record_transitions,
                fixed_action_values=fixed_action_values)
            for r in rows:
                r["seed"] = seed
            all_rows.extend(rows)
        print_results_table(all_rows)
        output_path = (
            Path(args.output)
            if args.output
            else _default_output_path(
                agents=args.agents,
                scenarios=args.scenarios,
                fixed_action_values=fixed_action_values,
            )
        )
        save_csv(all_rows, output_path)
    else:
        rows = benchmark(
            agents=args.agents,
            scenarios=args.scenarios,
            n_episodes=args.n_episodes,
            seed_start=args.seed,
            model_dir=args.model_dir,
            record_transitions=args.record_transitions,
            fixed_action_values=fixed_action_values)
        print_results_table(rows)
        output_path = (
            Path(args.output)
            if args.output
            else _default_output_path(
                agents=args.agents,
                scenarios=args.scenarios,
                fixed_action_values=fixed_action_values,
            )
        )
        save_csv(rows, output_path)
