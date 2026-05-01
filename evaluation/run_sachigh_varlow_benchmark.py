"""
evaluation/run_sachigh_varlow_benchmark.py
==========================================
Hierarchical benchmark: trained SAC macro agent + variable low-level controllers.

The SAC macro agent (trained on C2GMacroEnv) outputs 2-D actions
(bid_mw, price_eur_mwh) every 15-minute macro step.  At each of the
180 sub-steps within a macro tick, one of the classic low-level
controllers (bang_bang, pid, rule_based) maps the 18-D low-level
observation to 4-D hardware actions.

Usage
-----
  cd <project_root>

  # Run all three low-level controllers
  python evaluation/run_sachigh_varlow_benchmark.py \
      --macro-model-dir outputs/sac_default/sacmacro_rulelowlevel/seed100/phase2/best_model/final_model.zip

  # Specific low-level controllers
  python evaluation/run_sachigh_varlow_benchmark.py \
      --macro-model-dir path/to/macro/final_model.zip \
      --inner-agents pid rule_based

  # Multiple scenarios
  python evaluation/run_sachigh_varlow_benchmark.py \
      --macro-model-dir path/to/macro/final_model.zip \
      --scenarios default scenario_a --n_episodes 10
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from c2g_env import C2GFastEnv, C2GMacroEnv
from baselines.rule_based_mpc import RuleBasedController
from baselines.bang_bang import BangBangController
from baselines.pid_controller import PIDController
from baselines.metrics_callback import C2GTransitionLoggerCallback

SCENARIOS = ["default", "scenario_a", "scenario_b", "scenario_c"]
INNER_AGENTS = ["bang_bang", "pid", "rule_based"]


# ---------------------------------------------------------------------------
# Agent wrappers
# ---------------------------------------------------------------------------

class SB3Agent:
    """Wraps a loaded SB3 model (PPO or SAC)."""

    def __init__(self, model, algo_name: str):
        self._model = model
        self.algo_name = algo_name

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        return self._model.predict(obs, deterministic=deterministic)


def _load_sac_model(model_path: Path, label: str) -> SB3Agent:
    """Load a SAC model from a directory or direct path.

    Accepts:
      - A directory containing final_model.zip  (e.g. trained_models/sac_macro_default_s42/)
      - A direct path to a .zip file            (e.g. .../best_model/final_model.zip)
      - A path without .zip extension           (e.g. .../best_model/final_model)
    """
    from stable_baselines3 import SAC

    model_path = Path(model_path)

    # Case 1: path is already a .zip file
    if model_path.suffix == ".zip" and model_path.exists():
        load_path = str(model_path.with_suffix(""))
    # Case 2: path without extension but .zip exists alongside
    elif model_path.with_suffix(".zip").exists():
        load_path = str(model_path)
    # Case 3: path is a directory containing final_model.zip
    elif (model_path / "final_model.zip").exists():
        load_path = str(model_path / "final_model")
    else:
        zip_path = model_path / "final_model.zip"
        raise FileNotFoundError(
            f"No trained {label} model found. Searched:\n"
            f"  - {model_path}\n"
            f"  - {model_path.with_suffix('.zip')}\n"
            f"  - {zip_path}"
        )

    model = SAC.load(load_path)
    return SB3Agent(model, algo_name=label)


def _make_inner_controller(name: str):
    """Instantiate a low-level controller by name."""
    if name == "bang_bang":
        return BangBangController()
    if name == "pid":
        return PIDController()
    if name == "rule_based":
        return RuleBasedController()
    raise ValueError(f"Unknown inner controller '{name}'. Choose from: {INNER_AGENTS}")


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_hierarchical_episode(
    macro_agent: SB3Agent,
    inner_controller,
    inner_name: str,
    scenario: str,
    seed: int,
    episode_number: int = 0,
    record_transitions: bool = False,
) -> dict[str, float]:
    """Run one macro episode with SAC-high + classic low-level controller."""

    # Build the inner-action callback that C2GMacroEnv calls every sub-step.
    def inner_action_fn(inner_obs: np.ndarray, _macro_action: np.ndarray) -> np.ndarray:
        action, _ = inner_controller.predict(inner_obs, deterministic=True)
        return action

    env = C2GMacroEnv(scenario=scenario, inner_action_fn=inner_action_fn)
    obs, _ = env.reset(seed=seed)

    combo_name = f"sac_macro+{inner_name}"

    transition_logger = None
    if record_transitions:
        transition_logger = C2GTransitionLoggerCallback(
            output_dir="runs",
            algorithm_name=combo_name,
            scenario_name=scenario,
            agent_type="macro",
            episode_number=episode_number,
            fixed_action_values=None,
            verbose=0,
        )

    rewards: list[float] = []
    bids_accepted: list[float] = []
    reg_revenues: list[float] = []
    elec_costs: list[float] = []
    perf_scores: list[float] = []
    committed_mws: list[float] = []
    tracking_errs: list[float] = []
    temp_a_maxes: list[float] = []
    temp_b_maxes: list[float] = []
    flex_reductions: list[float] = []
    bess_actuals: list[float] = []
    cool_deltas: list[float] = []
    p_pumps: list[float] = []
    p_hvacs: list[float] = []
    inner_throttles: list[float] = []
    inner_pumps: list[float] = []
    inner_hvacs: list[float] = []
    inner_besses: list[float] = []

    done = False
    while not done:
        state = obs.copy()
        action, _ = macro_agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if transition_logger is not None:
            reward_components = {
                k: info.get(k, 0.0)
                for k in ("reward_regulation", "reward_sub", "reward_elec", "reward_churn")
            }
            transition_logger.record_transition(
                state=state,
                action=action,
                observation=obs,
                reward=float(reward),
                done=done,
                reward_components=reward_components,
            )

        rewards.append(float(reward))
        bids_accepted.append(float(info.get("bid_accepted", False)))
        reg_revenues.append(float(info.get("regulation_revenue", 0.0)))
        elec_costs.append(float(info.get("electricity_cost", 0.0)))
        perf_scores.append(float(info.get("perf_score", 0.0)))
        committed_mws.append(float(info.get("committed_mw", 0.0)))
        tracking_errs.append(float(info.get("mean_tracking_err", 0.0)) ** 2)
        temp_a_maxes.append(float(info.get("temp_A_max", 0.0)))
        temp_b_maxes.append(float(info.get("temp_B_max", 0.0)))
        flex_reductions.append(float(info.get("mean_flex_reduction_kw", 0.0)))
        bess_actuals.append(float(info.get("mean_bess_actual_kw", 0.0)))
        cool_deltas.append(float(info.get("mean_cool_delta_kw", 0.0)))
        p_pumps.append(float(info.get("mean_p_pump_mw", 0.0)))
        p_hvacs.append(float(info.get("mean_p_hvac_mw", 0.0)))
        inner_throttles.append(float(info.get("mean_inner_throttle", 0.0)))
        inner_pumps.append(float(info.get("mean_inner_pump", 0.0)))
        inner_hvacs.append(float(info.get("mean_inner_hvac", 0.0)))
        inner_besses.append(float(info.get("mean_inner_bess", 0.0)))

    if transition_logger is not None:
        transition_logger.close()

    n_steps = len(rewards)
    macro_ticks_full = env._episode_macro_ticks
    survived = 1.0 if n_steps >= macro_ticks_full else 0.0

    return {
        "mean_reward":          float(np.mean(rewards)),
        "total_reward":         float(np.sum(rewards)),
        "bid_acceptance_rate":  float(np.mean(bids_accepted)),
        "total_reg_revenue":    float(np.sum(reg_revenues)),
        "total_elec_cost":      float(np.sum(elec_costs)),
        "mean_perf_score":      float(np.mean(perf_scores)),
        "mean_committed_mw":    float(np.mean(committed_mws)),
        "tracking_rmse":        float(np.sqrt(np.mean(tracking_errs))) if tracking_errs else 0.0,
        "temp_A_max":           float(np.max(temp_a_maxes)) if temp_a_maxes else 0.0,
        "temp_B_max":           float(np.max(temp_b_maxes)) if temp_b_maxes else 0.0,
        "episode_length":       n_steps,
        "survived":             survived,
        "mean_flex_kw":         float(np.mean(flex_reductions)) if flex_reductions else 0.0,
        "mean_bess_kw":         float(np.mean(bess_actuals)) if bess_actuals else 0.0,
        "mean_cool_delta_kw":   float(np.mean(cool_deltas)) if cool_deltas else 0.0,
        "mean_p_pump_mw":       float(np.mean(p_pumps)) if p_pumps else 0.0,
        "mean_p_hvac_mw":       float(np.mean(p_hvacs)) if p_hvacs else 0.0,
        "mean_inner_throttle":  float(np.mean(inner_throttles)) if inner_throttles else 0.0,
        "mean_inner_pump":      float(np.mean(inner_pumps)) if inner_pumps else 0.0,
        "mean_inner_hvac":      float(np.mean(inner_hvacs)) if inner_hvacs else 0.0,
        "mean_inner_bess":      float(np.mean(inner_besses)) if inner_besses else 0.0,
    }


# ---------------------------------------------------------------------------
# Benchmark loop
# ---------------------------------------------------------------------------

def benchmark(
    inner_agents: list[str],
    scenarios: list[str],
    n_episodes: int,
    seed_start: int,
    macro_model_path: Path,
    record_transitions: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    if record_transitions:
        project_root = Path(__file__).resolve().parent.parent
        (project_root / "runs").mkdir(parents=True, exist_ok=True)

    # Load macro agent once (shared across all inner controllers and scenarios)
    macro_agent = _load_sac_model(macro_model_path, label="sac_macro")

    for scenario in tqdm(scenarios, desc="Scenarios", position=0):
        for inner_name in tqdm(inner_agents, desc="Inner agents", position=1, leave=False):
            inner_controller = _make_inner_controller(inner_name)
            combo_name = f"sac_macro+{inner_name}"

            ep_metrics: list[dict[str, float]] = []
            t0 = time.perf_counter()

            for ep in tqdm(range(n_episodes), desc=f"{combo_name}/{scenario}", position=2, leave=False):
                m = run_hierarchical_episode(
                    macro_agent=macro_agent,
                    inner_controller=inner_controller,
                    inner_name=inner_name,
                    scenario=scenario,
                    seed=seed_start + ep,
                    episode_number=ep,
                    record_transitions=record_transitions,
                )
                ep_metrics.append(m)

            elapsed = time.perf_counter() - t0

            # Aggregate across episodes
            keys = [k for k in ep_metrics[0].keys() if k != "survived"]
            agg = {k: float(np.mean([m[k] for m in ep_metrics])) for k in keys}
            std = {
                k: float(np.std([m[k] for m in ep_metrics], ddof=1)) if n_episodes > 1 else 0.0
                for k in keys
            }
            agg["survival_rate"] = float(np.mean([m["survived"] for m in ep_metrics]))

            row: dict[str, Any] = {
                "scenario":   scenario,
                "agent":      combo_name,
                "n_episodes": n_episodes,
                "wall_time_s": round(elapsed, 2),
                **{k: round(v, 4) for k, v in agg.items()},
                **{f"{k}_std": round(v, 4) for k, v in std.items()},
            }
            rows.append(row)

            tqdm.write(
                f"  {combo_name}/{scenario}  "
                f"reward={agg['mean_reward']:7.2f}  "
                f"accept_rate={agg['bid_acceptance_rate']:.3f}  "
                f"reg_rev={agg['total_reg_revenue']:8.1f}  "
                f"survive={agg['survival_rate']:.2f}  "
                f"flex={agg['mean_flex_kw']:.0f}kW  "
                f"bess={agg['mean_bess_kw']:.0f}kW  "
                f"cool_d={agg['mean_cool_delta_kw']:.0f}kW  "
                f"inner[thr={agg['mean_inner_throttle']:.2f} "
                f"pmp={agg['mean_inner_pump']:.2f} "
                f"hvac={agg['mean_inner_hvac']:.2f} "
                f"bess={agg['mean_inner_bess']:+.2f}]"
            )

    return rows


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

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


def save_csv(rows: list[dict[str, Any]], path: Path, append: bool = False) -> None:
    if not rows:
        print("No results to save.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    if append and path.exists():
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerows(rows)
        print(f"\nResults appended -> {path}")
    else:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nResults saved -> {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hierarchical benchmark: SAC macro + variable low-level controllers (bang_bang, pid, rule_based)"
    )
    parser.add_argument(
        "--macro-model-dir", type=str, required=True,
        help="Path to trained SAC macro model. Accepts: "
             "a .zip file, a path without .zip, or a directory containing final_model.zip",
    )
    parser.add_argument(
        "--inner-agents", nargs="+", type=str,
        default=INNER_AGENTS,
        choices=INNER_AGENTS,
        help=f"Low-level controllers to evaluate (default: {INNER_AGENTS})",
    )
    parser.add_argument(
        "--scenarios", nargs="+", type=str,
        default=["default"],
        choices=SCENARIOS,
        help="Evaluation scenarios (default: [default])",
    )
    parser.add_argument("--n_episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output CSV path. Default: evaluation/results/sac_macro+varlow_<scenario>.csv",
    )
    parser.add_argument(
        "--record_transitions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Log per-step transitions under runs/",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        default=False,
        help="Append results to existing CSV instead of overwriting",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    macro_path = Path(args.macro_model_dir)

    print(f"\n{'='*60}")
    print(f"SAC Macro model : {macro_path}")
    print(f"Inner agents    : {args.inner_agents}")
    print(f"Scenarios       : {args.scenarios}")
    print(f"Episodes        : {args.n_episodes}  |  Seed start: {args.seed}")
    print(f"{'='*60}\n")

    rows = benchmark(
        inner_agents=args.inner_agents,
        scenarios=args.scenarios,
        n_episodes=args.n_episodes,
        seed_start=args.seed,
        macro_model_path=macro_path,
        record_transitions=args.record_transitions,
    )

    print_results_table(rows)

    scenario_tag = args.scenarios[0] if len(args.scenarios) == 1 else "multi"
    inner_tag = "_".join(args.inner_agents) if len(args.inner_agents) < 4 else "varlow"
    output_path = (
        Path(args.output)
        if args.output
        else project_root / "evaluation" / "results" / f"sac_macro+{inner_tag}_{scenario_tag}.csv"
    )
    save_csv(rows, output_path, append=args.append)
