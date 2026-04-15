"""
evaluation/run_benchmark.py  —  Benchmark Evaluation Runner
============================================================
Runs all registered agents on all 4 evaluation scenarios, collects
per-episode metrics, and writes results to evaluation/results.csv.

Metrics (per episode)
---------------------
  mean_reward         — mean step reward
  total_reward        — sum of step rewards
  tracking_rmse       — RMSE of regulation signal tracking error (MW)
  thermal_viol_rate   — fraction of ticks with temp > T_warn (33°C)
  throughput_ratio    — mean(p_flex_kw / p_flex_nom_kw) over episode
  bess_degradation    — cumulative cycle ageing fraction * 1e4
  episode_length      — ticks in episode (< 288 means thermal termination)
  survival_rate       — fraction of episodes that ran to full 288 ticks

Usage
-----
  cd /lustre/guillant/C2G-Macro
  python evaluation/run_benchmark.py                         # rule-based only
  python evaluation/run_benchmark.py --agents rule_based ppo --n_episodes 10
  python evaluation/run_benchmark.py --model_dir trained_models/ppo_default_s42

Agents
------
  rule_based   — heuristic controller from baselines/rule_based_mpc.py
  rule_macro   — macro-level heuristic from baselines/rule_based_macro.py
  bang_bang     — hysteresis on/off controller
  pid          — multi-loop PID controller
  mpc_fast     — rolling-horizon MPC (fast env, scipy SLSQP)
  mpc_macro    — long-horizon MPC (macro env, scipy SLSQP)
  milp         — MILP economic dispatch (macro env)
  ppo          — loads trained_models/ppo_<scenario>_s42/final_model
  sac          — loads trained_models/sac_<scenario>_s42/final_model
  ppo_lag      — loads trained PPO-Lagrangian model
  cmaes        — loads CMA-ES trained linear policy
  pso          — loads PSO trained linear policy
  random       — np.random uniform (lower bound)
"""
from __future__ import annotations
import argparse, csv, time
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from c2g_env import C2GFastEnv
from baselines.rule_based_mpc import RuleBasedController
from baselines.rule_based_macro import RuleBasedMacroController
from baselines.bang_bang import BangBangController
from baselines.pid_controller import PIDController
from baselines.mpc_fast import MPCFastController
from baselines.mpc_macro import MPCMacroController
from baselines.milp_dispatch import MILPDispatchController

SCENARIOS    = ["default", "scenario_a", "scenario_b", "scenario_c"]
T_WARN_NORM  = 33.0 / 35.0   # normalised warning threshold
DT_S         = 300            # seconds per tick
COMMIT_MW    = {"default": 15.0, "scenario_a": 20.0, "scenario_b": 30.0, "scenario_c": 15.0}


# ---------------------------------------------------------------------------
# Agent wrappers — expose a common predict(obs) -> (action, state) interface
# ---------------------------------------------------------------------------

class RandomAgent:
    """Samples uniformly from the action space."""
    def __init__(self, env: C2GFastEnv):
        self._space = env.action_space

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        return self._space.sample(), None


class SB3Agent:
    """Wraps a loaded SB3 model (PPO or SAC)."""
    def __init__(self, model):
        self._model = model

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        return self._model.predict(obs, deterministic=deterministic)


def load_sb3_agent(algo: str, scenario: str, seed: int, model_dir: str | None):
    from stable_baselines3 import PPO, SAC
    algos = {"ppo": PPO, "sac": SAC}
    cls = algos.get(algo.lower())
    if cls is None:
        raise ValueError(f"Unknown algo '{algo}'. Use ppo or sac.")
    if model_dir:
        path = Path(model_dir) / "final_model"
    else:
        path = Path("trained_models") / f"{algo}_{scenario}_s{seed}" / "final_model"
    if not path.with_suffix(".zip").exists():
        raise FileNotFoundError(
            f"No trained model at {path}.zip — run baselines/train_{algo}.py first."
        )
    model = cls.load(str(path))
    return SB3Agent(model)


class EvolutionaryAgent:
    """Wraps a CMA-ES or PSO linear policy loaded from .npz."""
    def __init__(self, npz_path: str | Path):
        data = np.load(npz_path)
        self.W = data["W"]
        self.b = data["b"]
        self.act_low = data["act_low"]
        self.act_high = data["act_high"]

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        if obs.ndim == 1:
            action = np.clip(self.W @ obs + self.b, self.act_low, self.act_high)
        else:
            action = np.clip(obs @ self.W.T + self.b, self.act_low, self.act_high)
        return action.astype(np.float32), None


# ---------------------------------------------------------------------------
# Metric collection
# ---------------------------------------------------------------------------

def run_episode(agent, scenario: str, seed: int) -> dict[str, float]:
    """Run one episode and return a metrics dict."""
    env = C2GFastEnv(scenario=scenario)
    obs, _ = env.reset(seed=seed)

    committed_mw  = COMMIT_MW.get(scenario, 15.0)
    rewards       : list[float] = []
    tracking_errs : list[float] = []
    thermal_viols : int = 0
    throughputs   : list[float] = []
    bess_init_age : float | None = None

    done = False
    while not done:
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        rewards.append(float(reward))

        # Thermal violations
        if obs[0] >= T_WARN_NORM or obs[1] >= T_WARN_NORM:
            thermal_viols += 1

        # Tracking error: regd_signal * committed_mw vs actual BESS response
        # obs[6] = regd_signal ([-1,1]); info["delta_p_kw"] = BESS dispatch
        regd = float(obs[6])
        target_kw  = regd * committed_mw * 1e3          # kW
        actual_kw  = float(info.get("delta_p_kw", 0.0))
        tracking_errs.append((target_kw - actual_kw) ** 2)

        # Throughput ratio: flex_kw / flex_nom_kw
        tp = info.get("throughput_ratio", None)
        if tp is not None:
            throughputs.append(float(tp))

        # BESS degradation
        if bess_init_age is None:
            bess_init_age = float(info.get("bess_age_frac", 0.0))

    bess_final_age = float(info.get("bess_age_frac", bess_init_age or 0.0))
    bess_degradation = (bess_final_age - (bess_init_age or 0.0)) * 1e4

    n_ticks    = len(rewards)
    survived   = 1.0 if n_ticks >= 288 else 0.0

    return {
        "mean_reward"       : float(np.mean(rewards)),
        "total_reward"      : float(np.sum(rewards)),
        "tracking_rmse"     : float(np.sqrt(np.mean(tracking_errs))) if tracking_errs else 0.0,
        "thermal_viol_rate" : thermal_viols / max(n_ticks, 1),
        "throughput_ratio"  : float(np.mean(throughputs)) if throughputs else 0.0,
        "bess_degradation"  : bess_degradation,
        "episode_length"    : n_ticks,
        "survived"          : survived,
    }


def benchmark(
    agents      : list[str],
    scenarios   : list[str],
    n_episodes  : int,
    seed_start  : int,
    model_dir   : str | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    scenario_bar = tqdm(scenarios, desc="Scenarios", position=0)
    for scenario in scenario_bar:
        scenario_bar.set_description(f"Scenario: {scenario}")

        agent_bar = tqdm(agents, desc="Agents", position=1, leave=False)
        for agent_name in agent_bar:
            agent_bar.set_description(f"Agent: {agent_name}")

            # Instantiate agent once per (agent, scenario) to share weights
            env_for_space = C2GFastEnv(scenario=scenario)
            env_for_space.reset(seed=0)

            if agent_name == "rule_based":
                agent = RuleBasedController()
            elif agent_name == "rule_macro":
                agent = RuleBasedMacroController()
            elif agent_name == "bang_bang":
                agent = BangBangController()
            elif agent_name == "pid":
                agent = PIDController()
            elif agent_name == "mpc_fast":
                agent = MPCFastController()
            elif agent_name == "mpc_macro":
                agent = MPCMacroController()
            elif agent_name == "milp":
                agent = MILPDispatchController()
            elif agent_name == "random":
                agent = RandomAgent(env_for_space)
            elif agent_name in ("cmaes", "pso"):
                npz_name = f"{agent_name}_policy.npz"
                npz_dir = Path(model_dir) if model_dir else Path("trained_models") / f"{agent_name}_{scenario}_s{seed_start}"
                npz_path = npz_dir / npz_name
                if not npz_path.exists():
                    print(f"    SKIP: No trained policy at {npz_path}")
                    continue
                agent = EvolutionaryAgent(npz_path)
            else:
                try:
                    agent = load_sb3_agent(agent_name, scenario, seed_start, model_dir)
                except FileNotFoundError as exc:
                    print(f"    SKIP: {exc}")
                    continue

            ep_metrics: list[dict[str, float]] = []
            t0 = time.perf_counter()
            for ep in tqdm(range(n_episodes), desc="Episodes", position=2, leave=False):
                m = run_episode(agent, scenario, seed=seed_start + ep)
                ep_metrics.append(m)

            elapsed = time.perf_counter() - t0

            # Aggregate across episodes
            keys = list(ep_metrics[0].keys())
            agg  = {k: float(np.mean([m[k] for m in ep_metrics])) for k in keys}
            agg["survival_rate"] = float(np.mean([m["survived"] for m in ep_metrics]))

            row = {
                "scenario"          : scenario,
                "agent"             : agent_name,
                "n_episodes"        : n_episodes,
                "wall_time_s"       : round(elapsed, 2),
                **{k: round(v, 4) for k, v in agg.items() if k != "survived"},
            }
            rows.append(row)
            tqdm.write(
                f"  {agent_name}/{scenario}  "
                f"reward={agg['mean_reward']:7.2f}  "
                f"tracking_rmse={agg['tracking_rmse']:8.0f}kW  "
                f"thermal_viol={agg['thermal_viol_rate']:.3f}  "
                f"survive={agg['survival_rate']:.2f}"
            )

    return rows


def print_results_table(rows: list[dict[str, Any]]) -> None:
    """Print results as a formatted table."""
    if not rows:
        return
    cols = list(rows[0].keys())
    # Compute column widths
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
        print("No results to save.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved → {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="C2G-Bench evaluation runner")
    parser.add_argument(
        "--agents", nargs="+",
        default=["rule_based", "bang_bang", "pid", "random"],
        help="Agents to evaluate: rule_based rule_macro bang_bang pid mpc_fast mpc_macro milp ppo sac ppo_lag cmaes pso random",
    )
    parser.add_argument(
        "--scenarios", nargs="+",
        default=SCENARIOS,
        choices=SCENARIOS,
    )
    parser.add_argument("--n_episodes", type=int, default=5)
    parser.add_argument("--seed",       type=int, default=100)
    parser.add_argument(
        "--model_dir", default=None,
        help="Override model directory for SB3 agents (optional)",
    )
    parser.add_argument(
        "--output", default="evaluation/results.csv",
        help="Path to write the results CSV",
    )
    args = parser.parse_args()

    rows = benchmark(
        agents     = args.agents,
        scenarios  = args.scenarios,
        n_episodes = args.n_episodes,
        seed_start = args.seed,
        model_dir  = args.model_dir,
    )
    print_results_table(rows)
    save_csv(rows, Path(args.output))
