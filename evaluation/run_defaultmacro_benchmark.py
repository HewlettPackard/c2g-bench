"""
evaluation/run_defmacro_benchmark.py  --  Default-Macro Benchmark Runner
=========================================================================
Evaluates low-level controllers (e.g. trained SAC) under a *fixed*
default macro-level controller that always outputs the same bid.

The ``default_macro`` agent uses constant macro actions derived from the
scenario defaults (committed_mw and RMCP), letting the inner low-level
controller handle all hardware-level decisions.

Usage
-----
  # Evaluate SAC low-level controller with default_macro high-level:
  python evaluation/run_defmacro_benchmark.py --agents default_macro+sac

  # Pair default_macro with multiple inner agents:
  python evaluation/run_defmacro_benchmark.py \
      --agents default_macro+sac default_macro+pid default_macro+rule_based

  # Specify a custom model directory for the SAC model:
  python evaluation/run_defmacro_benchmark.py \
      --agents default_macro+sac --model_dir outputs/sac_default/seed_42/<timestamp>

  # Evaluate on multiple scenarios:
  python evaluation/run_defmacro_benchmark.py \
      --agents default_macro+sac \
      --scenarios default scenario_a scenario_b scenario_c

Agents
------
  default_macro       — fixed macro controller (constant bid MW + price)
  default_macro+sac   — default macro + trained SAC low-level controller
  default_macro+pid   — default macro + PID low-level controller
  default_macro+rule_based — default macro + rule-based low-level controller
  default_macro+bang_bang   — default macro + bang-bang low-level controller
  default_macro+mpc_fast   — default macro + MPC low-level controller
  default_macro+ppo   — default macro + trained PPO low-level controller
  default_macro+random — default macro + random low-level controller
"""
from __future__ import annotations
import argparse, csv, time
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from c2g_env import C2GFastEnv, C2GMacroEnv
from baselines.rule_based_mpc import RuleBasedController
from baselines.bang_bang import BangBangController
from baselines.pid_controller import PIDController
from baselines.mpc_fast import MPCFastController
from baselines.metrics_callback import C2GTransitionLoggerCallback

SCENARIOS   = ["default", "scenario_a", "scenario_b", "scenario_c"]
T_WARN_NORM = 33.0 / 35.0


# ---------------------------------------------------------------------------
# Default-Macro controller — fixed macro actions from scenario defaults
# ---------------------------------------------------------------------------

class DefaultMacroController:
    """
    A macro-level controller that always outputs a fixed (bid_mw_norm,
    bid_price_norm) pair.  This represents the scenario's *default*
    commitment strategy:

    * **bid_mw_norm = 0.50** — commit 50 % of max capacity (moderate).
    * **bid_price_norm = 0.25** — bid at roughly 50 % of RMCP (high
      acceptance probability, similar to rule_based_macro's pricing).

    These values can be overridden via constructor arguments.
    """

    def __init__(
        self,
        bid_mw_norm: float = 0.50,
        bid_price_norm: float = 0.25,
        algo_name: str = "default_macro",
    ):
        self._action = np.array(
            [bid_mw_norm, bid_price_norm], dtype=np.float32
        )
        self.algo_name = algo_name

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        return self._action.copy(), None


# ---------------------------------------------------------------------------
# Agent helpers (reused from run_benchmark.py)
# ---------------------------------------------------------------------------

class SB3Agent:
    """Wraps a loaded SB3 model (PPO or SAC)."""
    def __init__(self, model, algo_name: str, obs_normalizer=None):
        self._model = model
        self.algo_name = algo_name
        self._obs_normalizer = obs_normalizer

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        if self._obs_normalizer is not None:
            obs = self._obs_normalizer.normalize_obs(
                np.asarray(obs, dtype=np.float32)
            )
        return self._model.predict(obs, deterministic=deterministic)


class RandomAgent:
    """Samples uniformly from the action space."""
    def __init__(self, env, algo_name: str = "random"):
        self._space = env.action_space
        self.algo_name = algo_name

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        return self._space.sample(), None


_RL_ALGOS = {"ppo", "sac"}
_INNER_CONTROLLERS = {"random", "pid", "bang_bang", "rule_based", "mpc_fast", "ppo", "sac"}


def _resolve_sb3_spec(algo: str):
    from stable_baselines3 import PPO, SAC

    algo_key = algo.lower()
    if algo_key == "sac":
        return SAC, "sac", False
    if algo_key in _RL_ALGOS:
        return PPO, algo_key, True
    raise ValueError(f"Unknown RL algo '{algo}'.")


def _maybe_load_obs_normalizer(stats_path: Path, scenario: str):
    if not stats_path.exists():
        return None
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    vec_env = DummyVecEnv([lambda: C2GFastEnv(scenario=scenario)])
    vec_norm = VecNormalize.load(str(stats_path), vec_env)
    vec_norm.training = False
    vec_norm.norm_reward = False
    return vec_norm


def load_sb3_agent(algo: str, scenario: str, seed: int, model_dir: str | None):
    cls, train_key, should_restore_norm = _resolve_sb3_spec(algo)
    if model_dir:
        path = Path(model_dir) / "final_model"
    else:
        path = (
            Path("trained_models")
            / f"{train_key}_{scenario}_s{seed}"
            / "final_model"
        )
    if not path.with_suffix(".zip").exists():
        raise FileNotFoundError(
            f"No trained model at {path}.zip — "
            f"run baselines/train_{algo}.py first."
        )
    model = cls.load(str(path))
    obs_normalizer = None
    if should_restore_norm:
        obs_normalizer = _maybe_load_obs_normalizer(
            path.parent / "vec_normalize.pkl", scenario
        )
    return SB3Agent(model, algo_name=algo.lower(), obs_normalizer=obs_normalizer)


def _make_inner_controller(
    name: str,
    env: C2GFastEnv | None = None,
    scenario: str = "default",
    seed: int = 42,
    model_dir: str | None = None,
):
    """Instantiate a low-level controller by name."""
    if name == "random":
        if env is None:
            raise ValueError("env required for random inner controller")
        return RandomAgent(env)
    if name == "pid":
        return PIDController()
    if name == "bang_bang":
        return BangBangController()
    if name == "rule_based":
        return RuleBasedController()
    if name == "mpc_fast":
        return MPCFastController()
    if name == "ppo":
        return load_sb3_agent("ppo", scenario, seed, model_dir)
    if name == "sac":
        return load_sb3_agent("sac", scenario, seed, model_dir)
    raise ValueError(
        f"Unknown inner controller '{name}'. "
        f"Choose from: {_INNER_CONTROLLERS}"
    )


def _make_macro_env(scenario: str, **kwargs) -> C2GMacroEnv:
    return C2GMacroEnv(scenario=scenario, **kwargs)


def _make_env(scenario: str, **kwargs) -> C2GFastEnv:
    return C2GFastEnv(scenario=scenario, **kwargs)


# ---------------------------------------------------------------------------
# Macro episode runner
# ---------------------------------------------------------------------------

def run_macro_episode(
    agent,
    scenario: str,
    seed: int,
    algo_name: str | None = None,
    agent_type: str = "macro",
    episode_number: int = 0,
    inner_action_fn: Any = None,
    record_transitions: bool = True,
) -> dict[str, float]:
    """Run one macro-level episode and return metrics dict."""
    env = _make_macro_env(scenario=scenario, inner_action_fn=inner_action_fn)
    obs, _ = env.reset(seed=seed)
    algo_for_logging = getattr(agent, "algo_name", (algo_name or "unknown"))

    transition_logger = None
    if record_transitions:
        transition_logger = C2GTransitionLoggerCallback(
            output_dir="runs",
            algorithm_name=algo_for_logging,
            scenario_name=scenario,
            agent_type=agent_type,
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

    done = False
    while not done:
        state = obs.copy()
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if transition_logger is not None:
            reward_components = {
                k: info.get(k, 0.0)
                for k in (
                    "reward_regulation",
                    "reward_sub",
                    "reward_elec",
                    "reward_churn",
                )
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

    if transition_logger is not None:
        transition_logger.close()

    n_steps = len(rewards)
    macro_ticks_full = env._episode_macro_ticks
    survived = 1.0 if n_steps >= macro_ticks_full else 0.0

    return {
        "mean_reward": float(np.mean(rewards)),
        "total_reward": float(np.sum(rewards)),
        "bid_acceptance_rate": float(np.mean(bids_accepted)),
        "total_reg_revenue": float(np.sum(reg_revenues)),
        "total_elec_cost": float(np.sum(elec_costs)),
        "mean_perf_score": float(np.mean(perf_scores)),
        "mean_committed_mw": float(np.mean(committed_mws)),
        "tracking_rmse": float(np.sqrt(np.mean(tracking_errs)))
        if tracking_errs
        else 0.0,
        "temp_A_max": float(np.max(temp_a_maxes)) if temp_a_maxes else 0.0,
        "temp_B_max": float(np.max(temp_b_maxes)) if temp_b_maxes else 0.0,
        "episode_length": n_steps,
        "survived": survived,
        "mean_flex_kw": float(np.mean(flex_reductions))
        if flex_reductions
        else 0.0,
        "mean_bess_kw": float(np.mean(bess_actuals)) if bess_actuals else 0.0,
        "mean_cool_delta_kw": float(np.mean(cool_deltas))
        if cool_deltas
        else 0.0,
        "mean_p_pump_mw": float(np.mean(p_pumps)) if p_pumps else 0.0,
        "mean_p_hvac_mw": float(np.mean(p_hvacs)) if p_hvacs else 0.0,
    }


# ---------------------------------------------------------------------------
# Benchmark loop
# ---------------------------------------------------------------------------

def benchmark(
    agents: list[str],
    scenarios: list[str],
    n_episodes: int,
    seed_start: int,
    model_dir: str | None,
    record_transitions: bool = True,
    bid_mw_norm: float = 0.50,
    bid_price_norm: float = 0.25,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    if record_transitions:
        project_root = Path(__file__).resolve().parent.parent
        (project_root / "runs").mkdir(parents=True, exist_ok=True)

    scenario_bar = tqdm(scenarios, desc="Scenarios", position=0)
    for scenario in scenario_bar:
        scenario_bar.set_description(f"Scenario: {scenario}")

        agent_bar = tqdm(agents, desc="Agents", position=1, leave=False)
        for agent_name in agent_bar:
            agent_bar.set_description(f"Agent: {agent_name}")

            # ── Parse agent name ──────────────────────────────────
            if "+" in agent_name:
                macro_part, inner_part = agent_name.split("+", 1)
            else:
                macro_part = agent_name
                inner_part = None

            if macro_part != "default_macro":
                print(
                    f"  SKIP: '{macro_part}' is not default_macro. "
                    f"Use run_benchmark.py for other macro agents."
                )
                continue

            # ── Build inner (low-level) controller ────────────────
            inner_action_fn = None
            if inner_part is not None:
                inner_env = _make_env(scenario=scenario)
                inner_env.reset(seed=0)
                inner_ctrl = _make_inner_controller(
                    inner_part,
                    env=inner_env,
                    scenario=scenario,
                    seed=seed_start,
                    model_dir=model_dir,
                )
                inner_action_fn = (
                    lambda obs, _act, c=inner_ctrl: c.predict(obs)[0]
                )

            # ── Build default_macro high-level controller ─────────
            agent = DefaultMacroController(
                bid_mw_norm=bid_mw_norm,
                bid_price_norm=bid_price_norm,
                algo_name=agent_name,
            )

            ep_metrics: list[dict[str, float]] = []
            t0 = time.perf_counter()
            for ep in tqdm(
                range(n_episodes), desc="Episodes", position=2, leave=False
            ):
                m = run_macro_episode(
                    agent=agent,
                    scenario=scenario,
                    seed=seed_start + ep,
                    agent_type="macro",
                    episode_number=ep,
                    inner_action_fn=inner_action_fn,
                    record_transitions=record_transitions,
                )
                ep_metrics.append(m)

            elapsed = time.perf_counter() - t0

            # Aggregate across episodes
            keys = list(ep_metrics[0].keys())
            agg = {
                k: float(np.mean([m[k] for m in ep_metrics])) for k in keys
            }
            std = {
                k: float(np.std([m[k] for m in ep_metrics], ddof=1))
                if n_episodes > 1
                else 0.0
                for k in keys
            }
            agg["survival_rate"] = float(
                np.mean([m["survived"] for m in ep_metrics])
            )

            row = {
                "scenario": scenario,
                "agent": agent_name,
                "n_episodes": n_episodes,
                "wall_time_s": round(elapsed, 2),
                **{
                    k: round(v, 4)
                    for k, v in agg.items()
                    if k != "survived"
                },
                **{
                    f"{k}_std": round(v, 4)
                    for k, v in std.items()
                    if k != "survived"
                },
            }
            rows.append(row)
            tqdm.write(
                f"  {agent_name}/{scenario}  "
                f"reward={agg['mean_reward']:7.2f}  "
                f"accept_rate={agg['bid_acceptance_rate']:.3f}  "
                f"reg_rev={agg['total_reg_revenue']:8.1f}  "
                f"survive={agg['survival_rate']:.2f}  "
                f"flex={agg['mean_flex_kw']:.0f}kW  "
                f"bess={agg['mean_bess_kw']:.0f}kW  "
                f"cool_d={agg['mean_cool_delta_kw']:.0f}kW"
            )

    return rows


# ---------------------------------------------------------------------------
# Display / save helpers
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


def save_csv(
    rows: list[dict[str, Any]], path: Path, append: bool = False
) -> None:
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
        description="C2G-Bench evaluation: default_macro high-level + low-level controller"
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        default=["default_macro+sac"],
        help=(
            "Agents to evaluate. Use default_macro or default_macro+<inner> "
            "where <inner> is one of: "
            "sac, ppo, pid, bang_bang, rule_based, mpc_fast, random"
        ),
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        type=str,
        default=["default"],
        choices=SCENARIOS,
    )
    parser.add_argument("--n_episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument(
        "--model_dir",
        default=None,
        help="Override model directory for SB3 agents (optional)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write the results CSV.",
    )
    parser.add_argument(
        "--record_transitions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable per-step transition logging under runs/",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        default=False,
        help="Append results to existing CSV instead of overwriting",
    )
    parser.add_argument(
        "--bid_mw_norm",
        type=float,
        default=0.50,
        help="Fixed bid MW (normalised [0,1]). Default 0.50 = 50%% of max capacity.",
    )
    parser.add_argument(
        "--bid_price_norm",
        type=float,
        default=0.25,
        help="Fixed bid price (normalised [0,1]). Default 0.25 ≈ 50%% of RMCP.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    evaluation_results_dir = project_root / "evaluation" / "results"
    evaluation_results_dir.mkdir(parents=True, exist_ok=True)

    rows = benchmark(
        agents=args.agents,
        scenarios=args.scenarios,
        n_episodes=args.n_episodes,
        seed_start=args.seed,
        model_dir=args.model_dir,
        record_transitions=args.record_transitions,
        bid_mw_norm=args.bid_mw_norm,
        bid_price_norm=args.bid_price_norm,
    )
    print_results_table(rows)

    if args.output:
        output_path = Path(args.output)
    else:
        agent_tag = args.agents[0] if len(args.agents) == 1 else "multi"
        scenario_tag = (
            args.scenarios[0] if len(args.scenarios) == 1 else "multi"
        )
        output_path = (
            Path("evaluation")
            / "results"
            / f"defmacro_{agent_tag}_{scenario_tag}.csv"
        )
    save_csv(rows, output_path, append=args.append)
