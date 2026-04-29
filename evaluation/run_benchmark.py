"""
evaluation/run_benchmark.py  --  Benchmark Evaluation Runner
============================================================
Runs all registered agents on all 4 evaluation scenarios, collects
per-episode metrics, and writes under evaluation/results.

Metrics (per episode)
---------------------
  mean_reward         — mean step reward
  total_reward        — sum of step rewards
  tracking_rmse       — RMSE of regulation signal tracking error (MW)
  thermal_viol_rate   — fraction of ticks with temp > T_warn (33°C)
  throughput_ratio    — mean(p_flex_kw / p_flex_nom_kw) over episode
  bess_degradation    — cumulative cycle ageing fraction * 1e4
  episode_length      — ticks in episode (< episode_ticks means early termination)
  survival_rate       — fraction of episodes that ran to full episode_ticks ticks

Usage
-----
  cd /lustre/guillant/C2G-Macro
  python evaluation/run_benchmark.py                         # rule-based only
  python evaluation/run_benchmark.py --agents rule_based ppo --n_episodes 10
  python evaluation/run_benchmark.py --model_dir trained_models/ppo_default_s42

  # Hierarchical: macro agent + low-level controller
  python evaluation/run_benchmark.py --agents rule_macro --inner-agents random pid bang_bang rule_based
  python evaluation/run_benchmark.py --agents rule_macro+pid rule_macro+bang_bang  # explicit combos

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
  random_macro — np.random uniform over macro action space (bid MW + price)

High-Assurance Agents
  simplex_ppo  — PPO + Simplex safety shield (runtime filter)
  cbf_ppo      — PPO + Control Barrier Function shield
  hj_ppo       — PPO + Hamilton-Jacobi reachability shield
  mpcsf_ppo    — PPO + Model-Predictive Safety Filter
  cpo          — Constrained Policy Optimisation
  reward_shaping — PPO w/ shield-penalty reward shaping
  ha_c2g       — HA-C2G (CBM + safe projection + physics shield)

Tier 3 Ablations
  cbm_only     — PPO + concept bottleneck (no gate, no shield)
  cbm_gate     — PPO + concept bottleneck + trained gate (no shield)
  cbm_shield   — PPO + concept bottleneck + physics shield (no gate)
"""
from __future__ import annotations
import argparse, csv, json, re, time
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from tqdm import tqdm

from c2g_env import C2GFastEnv, C2GMacroEnv
from baselines.rule_based_mpc import RuleBasedController
from baselines.rule_based_macro import RuleBasedMacroController
from baselines.bang_bang import BangBangController
from baselines.pid_controller import PIDController
from baselines.mpc_fast import MPCFastController
from baselines.mpc_macro import MPCMacroController
from baselines.milp_dispatch import MILPDispatchController
from baselines.metrics_callback import C2GTransitionLoggerCallback, build_ablation_suffix, STATE_COLUMNS
from baselines.train_llm_agents import (
    LLMPolicyAgent,
    load_prompt_templates,
    validate_llm_model_id,
)

# ── High-Assurance agents ────────────────────────────────────────
from baselines.safety.cbf_shield import CBFShield, CBFShieldedAgent
from baselines.safety.hj_shield import HJShield
from baselines.safety.mpc_safety_filter import MPCSafetyFilter
from baselines.safety_shield import SafetyShield

SCENARIOS    = ["default", "scenario_a", "scenario_b", "scenario_c"]
T_WARN_NORM  = 33.0 / 35.0   # normalised warning threshold
_RL_ALGOS = {
    "ppo",
    "sac"
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


def _make_macro_env(scenario: str, **kwargs) -> C2GMacroEnv:
    """Return a C2GMacroEnv for macro-level agent evaluation."""
    return C2GMacroEnv(scenario=scenario, **kwargs)


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


_INNER_CONTROLLERS = {"random", "pid", "bang_bang", "rule_based", "mpc_fast", "ppo"}


def _make_inner_controller(
    name: str,
    env: "C2GFastEnv | None" = None,
    scenario: str = "default",
    seed: int = 42,
    model_dir: str | None = None,
):
    """Instantiate a low-level controller by name."""
    if name == "random":
        if env is None:
            raise ValueError("env must be provided to build a random inner controller")
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
    raise ValueError(f"Unknown inner controller '{name}'. Choose from: {_INNER_CONTROLLERS}")


def _infer_agent_type(agent_name: str) -> str:
    """Classify benchmark agents as macro vs hardware controllers."""
    macro_agents = {"rule_macro", "random_macro", "mpc_macro", "milp", "ppo_macro"}
    # Hierarchical combos like rule_macro+pid are also macro agents
    if "+" in agent_name:
        return "macro"
    return "macro" if agent_name in macro_agents else "hardware"


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

    agent_types = {_infer_agent_type(name) for name in unique_agents}
    if not agent_types:
        agent_type_tag = "unknown"
    elif len(agent_types) == 1:
        agent_type_tag = next(iter(agent_types))
    else:
        agent_type_tag = "mixed"

    ablation_suffix = build_ablation_suffix(fixed_action_values)
    ablation_tag = ablation_suffix.lstrip("_") if ablation_suffix else "base"

    return Path("evaluation") / "results" / f"{algo_tag}_{scenario_tag}_{agent_type_tag}_{ablation_tag}.csv"


# ---------------------------------------------------------------------------
# Agent wrappers — expose a common predict(obs) -> (action, state) interface
# ---------------------------------------------------------------------------

class RandomAgent:
    """Samples uniformly from the action space."""
    def __init__(self, env: C2GFastEnv, algo_name: str = "random"):
        self._space = env.action_space
        self.algo_name = algo_name

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        return self._space.sample(), None


class MacroRandomAgent:
    """Samples uniformly from the macro (2-D) action space."""
    def __init__(self, env: C2GMacroEnv, algo_name: str = "random_macro"):
        self._space = env.action_space
        self.algo_name = algo_name

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        return self._space.sample(), None


class SB3Agent:
    """Wraps a loaded SB3 model (PPO or SAC)."""
    def __init__(self, model, algo_name: str, obs_normalizer=None):
        self._model = model
        self.algo_name = algo_name
        self._obs_normalizer = obs_normalizer

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        if self._obs_normalizer is not None:
            obs = self._obs_normalizer.normalize_obs(np.asarray(obs, dtype=np.float32))
        return self._model.predict(obs, deterministic=deterministic)


def _resolve_sb3_spec(algo: str):
    from stable_baselines3 import PPO, SAC

    algo_key = algo.lower()
    if algo_key == "sac":
        return SAC, "sac", False
    if algo_key in _RL_ALGOS:
        train_key_map = {
            "ppo_lag": "ppo_lagrangian",
            "reward_shaping": "shield_reward_shaping",
        }
        return PPO, train_key_map.get(algo_key, algo_key), True
    raise ValueError(f"Unknown algo '{algo}'. Use ppo, sac, ppo_lag, ppo_lagrangian, or a registered PPO-style benchmark agent.")


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
        path = Path("trained_models") / f"{train_key}_{scenario}_s{seed}" / "final_model"
    if not path.with_suffix(".zip").exists():
        raise FileNotFoundError(
            f"No trained model at {path}.zip — run baselines/train_{algo}.py first."
        )
    model = cls.load(str(path))
    obs_normalizer = None
    if should_restore_norm:
        obs_normalizer = _maybe_load_obs_normalizer(path.parent / "vec_normalize.pkl", scenario)
    return SB3Agent(model, algo_name=algo.lower(), obs_normalizer=obs_normalizer)


class EvolutionaryAgent:
    """Wraps a CMA-ES or PSO linear policy loaded from .npz."""
    def __init__(self, npz_path: str | Path, algo_name: str):
        data = np.load(npz_path)
        self.W = data["W"]
        self.b = data["b"]
        self.act_low = data["act_low"]
        self.act_high = data["act_high"]
        self.algo_name = algo_name

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        if obs.ndim == 1:
            action = np.clip(self.W @ obs + self.b, self.act_low, self.act_high)
        else:
            action = np.clip(obs @ self.W.T + self.b, self.act_low, self.act_high)
        return action.astype(np.float32), None


class ShieldedSB3Agent:
    """Wraps an SB3 model with a runtime safety shield."""
    def __init__(self, base_agent, shield):
        self._base_agent = base_agent
        self._shield = shield
        self.algo_name = getattr(base_agent, "algo_name", "shielded_sb3")

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        action, state = self._base_agent.predict(obs, deterministic=deterministic)
        safe_action, _, _ = self._shield.filter(action, obs)
        return safe_action, state


# ── High-Assurance agents ────────────────────────────────────────
# Metric collection
# ---------------------------------------------------------------------------

def run_episode(
    agent,
    scenario: str,
    seed: int,
    algo_name: str | None = None,
    agent_type: str = "hardware",
    episode_number: int = 0,
    record_transitions: bool = True,
    fixed_action_values: dict[str, float] | None = None,
) -> dict[str, float]:
    """Run one episode and return a metrics dict."""
    env = _make_env(
        scenario=scenario,
        fixed_action_values=fixed_action_values,
    )
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
            fixed_action_values=fixed_action_values,
            verbose=0,
        )

    rewards       : list[float] = []
    tracking_errs : list[float] = []
    thermal_viols : int = 0
    throughputs   : list[float] = []
    bess_init_age : float | None = None
    # Cumulative power metrics
    cumul_p_pump_mw     : float = 0.0
    cumul_p_hvac_mw     : float = 0.0
    cumul_flex_reduction_kw : float = 0.0
    cumul_bess_actual_kw : float = 0.0

    done = False
    try:
      while not done:
        state = obs.copy()
        if getattr(agent, "uses_env_context", False):
            action, _ = agent.predict(obs, deterministic=True, env=env, scenario=scenario)
        else:
            action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if transition_logger is not None:
            reward_components = {
                k: info[k] for k in (
                    "reward_throughput", "reward_tracking", "reward_thermal",
                    "reward_soc", "reward_freq", "reward_volt", "reward_backlog",
                )
            }

            print(f"[Transition] Step reward: {reward:.3f}, components: {reward_components}", flush=True)
            transition_logger.record_transition(
                state=state,
                action=action,
                observation=obs,
                reward=float(reward),
                done=done,
                reward_components=reward_components,
            )

        rewards.append(float(reward))

        # Thermal violations
        if obs[0] >= T_WARN_NORM or obs[1] >= T_WARN_NORM:
            thermal_viols += 1

        # Tracking error: reuse the env's own computation (against prev regd signal)
        tracking_errs.append(info["tracking_err_kw"] ** 2)

        # Throughput ratio: flex_kw / flex_nom_kw
        tp = info.get("throughput_ratio", None)
        if tp is not None:
            throughputs.append(float(tp))

        # Accumulate power metrics
        cumul_p_pump_mw += float(info.get("p_pump_mw", 0.0))
        cumul_p_hvac_mw += float(info.get("p_hvac_mw", 0.0))
        cumul_flex_reduction_kw += float(info.get("flex_reduction_kw", 0.0))
        cumul_bess_actual_kw += float(info.get("bess_actual_kw", 0.0))

        # BESS degradation
        if bess_init_age is None:
            bess_init_age = float(info.get("bess_age_frac", 0.0))

    finally:
        if transition_logger is not None:
            transition_logger.close()

    bess_final_age = float(info.get("bess_age_frac", bess_init_age or 0.0))
    bess_degradation = (bess_final_age - (bess_init_age or 0.0)) * 1e4

    n_ticks = len(rewards)
    survived   = 1.0 if n_ticks >= env._episode_ticks else 0.0

    return {
        "mean_reward"       : float(np.mean(rewards)),
        "total_reward"      : float(np.sum(rewards)),
        "tracking_rmse"     : float(np.sqrt(np.mean(tracking_errs))) if tracking_errs else 0.0,
        "thermal_viol_rate" : thermal_viols / max(n_ticks, 1),
        "throughput_ratio"  : float(np.mean(throughputs)) if throughputs else 0.0,
        "bess_degradation"  : bess_degradation,
        "episode_length"    : n_ticks,
        "survived"          : survived,
        "cumul_p_pump_mw"   : cumul_p_pump_mw,
        "cumul_p_hvac_mw"   : cumul_p_hvac_mw,
        "cumul_flex_reduction_kw" : cumul_flex_reduction_kw,
        "cumul_bess_actual_kw" : cumul_bess_actual_kw,
    }


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

    # Extract static env attributes once for LLM agents that need them
    macro_env_info: dict[str, Any] = {
        "committed_mw_max": float(getattr(env, "_committed_max_mw", 30.0)),
        "dr_baseline_mw":   float(getattr(env, "_dr_baseline_mw", 5.0)),
        "bess_p_max_mw":    float(getattr(env._fast_env._bess, "P_MAX_MW", 5.0)),
    } if getattr(agent, "uses_env_context", False) else {}

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
    # Per-lever tracking contributions (mean across sub-steps, per macro tick)
    flex_reductions: list[float] = []
    bess_actuals: list[float] = []
    cool_deltas: list[float] = []
    p_pumps: list[float] = []
    p_hvacs: list[float] = []

    done = False
    try:
      while not done:
        state = obs.copy()
        if macro_env_info:
            action, _ = agent.predict(obs, deterministic=True, static_env_info=macro_env_info,
                                      scenario=scenario)
        else:
            action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if transition_logger is not None:
            reward_components = {
                k: info.get(k, 0.0) for k in (
                    "reward_regulation","reward_sub", "reward_elec",  "reward_churn"
                )
            }
            print(f"[Transition] Step reward: {reward:.3f}, components: {reward_components}", flush=True)
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
        # Per-lever tracking contributions (mean over 180 sub-steps)
        flex_reductions.append(float(info.get("mean_flex_reduction_kw", 0.0)))
        bess_actuals.append(float(info.get("mean_bess_actual_kw", 0.0)))
        cool_deltas.append(float(info.get("mean_cool_delta_kw", 0.0)))
        p_pumps.append(float(info.get("mean_p_pump_mw", 0.0)))
        p_hvacs.append(float(info.get("mean_p_hvac_mw", 0.0)))
    finally:
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
    }


def benchmark(
    agents      : list[str],
    scenarios   : list[str],
    n_episodes  : int,
    seed_start  : int,
    model_dir   : str | None,
    record_transitions: bool = False,
    fixed_action_values: dict[str, float] | None = None,
    llm_model_id: str | None = None,
    llm_api_base: str = "http://localhost:8000/v1",
    llm_mode: str = "hardware",
    llm_template_path: str = "conf/chat_templates/run_benchmark.yaml",
    llm_max_new_tokens: int = 9216,
    llm_temperature: float = 0.0,
    llm_enable_thinking: bool = True,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    if "llm_policy" in set(agents):
        llm_model_id = validate_llm_model_id(str(llm_model_id or ""))

    if llm_mode not in ["hardware", "macro"]:
        raise ValueError(f"Invalid agent mode '{llm_mode}'. Must be 'hardware' or 'macro'.")

    if record_transitions:
        project_root = Path(__file__).resolve().parent.parent
        (project_root / "runs").mkdir(parents=True, exist_ok=True)

    prompt_templates = load_prompt_templates(llm_template_path)

    scenario_bar = tqdm(scenarios, desc="Scenarios", position=0)
    for scenario in scenario_bar:
        scenario_bar.set_description(f"Scenario: {scenario}")

        agent_bar = tqdm(agents, desc="Agents", position=1, leave=False)
        for agent_name in agent_bar:
            agent_bar.set_description(f"Agent: {agent_name}")
            agent_type = llm_mode if agent_name == "llm_policy" else _infer_agent_type(agent_name)

            # Instantiate agent once per (agent, scenario) to share weights
            if agent_type == "macro":
                env_for_space = _make_macro_env(scenario=scenario)
            else:
                env_for_space = _make_env(
                    scenario=scenario,
                    fixed_action_values=fixed_action_values,
                )
            env_for_space.reset(seed=0)

            # ── Hierarchical combo agents (e.g. rule_macro+pid) ────
            inner_action_fn = None
            macro_part = agent_name
            if "+" in agent_name:
                macro_part, inner_part = agent_name.split("+", 1)
                inner_env = _make_env(scenario=scenario)
                inner_env.reset(seed=0)
                inner_ctrl = _make_inner_controller(
                    inner_part, env=inner_env,
                    scenario=scenario, seed=seed_start, model_dir=model_dir,
                )
                inner_action_fn = lambda obs, _act, c=inner_ctrl: c.predict(obs)[0]

            if agent_name == "rule_based":
                agent = RuleBasedController()
            elif macro_part == "rule_macro":
                agent = RuleBasedMacroController()
            elif macro_part == "random_macro":
                agent = MacroRandomAgent(env_for_space, algo_name=agent_name)
            elif agent_name == "bang_bang":
                agent = BangBangController()
            elif agent_name == "pid":
                agent = PIDController()
            elif agent_name == "mpc_fast":
                agent = MPCFastController()
            elif macro_part == "mpc_macro":
                agent = MPCMacroController()
            elif macro_part == "milp":
                agent = MILPDispatchController()
            elif agent_name == "random":
                agent = RandomAgent(env_for_space, algo_name=agent_name)
            elif agent_name in ("cmaes", "pso"):
                npz_name = f"{agent_name}_policy.npz"
                npz_dir = Path(model_dir) if model_dir else Path("trained_models") / f"{agent_name}_{scenario}_s{seed_start}"
                npz_path = npz_dir / npz_name
                if not npz_path.exists():
                    print(f"    SKIP: No trained policy at {npz_path}")
                    continue
                agent = EvolutionaryAgent(npz_path, algo_name=agent_name)
            elif agent_name == "llm_policy":
                try:
                    state_names = [name.removeprefix("s_") for name in STATE_COLUMNS]
                    agent = LLMPolicyAgent(
                        model_id=llm_model_id,
                        mode=llm_mode,
                        prompts=prompt_templates,
                        state_names=state_names,
                        max_new_tokens=llm_max_new_tokens,
                        temperature=llm_temperature,
                        api_base=llm_api_base,
                        enable_thinking=llm_enable_thinking,
                    )
                except Exception as exc:
                    print(f"    SKIP: Failed to initialize llm_policy agent: {exc}")
                    continue
            # ── High-Assurance shielded agents ─────────────────
            elif agent_name == "simplex_ppo":
                try:
                    base = load_sb3_agent("ppo", scenario, seed_start, model_dir)
                    agent = ShieldedSB3Agent(base, SafetyShield())
                except FileNotFoundError as exc:
                    print(f"    SKIP: {exc}")
                    continue
            elif agent_name == "cbf_ppo":
                try:
                    base = load_sb3_agent("ppo", scenario, seed_start, model_dir)
                    agent = ShieldedSB3Agent(base, CBFShield())
                except FileNotFoundError as exc:
                    print(f"    SKIP: {exc}")
                    continue
            elif agent_name == "hj_ppo":
                try:
                    base = load_sb3_agent("ppo", scenario, seed_start, model_dir)
                    agent = ShieldedSB3Agent(base, HJShield(precompute=True))
                except FileNotFoundError as exc:
                    print(f"    SKIP: {exc}")
                    continue
            elif agent_name == "mpcsf_ppo":
                try:
                    base = load_sb3_agent("ppo", scenario, seed_start, model_dir)
                    agent = ShieldedSB3Agent(base, MPCSafetyFilter())
                except FileNotFoundError as exc:
                    print(f"    SKIP: {exc}")
                    continue
            elif agent_name in ("cpo", "reward_shaping", "ha_c2g", "cbm_only", "cbm_gate", "cbm_shield"):
                try:
                    agent = load_sb3_agent(agent_name, scenario, seed_start, model_dir)
                except FileNotFoundError as exc:
                    print(f"    SKIP: {exc}")
                    continue
            else:
                try:
                    agent = load_sb3_agent(agent_name, scenario, seed_start, model_dir)
                except FileNotFoundError as exc:
                    print(f"    SKIP: {exc}")
                    continue

            if not hasattr(agent, "algo_name"):
                setattr(agent, "algo_name", agent_name)

            ep_metrics: list[dict[str, float]] = []
            t0 = time.perf_counter()
            for ep in tqdm(range(n_episodes), desc="Episodes", position=2, leave=False):
                if agent_type == "macro":
                    m = run_macro_episode(
                        agent=agent,
                        scenario=scenario,
                        seed=seed_start + ep,
                        agent_type=agent_type,
                        episode_number=ep,
                        inner_action_fn=inner_action_fn,
                        record_transitions=record_transitions,
                    )
                else:
                    m = run_episode(
                        agent=agent,
                        scenario=scenario,
                        seed=seed_start + ep,
                        agent_type=agent_type,
                        episode_number=ep,
                        record_transitions=record_transitions,
                        fixed_action_values=fixed_action_values,
                    )
                ep_metrics.append(m)

            elapsed = time.perf_counter() - t0

            # Aggregate across episodes (mean + std)
            keys = list(ep_metrics[0].keys())
            agg  = {k: float(np.mean([m[k] for m in ep_metrics])) for k in keys}
            std  = {k: float(np.std([m[k] for m in ep_metrics], ddof=1)) if n_episodes > 1 else 0.0 for k in keys}
            agg["survival_rate"] = float(np.mean([m["survived"] for m in ep_metrics]))

            row = {
                "scenario"          : scenario,
                "agent"             : agent_name,
                "n_episodes"        : n_episodes,
                "wall_time_s"       : round(elapsed, 2),
                **{k: round(v, 4) for k, v in agg.items() if k != "survived"},
                **{f"{k}_std": round(v, 4) for k, v in std.items() if k != "survived"},
            }
            rows.append(row)
            if agent_type == "macro":
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
            else:
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
    print(f"\nResults saved -> {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="C2G-Bench evaluation runner")
    parser.add_argument(
        "--agents", nargs="+",
        default=["rule_based", "bang_bang", "pid", "random"],
        help="Agents to evaluate: rule_based rule_macro random_macro bang_bang pid mpc_fast "
             "mpc_macro milp ppo sac ppo_lag cmaes pso random "
               "simplex_ppo cbf_ppo hj_ppo mpcsf_ppo cpo reward_shaping ha_c2g "
                             "llm_policy "
             "cbm_only cbm_gate cbm_shield",
    )
    parser.add_argument(
        "--scenarios", nargs="+",
        default=SCENARIOS,
        choices=SCENARIOS,
    )
    parser.add_argument(
        "--inner-agents", nargs="+",
        default=None,
        help="Low-level controllers to pair with each macro agent "
             "(e.g. --inner-agents pid bang_bang rule_based). "
             "Creates hierarchical combos like rule_macro+pid.",
    )
    parser.add_argument("--n_episodes", type=int, default=5)
    parser.add_argument("--seed",       type=int, default=100)
    parser.add_argument(
        "--model_dir", default=None,
        help="Override model directory for SB3 agents (optional)",
    )
    parser.add_argument(
        "--output", default=None,
        help=(
            "Path to write the results CSV. If omitted, defaults to "
            "evaluation/results/algoname_scenario_agenttype_ablation.csv"
        ),
    )
    parser.add_argument(
        "--record_transitions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable/disable per-step transition logging under runs/<algo>_<scenario>_<agent>/transitions_<episode>.csv",
    )
    parser.add_argument(
        "--fixed-action",
        action="append",
        default=[],
        help="Optional fixed value for a disabled action, e.g. --fixed-action hvac_effort=0.8",
    )
    parser.add_argument(
        "--llm-model-id",
        default="HuggingFaceTB/SmolLM2-360M-Instruct",
        help="Model name to query on vLLM server (e.g., org/model)",
    )
    parser.add_argument(
        "--llm-api-base",
        default="http://localhost:8000/v1",
        help="vLLM server base URL for OpenAI-compatible API",
    )
    parser.add_argument(
        "--llm-mode",
        choices=["hardware", "macro"],
        default="hardware",
        help="Control mode for llm_policy agent",
    )
    parser.add_argument(
        "--llm-template-path",
        default="conf/chat_templates/run_benchmark.yaml",
        help="YAML path with hardware_prompt and macro_prompt templates",
    )
    parser.add_argument(
        "--llm-max-new-tokens",
        type=int,
        default=9216,
        help="Max new tokens for llm_policy generation. vLLM has no per-request thinking-budget "
             "parameter, so this is the only knob: the model fills <think> first, then emits JSON. "
             "Default 9216 = ~8704 thinking tokens + ~512 for the JSON action output. "
             "Must be less than the server's --max-model-len minus the prompt length (~2500 tokens).",
    )
    parser.add_argument(
        "--llm-temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for llm_policy generation (0 = greedy)",
    )
    parser.add_argument(
        "--llm-no-thinking",
        dest="llm_enable_thinking",
        action="store_false",
        default=True,
        help="Disable <think> reasoning for all LLM agents (faster, lower token cost).",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    evaluation_results_dir = project_root / "evaluation" / "results"
    if not (evaluation_results_dir.exists() and evaluation_results_dir.is_dir()):
        evaluation_results_dir.mkdir(parents=True, exist_ok=True)

    try:
        fixed_action_values = _parse_fixed_action_args(args.fixed_action)
    except ValueError as exc:
        parser.error(str(exc))

    # Expand --inner-agents: for each macro agent × inner agent, add a combo
    agents = list(args.agents)
    if args.inner_agents:
        macro_agents_in_list = [a for a in agents if _infer_agent_type(a) == "macro"]
        for macro_name in macro_agents_in_list:
            for inner_name in args.inner_agents:
                combo = f"{macro_name}+{inner_name}"
                if combo not in agents:
                    agents.append(combo)

    llm_model_id = args.llm_model_id
    if "llm_policy" in set(args.agents):
        try:
            llm_model_id = validate_llm_model_id(args.llm_model_id)
        except (ValueError, ImportError) as exc:
            parser.error(str(exc))

    rows = benchmark(
        agents     = agents,
        scenarios  = args.scenarios,
        n_episodes = args.n_episodes,
        seed_start = args.seed,
        model_dir  = args.model_dir,
        record_transitions = args.record_transitions,
        fixed_action_values = fixed_action_values,
        llm_model_id = llm_model_id,
        llm_api_base = args.llm_api_base,
        llm_mode = args.llm_mode,
        llm_template_path = args.llm_template_path,
        llm_max_new_tokens = args.llm_max_new_tokens,
        llm_temperature = args.llm_temperature,
        llm_enable_thinking = args.llm_enable_thinking,
    )
    print_results_table(rows)
    output_path = (
        Path(args.output)
        if args.output
        else _default_output_path(
            agents=agents,
            scenarios=args.scenarios,
            fixed_action_values=fixed_action_values,
        )
    )
    save_csv(rows, output_path)
