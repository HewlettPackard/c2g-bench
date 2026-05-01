"""
baselines/train_hierarchical_sac.py  —  Rule-Based Low + SAC High Hierarchical
================================================================================
Two-phase hierarchical pipeline:

  Phase 1  Rule-based low-level controller (no training needed)
  Phase 2  Train SAC high-level agent on C2GMacroEnv (15-minute bidding)
           using the rule-based controller as the inner policy

The rule-based controller (RuleBasedMacroController) acts as the frozen
inner policy — it doesn't need training, so Phase 1 is instant.

Usage
-----
  # Default (seed=42, 400k high-level steps)
  python baselines/train_hierarchical_sac.py

  # Custom seed
  python baselines/train_hierarchical_sac.py experiment.seed=100

  # Multiple seeds (Hydra multirun)
  python baselines/train_hierarchical_sac.py --multirun experiment.seed=1,2,3,42,100

  # Custom output directory
  python baselines/train_hierarchical_sac.py hydra.run.dir=outputs/my_experiment/run1

  # Override high-level training steps
  python baselines/train_hierarchical_sac.py +hrl.phase2_steps=800000

  # Override scenario
  python baselines/train_hierarchical_sac.py scenario=scenario_a

Outputs
-------
  outputs/hrl_sac_<scenario>/seed_<N>/<timestamp>/
      phase2/              — SAC macro-level artifacts
      final_model.zip      — the macro-level SAC policy (the one you deploy)
      episode_metrics.csv  — per-episode metrics
"""
from __future__ import annotations
from pathlib import Path

import baselines._hydra_compat  # noqa: F401  # Hydra 1.3.x + Python ≥3.14 fix

import numpy as np
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env

from c2g_env import C2GMacroEnv
from baselines.rule_based_mpc import RuleBasedController
from baselines.metrics_callback import C2GMetricsCallback


# ---------------------------------------------------------------------------
# Inner policy: rule-based low-level controller
# ---------------------------------------------------------------------------

def _make_rule_based_inner_fn():
    """Create a rule-based inner_action_fn for C2GMacroEnv.

    Uses RuleBasedController (low-level, 4D actions for C2GFastEnv).
    """
    controller = RuleBasedController()

    def inner_fn(inner_obs: np.ndarray, macro_action: np.ndarray) -> np.ndarray:
        action, _ = controller.predict(inner_obs, deterministic=True)
        return action.astype(np.float32)

    return inner_fn


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def _make_macro_env(scenario: str, seed: int, inner_fn):
    def _init():
        env = C2GMacroEnv(scenario=scenario, inner_action_fn=inner_fn)
        env.reset(seed=seed)
        return env
    return _init


# ---------------------------------------------------------------------------
# Phase 2: SAC high-level training
# ---------------------------------------------------------------------------

def train_macro_sac(scenario: str, seed: int, timesteps: int,
                    inner_fn, out_dir: Path, log_cfg, algo_cfg: dict) -> Path:
    """Train a SAC agent on C2GMacroEnv with a rule-based inner policy."""
    phase_dir = out_dir / "phase2"
    phase_dir.mkdir(parents=True, exist_ok=True)

    # SAC is off-policy — single env, no VecNormalize needed
    env = _make_macro_env(scenario, seed, inner_fn)()

    eval_env = make_vec_env(_make_macro_env(scenario, seed + 999, inner_fn),
                            n_envs=1, seed=seed + 999)

    # ── Callbacks ─────────────────────────────────────────────────────────
    checkpoint_cb = CheckpointCallback(
        save_freq=algo_cfg.get("eval_freq", 5_000),
        save_path=str(phase_dir / "checkpoints"),
        name_prefix="ckpt",
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(phase_dir / "best_model"),
        log_path=str(phase_dir / "tensorboard"),
        eval_freq=algo_cfg.get("eval_freq", 5_000),
        n_eval_episodes=algo_cfg.get("n_eval_episodes", 5),
        deterministic=True, verbose=0,
    )
    metrics_cb = C2GMetricsCallback(
        print_freq=getattr(log_cfg, "console_freq", 20),
        csv_path=phase_dir / "episode_metrics.csv",
        verbose=1,
    )

    # ── SAC Model ─────────────────────────────────────────────────────────
    net_arch = algo_cfg.get("net_arch", [256, 256])

    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=algo_cfg.get("learning_rate", 3e-4),
        buffer_size=algo_cfg.get("buffer_size", 100_000),
        learning_starts=algo_cfg.get("learning_starts", 2_000),
        batch_size=algo_cfg.get("batch_size", 256),
        tau=algo_cfg.get("tau", 0.005),
        gamma=algo_cfg.get("gamma", 0.99),
        train_freq=algo_cfg.get("train_freq", 1),
        gradient_steps=algo_cfg.get("gradient_steps", 1),
        ent_coef=algo_cfg.get("ent_coef", "auto"),
        target_update_interval=algo_cfg.get("target_update_interval", 1),
        policy_kwargs=dict(net_arch=net_arch),
        tensorboard_log=str(phase_dir / "tensorboard"),
        verbose=0,
        seed=seed,
    )

    # ── Train ─────────────────────────────────────────────────────────────
    print(f"[Phase 2] Training macro SAC (rule-based inner): {timesteps:,} steps")
    model.learn(
        total_timesteps=timesteps,
        callback=[checkpoint_cb, eval_cb, metrics_cb],
        tb_log_name="phase2_sac_macro",
        reset_num_timesteps=True,
    )

    model_path = phase_dir / "final_model"
    model.save(str(model_path))
    print(f"[Phase 2] Complete → {model_path}")
    return model_path


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    out_dir  = Path(HydraConfig.get().runtime.output_dir)
    scenario = cfg.scenario.env_id
    seed     = cfg.experiment.seed
    log_cfg  = cfg.logging

    # Get algo config (use sac.yaml if available)
    algo_cfg = OmegaConf.to_container(cfg.algo, resolve=True) if "algo" in cfg else {}

    # HRL-specific overrides (optional, via +hrl.xxx on CLI)
    hrl_cfg = cfg.get("hrl")
    if hrl_cfg is None:
        hrl = {}
    elif OmegaConf.is_config(hrl_cfg):
        hrl = OmegaConf.to_container(hrl_cfg, resolve=True)
    else:
        hrl = dict(hrl_cfg)
    phase2_steps = hrl.get("phase2_steps", algo_cfg.get("timesteps", 400_000))

    print("═" * 60)
    print("  C2G-Bench: Rule-Based Low + SAC High Hierarchical")
    print(f"  scenario={scenario}  seed={seed}")
    print(f"  Phase 1: Rule-based controller (no training)")
    print(f"  Phase 2: SAC — {phase2_steps:,} steps")
    print("═" * 60)

    # ── Phase 1: Rule-based (instant) ────────────────────────────────────
    inner_fn = _make_rule_based_inner_fn()
    print("[Phase 1] Using RuleBasedMacroController as inner policy (no training)")

    # ── Phase 2: SAC high-level ──────────────────────────────────────────
    macro_model_path = train_macro_sac(
        scenario, seed, phase2_steps,
        inner_fn, out_dir, log_cfg, algo_cfg,
    )

    # Copy final model to top-level for easy discovery
    import shutil
    shutil.copy2(str(macro_model_path) + ".zip",
                 str(out_dir / "final_model.zip"))

    print(f"\n{'═' * 60}")
    print(f"  HRL-SAC training complete → {out_dir.resolve()}")
    print(f"  Inner policy:  RuleBasedMacroController (deterministic)")
    print(f"  Macro model:   {macro_model_path}")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
