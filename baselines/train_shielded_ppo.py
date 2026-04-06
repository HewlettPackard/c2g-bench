"""
baselines/train_shielded_ppo.py  —  PPO with Safety Shield (High-Assurance)
=============================================================================
Trains PPO inside a ShieldedEnv: every action the agent takes is filtered
through the Simplex safety shield before reaching the physics simulator.

This produces a **high-assurance controller** — the resulting policy:
  1. Learns to avoid unsafe regions (reward shaping from shield overrides)
  2. Has hard safety guarantees at deployment (shield is always active)
  3. Reports shield intervention rate as a key metric

The shield intervention rate should decrease over training as the agent
learns the safe operating envelope.  A well-trained shielded agent
achieves near-zero intervention rate while maintaining full safety.

Usage
-----
  python baselines/train_shielded_ppo.py algo=ppo

  # Compare shielded vs unshielded:
  python baselines/train_ppo.py              # unshielded
  python baselines/train_shielded_ppo.py     # shielded
"""
from __future__ import annotations
from pathlib import Path

import baselines._hydra_compat  # noqa: F401  # Hydra 1.3.x + Python ≥3.14 fix

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from c2g_env import C2GFastEnv
from baselines.safety_shield import SafetyShield, ShieldedEnv
from baselines.metrics_callback import C2GMetricsCallback


def make_shielded_env_fn(scenario: str, seed: int):
    def _init():
        base_env = C2GFastEnv(scenario=scenario)
        env = ShieldedEnv(base_env, shield=SafetyShield())
        env.reset(seed=seed)
        return env
    return _init


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def train(cfg: DictConfig) -> None:
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    print(OmegaConf.to_yaml(cfg))

    scenario  = cfg.scenario.env_id
    seed      = cfg.experiment.seed
    algo_cfg  = cfg.algo
    log_cfg   = cfg.logging

    print(f"[Shielded-PPO] scenario={scenario}  seed={seed}  "
          f"timesteps={algo_cfg.timesteps:,}")

    # ── Environments (all shielded) ───────────────────────────────────
    vec_env = make_vec_env(make_shielded_env_fn(scenario, seed),
                           n_envs=algo_cfg.n_envs, seed=seed)
    vec_env = VecNormalize(
        vec_env,
        norm_obs    = algo_cfg.norm_obs,
        norm_reward = algo_cfg.norm_reward,
        clip_obs    = algo_cfg.clip_obs,
        clip_reward = algo_cfg.clip_reward,
    )

    eval_env = make_vec_env(make_shielded_env_fn(scenario, seed + 999),
                            n_envs=1, seed=seed + 999)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False,
                            clip_obs=algo_cfg.clip_obs, training=False)

    # ── Callbacks ─────────────────────────────────────────────────────
    checkpoint_cb = CheckpointCallback(
        save_freq   = max(algo_cfg.eval_freq, 1),
        save_path   = str(out_dir / "checkpoints"),
        name_prefix = "ckpt",
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = str(out_dir / "best_model"),
        log_path             = str(out_dir / "tensorboard"),
        eval_freq            = algo_cfg.eval_freq,
        n_eval_episodes      = algo_cfg.n_eval_episodes,
        deterministic        = True,
        verbose              = 0,
    )
    metrics_cb = C2GMetricsCallback(
        print_freq = log_cfg.console_freq,
        csv_path   = out_dir / "episode_metrics.csv" if log_cfg.csv else None,
        verbose    = 1,
    )

    # ── Model ─────────────────────────────────────────────────────────
    net_arch = OmegaConf.to_container(algo_cfg.net_arch, resolve=True)

    model = PPO(
        policy          = "MlpPolicy",
        env             = vec_env,
        learning_rate   = algo_cfg.learning_rate,
        n_steps         = algo_cfg.n_steps,
        batch_size      = algo_cfg.batch_size,
        n_epochs        = algo_cfg.n_epochs,
        gamma           = algo_cfg.gamma,
        gae_lambda      = algo_cfg.gae_lambda,
        clip_range      = algo_cfg.clip_range,
        ent_coef        = algo_cfg.ent_coef,
        vf_coef         = algo_cfg.vf_coef,
        max_grad_norm   = algo_cfg.max_grad_norm,
        policy_kwargs   = dict(net_arch=net_arch),
        tensorboard_log = str(out_dir / "tensorboard") if log_cfg.tensorboard else None,
        verbose         = 0,
        seed            = seed,
    )

    # ── Train ─────────────────────────────────────────────────────────
    model.learn(
        total_timesteps     = algo_cfg.timesteps,
        callback            = [checkpoint_cb, eval_cb, metrics_cb],
        tb_log_name         = cfg.experiment.name,
        reset_num_timesteps = True,
    )

    model.save(str(out_dir / "final_model"))
    vec_env.save(str(out_dir / "vec_normalize.pkl"))
    print(f"\n[Shielded-PPO] Training complete → {out_dir.resolve()}")


if __name__ == "__main__":
    train()
