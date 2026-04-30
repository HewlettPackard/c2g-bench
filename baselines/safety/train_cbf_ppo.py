"""
baselines/safety/train_cbf_ppo.py  —  PPO with CBF Safety Filter
==========================================================
Trains PPO inside a CBFShieldedEnv: every action is projected into the
CBF-safe set via a QP solver before reaching the physics simulator.

The CBF filter is more permissive than the Simplex shield because it
exploits the system dynamics model (barrier function derivatives).

Usage
-----
  uv run python baselines/safety/train_cbf_ppo.py
  uv run python baselines/safety/train_cbf_ppo.py scenario=scenario_b
"""
from __future__ import annotations
from pathlib import Path

import baselines._hydra_compat  # noqa: F401

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from c2g_env import C2GFastEnv
from baselines.safety.cbf_shield import CBFShield, CBFShieldedEnv
from baselines.metrics_callback import C2GMetricsCallback


def make_cbf_env_fn(scenario: str, seed: int, margin: float = 0.5):
    def _init():
        base_env = C2GFastEnv(scenario=scenario)
        shield = CBFShield(margin=margin)
        env = CBFShieldedEnv(base_env, shield=shield)
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

    margin = float(getattr(algo_cfg, "cbf_margin", 0.5))

    print(f"[CBF-PPO] scenario={scenario}  seed={seed}  "
          f"margin={margin}  timesteps={algo_cfg.timesteps:,}")

    # ── Environments (all CBF-shielded) ──────────────────────────
    vec_env = make_vec_env(
        make_cbf_env_fn(scenario, seed, margin),
        n_envs=algo_cfg.n_envs, seed=seed)
    vec_env = VecNormalize(
        vec_env,
        norm_obs=algo_cfg.norm_obs, norm_reward=algo_cfg.norm_reward,
        clip_obs=algo_cfg.clip_obs, clip_reward=algo_cfg.clip_reward)

    eval_env = make_vec_env(
        make_cbf_env_fn(scenario, seed + 999, margin),
        n_envs=1, seed=seed + 999)
    eval_env = VecNormalize(
        eval_env, norm_obs=True, norm_reward=False,
        clip_obs=algo_cfg.clip_obs, training=False)

    # ── Callbacks ────────────────────────────────────────────────
    checkpoint_cb = CheckpointCallback(
        save_freq=max(algo_cfg.eval_freq, 1),
        save_path=str(out_dir / "checkpoints"),
        name_prefix="ckpt")
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(out_dir / "best_model"),
        log_path=str(out_dir / "tensorboard"),
        eval_freq=algo_cfg.eval_freq,
        n_eval_episodes=algo_cfg.n_eval_episodes,
        deterministic=True, verbose=0)
    metrics_cb = C2GMetricsCallback(
        print_freq=log_cfg.console_freq,
        csv_path=out_dir / "episode_metrics.csv" if log_cfg.csv else None,
        verbose=1)

    # ── Model ────────────────────────────────────────────────────
    net_arch = OmegaConf.to_container(algo_cfg.net_arch, resolve=True)

    model = PPO(
        policy="MlpPolicy", env=vec_env,
        learning_rate=algo_cfg.learning_rate,
        n_steps=algo_cfg.n_steps,
        batch_size=algo_cfg.batch_size,
        n_epochs=algo_cfg.n_epochs,
        gamma=algo_cfg.gamma,
        gae_lambda=algo_cfg.gae_lambda,
        clip_range=algo_cfg.clip_range,
        ent_coef=algo_cfg.ent_coef,
        vf_coef=algo_cfg.vf_coef,
        max_grad_norm=algo_cfg.max_grad_norm,
        policy_kwargs=dict(net_arch=net_arch),
        tensorboard_log=str(out_dir / "tensorboard") if log_cfg.tensorboard else None,
        verbose=0, seed=seed)

    # ── Train ────────────────────────────────────────────────────
    model.learn(
        total_timesteps=algo_cfg.timesteps,
        callback=[checkpoint_cb, eval_cb, metrics_cb],
        tb_log_name=cfg.experiment.name,
        reset_num_timesteps=True)

    model.save(str(out_dir / "final_model"))
    vec_env.save(str(out_dir / "vec_normalize.pkl"))
    print(f"\n[CBF-PPO] Training complete → {out_dir.resolve()}")


if __name__ == "__main__":
    train()
