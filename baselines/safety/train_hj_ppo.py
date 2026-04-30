"""
baselines/safety/train_hj_ppo.py  —  PPO with Hamilton-Jacobi Reachability Shield
============================================================================
Trains PPO inside an HJShieldedEnv: the HJ value function is precomputed
offline, and at runtime the shield overrides actions near the BRS boundary.

Usage
-----
  uv run python baselines/safety/train_hj_ppo.py
  uv run python baselines/safety/train_hj_ppo.py scenario=scenario_b
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
from baselines.safety.hj_shield import HJShield, HJShieldedEnv
from baselines.metrics_callback import C2GMetricsCallback


# Pre-compute a single shared HJ shield (offline DP is expensive)
_SHARED_HJ_SHIELD: HJShield | None = None


def _get_shared_hj_shield(delta: float, n_grid: int) -> HJShield:
    global _SHARED_HJ_SHIELD
    if _SHARED_HJ_SHIELD is None:
        print("[HJ-PPO] Precomputing HJ value functions (one-time)...")
        _SHARED_HJ_SHIELD = HJShield(delta=delta, n_grid=n_grid, precompute=True)
        print("[HJ-PPO] HJ precomputation complete.")
    return _SHARED_HJ_SHIELD


def make_hj_env_fn(scenario: str, seed: int, delta: float, n_grid: int):
    def _init():
        base_env = C2GFastEnv(scenario=scenario)
        shield = _get_shared_hj_shield(delta, n_grid)
        env = HJShieldedEnv(base_env, shield=shield)
        env.reset(seed=seed)
        return env
    return _init


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def train(cfg: DictConfig) -> None:
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    print(OmegaConf.to_yaml(cfg))

    scenario = cfg.scenario.env_id
    seed     = cfg.experiment.seed
    algo_cfg = cfg.algo
    log_cfg  = cfg.logging

    delta  = float(getattr(algo_cfg, "hj_delta", 1.0))
    n_grid = int(getattr(algo_cfg, "hj_n_grid", 100))

    print(f"[HJ-PPO] scenario={scenario}  seed={seed}  "
          f"delta={delta}  n_grid={n_grid}  timesteps={algo_cfg.timesteps:,}")

    vec_env = make_vec_env(
        make_hj_env_fn(scenario, seed, delta, n_grid),
        n_envs=algo_cfg.n_envs, seed=seed)
    vec_env = VecNormalize(
        vec_env,
        norm_obs=algo_cfg.norm_obs, norm_reward=algo_cfg.norm_reward,
        clip_obs=algo_cfg.clip_obs, clip_reward=algo_cfg.clip_reward)

    eval_env = make_vec_env(
        make_hj_env_fn(scenario, seed + 999, delta, n_grid),
        n_envs=1, seed=seed + 999)
    eval_env = VecNormalize(
        eval_env, norm_obs=True, norm_reward=False,
        clip_obs=algo_cfg.clip_obs, training=False)

    checkpoint_cb = CheckpointCallback(
        save_freq=max(algo_cfg.eval_freq, 1),
        save_path=str(out_dir / "checkpoints"), name_prefix="ckpt")
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

    model.learn(
        total_timesteps=algo_cfg.timesteps,
        callback=[checkpoint_cb, eval_cb, metrics_cb],
        tb_log_name=cfg.experiment.name,
        reset_num_timesteps=True)

    model.save(str(out_dir / "final_model"))
    vec_env.save(str(out_dir / "vec_normalize.pkl"))
    print(f"\n[HJ-PPO] Training complete → {out_dir.resolve()}")


if __name__ == "__main__":
    train()
