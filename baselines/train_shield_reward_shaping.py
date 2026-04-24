"""
baselines/train_shield_reward_shaping.py  —  Fixed Shield-Penalty Reward Shaping
==================================================================================
Augments the step reward with fixed, engineered penalty functions derived
from distance-to-constraint-boundary. Unlike PPO-Lagrangian (adaptive
multipliers) or CPO (constrained updates), this approach uses
**pre-designed penalty functions** that do not adapt during training.

The shaped reward is:
  r' = r
     − w_thermal · max(0, T_max − T_warn)²
     − w_soc     · max(0, SOC_min + 0.05 − SOC)² + max(0, SOC − SOC_max + 0.05)²
     − w_freq    · max(0, |Δf| − 0.2)²
     − w_voltage · max(0, V_min + 0.02 − V_pcc)²
     − w_shield  · I(shield_intervened)

The quadratic penalties provide smooth gradients near constraint boundaries,
while the fixed shield intervention penalty (from the SC26 paper) creates a
discrete cost for triggering the safety shield.

Usage
-----
  uv run python baselines/train_shield_reward_shaping.py algo=shield_reward_shaping
  uv run python baselines/train_shield_reward_shaping.py algo=shield_reward_shaping scenario=scenario_b
"""
from __future__ import annotations

import logging
from pathlib import Path

import baselines._hydra_compat  # noqa: F401

import hydra
from omegaconf import DictConfig

import numpy as np
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback, CheckpointCallback, EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from c2g_env import C2GFastEnv
from baselines.safety_shield import SafetyShield
from baselines.metrics_callback import C2GMetricsCallback

log = logging.getLogger(__name__)

# Obs indices
_I_TEMP_A   = 0
_I_TEMP_B   = 1
_I_SOC      = 2
_I_FREQ_DEV = 14
_I_VPCC     = 15

_T_SAFE = 35.0
_T_WARN = 33.0
_SOC_MIN = 0.10
_SOC_MAX = 0.95


class ShieldRewardShapingWrapper(gym.Wrapper):
    """
    Applies:
      1. Fixed quadratic penalty functions near constraint boundaries
      2. A discrete penalty every time the safety shield intervenes
      3. The Simplex safety shield for hard constraint satisfaction

    This is "shield-in-the-loop" training with engineered reward shaping.
    """

    def __init__(
        self,
        env: gym.Env,
        shield: SafetyShield,
        w_thermal: float = 2.0,
        w_soc: float = 1.0,
        w_freq: float = 1.5,
        w_voltage: float = 1.0,
        w_shield: float = 0.5,
    ):
        super().__init__(env)
        self.shield = shield
        self.w_thermal = w_thermal
        self.w_soc = w_soc
        self.w_freq = w_freq
        self.w_voltage = w_voltage
        self.w_shield = w_shield
        self._last_obs = None

    def reset(self, **kwargs):
        self.shield.reset()
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        return obs, info

    def step(self, action):
        # Apply safety shield
        obs_prev = self._last_obs if self._last_obs is not None else np.zeros(18, dtype=np.float32)
        safe_action, was_modified, shield_info = self.shield.filter(action, obs_prev)

        obs, reward, terminated, truncated, info = self.env.step(safe_action)
        self._last_obs = obs

        # ── Quadratic penalties ──────────────────────────────────
        T_A = float(obs[_I_TEMP_A]) * _T_SAFE
        T_B = float(obs[_I_TEMP_B]) * _T_SAFE
        soc = float(obs[_I_SOC])
        freq_dev = abs(float(obs[_I_FREQ_DEV]) * 0.5)
        v_pcc = float(obs[_I_VPCC])

        T_max = max(T_A, T_B)
        thermal_penalty = self.w_thermal * max(0.0, T_max - _T_WARN) ** 2

        soc_penalty = self.w_soc * (
            max(0.0, _SOC_MIN + 0.05 - soc) ** 2 +
            max(0.0, soc - _SOC_MAX + 0.05) ** 2
        )

        freq_penalty = self.w_freq * max(0.0, freq_dev - 0.2) ** 2

        voltage_penalty = self.w_voltage * max(0.0, 0.92 - v_pcc) ** 2

        # ── Shield intervention penalty ──────────────────────────
        shield_penalty = self.w_shield if was_modified else 0.0

        # ── Total shaped reward ──────────────────────────────────
        total_penalty = thermal_penalty + soc_penalty + freq_penalty + voltage_penalty + shield_penalty
        shaped_reward = reward - total_penalty

        info.update(shield_info)
        info["shield_stats"] = self.shield.stats.as_dict()
        info["reward_shaping"] = {
            "base_reward": float(reward),
            "thermal_penalty": float(thermal_penalty),
            "soc_penalty": float(soc_penalty),
            "freq_penalty": float(freq_penalty),
            "voltage_penalty": float(voltage_penalty),
            "shield_penalty": float(shield_penalty),
            "total_penalty": float(total_penalty),
        }

        return obs, shaped_reward, terminated, truncated, info


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def train(cfg: DictConfig) -> None:
    scenario = cfg.scenario.name
    algo_cfg = cfg.algo
    seed     = cfg.experiment.seed
    out_dir  = Path(".")

    w_thermal = float(getattr(algo_cfg, "w_thermal", 2.0))
    w_soc     = float(getattr(algo_cfg, "w_soc", 1.0))
    w_freq    = float(getattr(algo_cfg, "w_freq", 1.5))
    w_voltage = float(getattr(algo_cfg, "w_voltage", 1.0))
    w_shield  = float(getattr(algo_cfg, "w_shield", 0.5))

    log.info(f"Shield-Reward-Shaping: scenario={scenario}, seed={seed}")
    log.info(f"  Weights: thermal={w_thermal}, soc={w_soc}, "
             f"freq={w_freq}, voltage={w_voltage}, shield={w_shield}")

    def make_env():
        env = C2GFastEnv(scenario=scenario)
        shield = SafetyShield()
        return ShieldRewardShapingWrapper(
            env, shield,
            w_thermal=w_thermal, w_soc=w_soc, w_freq=w_freq,
            w_voltage=w_voltage, w_shield=w_shield)

    n_envs = int(getattr(algo_cfg, "n_envs", 4))
    vec_env = make_vec_env(make_env, n_envs=n_envs, seed=seed)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True,
                           clip_obs=10.0, clip_reward=10.0)

    eval_env = make_vec_env(lambda: C2GFastEnv(scenario=scenario),
                            n_envs=1, seed=seed + 999)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False,
                            training=False)

    timesteps = int(getattr(algo_cfg, "timesteps", 300_000))

    model = PPO(
        "MlpPolicy", vec_env,
        learning_rate=float(getattr(algo_cfg, "learning_rate", 3e-4)),
        n_steps=int(getattr(algo_cfg, "n_steps", 512)),
        batch_size=int(getattr(algo_cfg, "batch_size", 128)),
        n_epochs=int(getattr(algo_cfg, "n_epochs", 10)),
        gamma=float(getattr(algo_cfg, "gamma", 0.99)),
        gae_lambda=float(getattr(algo_cfg, "gae_lambda", 0.95)),
        clip_range=float(getattr(algo_cfg, "clip_range", 0.2)),
        ent_coef=float(getattr(algo_cfg, "ent_coef", 0.005)),
        vf_coef=float(getattr(algo_cfg, "vf_coef", 0.5)),
        max_grad_norm=float(getattr(algo_cfg, "max_grad_norm", 0.5)),
        seed=seed, verbose=1)

    callbacks = [
        CheckpointCallback(save_freq=10_000,
                           save_path=str(out_dir / "checkpoints")),
        EvalCallback(eval_env, eval_freq=int(getattr(algo_cfg, "eval_freq", 10_000)),
                     n_eval_episodes=int(getattr(algo_cfg, "n_eval_episodes", 5)),
                     best_model_save_path=str(out_dir / "best_model"),
                     deterministic=True),
        C2GMetricsCallback(log_dir=str(out_dir)),
    ]

    model.learn(total_timesteps=timesteps, callback=callbacks)
    model.save(str(out_dir / "final_model"))
    vec_env.save(str(out_dir / "vec_normalize.pkl"))
    log.info("Shield-Reward-Shaping training complete.")


if __name__ == "__main__":
    train()
