"""
baselines/train_cbm_shield.py  —  CBM+Shield Ablation (Tier 3 Ablation)
=========================================================================
PPO with Concept Bottleneck feature extractor AND the Simplex physics
shield (shield-in-the-loop with reward penalty), but NO trained safety
gate.  The concept encoder is trained with decaying supervision loss.

This ablation answers: "Does the trained gate add value when the hard
shield is already present?"

Comparison ladder:
  cbm_only    → CBM, no gate, no shield
  cbm_gate    → CBM + gate, no shield
  cbm_shield  → CBM + shield, no gate   ← THIS
  ha_c2g      → CBM + gate + shield

Usage
-----
  python baselines/train_cbm_shield.py algo=cbm_shield
  python baselines/train_cbm_shield.py algo=cbm_shield scenario=scenario_b
"""
from __future__ import annotations

import logging
from pathlib import Path

import baselines._hydra_compat  # noqa: F401

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback, EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from c2g_env import C2GFastEnv
from baselines.safety.concept_bottleneck import (
    C2GConceptFeatureExtractor,
)
from baselines.train_ha_c2g import HAC2GShieldWrapper
from baselines.train_cbm_only import ConceptSupervisionCallback
from baselines.metrics_callback import C2GMetricsCallback

log = logging.getLogger(__name__)


def make_cbm_shield_env_fn(scenario: str, seed: int, shield_penalty: float = 0.5):
    def _init():
        from baselines.safety_shield import SafetyShield
        base_env = C2GFastEnv(scenario=scenario)
        env = HAC2GShieldWrapper(
            base_env,
            shield=SafetyShield(),
            shield_penalty=shield_penalty,
            generate_proof_trees=False,
        )
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

    n_concepts = int(getattr(algo_cfg, "n_concepts", 10))
    shield_pen = float(getattr(algo_cfg, "shield_penalty", 0.5))

    print(f"[CBM+Shield] scenario={scenario}  seed={seed}  "
          f"concepts={n_concepts}  shield_penalty={shield_pen}  "
          f"timesteps={algo_cfg.timesteps:,}")

    # ── Environments (WITH shield, NO gate) ──────────────────────
    vec_env = make_vec_env(
        make_cbm_shield_env_fn(scenario, seed, shield_pen),
        n_envs=algo_cfg.n_envs, seed=seed)
    vec_env = VecNormalize(
        vec_env,
        norm_obs=algo_cfg.norm_obs, norm_reward=algo_cfg.norm_reward,
        clip_obs=algo_cfg.clip_obs, clip_reward=algo_cfg.clip_reward)

    eval_env = make_vec_env(
        make_cbm_shield_env_fn(scenario, seed + 999, shield_penalty=0.0),
        n_envs=1, seed=seed + 999)
    eval_env = VecNormalize(
        eval_env, norm_obs=True, norm_reward=False,
        clip_obs=algo_cfg.clip_obs, training=False)

    # ── Model with Concept Feature Extractor (no gate) ───────────
    net_arch = OmegaConf.to_container(algo_cfg.net_arch, resolve=True)

    policy_kwargs = dict(
        features_extractor_class=C2GConceptFeatureExtractor,
        features_extractor_kwargs=dict(
            n_concepts=n_concepts,
            hidden=64,
        ),
        net_arch=net_arch,
    )

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
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
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(out_dir / "tensorboard") if log_cfg.tensorboard else None,
        verbose=0,
        seed=seed,
    )

    concept_encoder = model.policy.features_extractor.concept_encoder

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

    concept_cb = ConceptSupervisionCallback(
        concept_encoder=concept_encoder,
        total_timesteps=algo_cfg.timesteps,
        supervision_freq=int(getattr(algo_cfg, "supervision_freq", 2048)),
        batch_size=256,
        lr=1e-3,
        verbose=1,
    )

    # ── Train ────────────────────────────────────────────────────
    model.learn(
        total_timesteps=algo_cfg.timesteps,
        callback=[checkpoint_cb, eval_cb, metrics_cb, concept_cb],
        tb_log_name=cfg.experiment.name,
        reset_num_timesteps=True,
    )

    # ── Save ─────────────────────────────────────────────────────
    model.save(str(out_dir / "final_model"))
    vec_env.save(str(out_dir / "vec_normalize.pkl"))
    torch.save(concept_encoder.state_dict(),
               str(out_dir / "concept_encoder.pt"))

    if concept_cb.concept_losses:
        final_cl = np.mean(concept_cb.concept_losses[-10:])
        print(f"\n[CBM+Shield] Final concept_loss={final_cl:.6f}")

    print(f"\n[CBM+Shield] Training complete → {out_dir.resolve()}")


if __name__ == "__main__":
    train()
