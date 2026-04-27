"""
baselines/train_cbm_gate.py  —  CBM+Gate Ablation (Tier 3 Ablation)
=====================================================================
PPO with Concept Bottleneck + actively trained Safe Projection Gate,
but NO physics shield.  The concept encoder and gate are jointly
trained with supervision losses, and the gate attenuates actions
based on safety concepts.  However, there is no hard guarantee —
the gate is a soft learned filter.

This ablation answers: "Does the trained gate reduce violations
without a hard shield?"

Usage
-----
  python baselines/train_cbm_gate.py algo=cbm_gate
  python baselines/train_cbm_gate.py algo=cbm_gate scenario=scenario_b
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
    C2GConceptEncoder,
    C2GGatedConceptFeatureExtractor,
)
from baselines.safety.safe_projection import SafeProjectionGate

# Re-use the concept+gate supervision callback from HA-C2G
from baselines.train_ha_c2g import ConceptGateSupervisionCallback, HAC2GShieldWrapper
from baselines.metrics_callback import C2GMetricsCallback

log = logging.getLogger(__name__)


class _PassthroughShieldStats:
    def as_dict(self):
        return {}


class PassthroughShield:
    def __init__(self):
        self.stats = _PassthroughShieldStats()

    def filter(self, action, obs):
        return np.asarray(action, dtype=np.float32), False, {}

    def reset(self):
        pass


def make_gate_env_fn(
    scenario: str,
    seed: int,
    concept_encoder: C2GConceptEncoder,
    safety_gate: SafeProjectionGate,
):
    def _init():
        base_env = C2GFastEnv(scenario=scenario)
        env = HAC2GShieldWrapper(
            base_env,
            shield=PassthroughShield(),
            shield_penalty=0.0,
            generate_proof_trees=False,
            concept_encoder=concept_encoder,
            safety_gate=safety_gate,
        )
        env.reset(seed=seed)
        return env
    return _init


# ═══════════════════════════════════════════════════════════════════
# TRAINING ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def train(cfg: DictConfig) -> None:
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    print(OmegaConf.to_yaml(cfg))

    scenario = cfg.scenario.env_id
    seed     = cfg.experiment.seed
    algo_cfg = cfg.algo
    log_cfg  = cfg.logging

    n_concepts  = int(getattr(algo_cfg, "n_concepts", 10))
    gate_alpha  = float(getattr(algo_cfg, "gate_alpha", 0.5))
    gate_beta   = float(getattr(algo_cfg, "gate_beta", 0.3))
    gate_weight = float(getattr(algo_cfg, "gate_loss_weight", 0.1))

    print(f"[CBM+Gate] scenario={scenario}  seed={seed}  "
          f"concepts={n_concepts}  gate_α={gate_alpha}  "
          f"timesteps={algo_cfg.timesteps:,}")

    # ── Shared concept encoder + gate (for features and action path) ─────
    shared_concept_encoder = C2GConceptEncoder(obs_dim=18, n_concepts=n_concepts, hidden=64)
    shared_safety_gate = SafeProjectionGate(concept_dim=n_concepts, action_dim=4)

    # ── Environments (gate active, no physics shield) ────────────────────
    vec_env = make_vec_env(
        make_gate_env_fn(scenario, seed, shared_concept_encoder, shared_safety_gate),
        n_envs=algo_cfg.n_envs, seed=seed)
    vec_env = VecNormalize(
        vec_env,
        norm_obs=algo_cfg.norm_obs, norm_reward=algo_cfg.norm_reward,
        clip_obs=algo_cfg.clip_obs, clip_reward=algo_cfg.clip_reward)

    eval_env = make_vec_env(
        make_gate_env_fn(scenario, seed + 999, shared_concept_encoder, shared_safety_gate),
        n_envs=1, seed=seed + 999)
    eval_env = VecNormalize(
        eval_env, norm_obs=True, norm_reward=False,
        clip_obs=algo_cfg.clip_obs, training=False)

    # ── Model with Gated Concept Feature Extractor ───────────────
    net_arch = OmegaConf.to_container(algo_cfg.net_arch, resolve=True)

    policy_kwargs = dict(
        features_extractor_class=C2GGatedConceptFeatureExtractor,
        features_extractor_kwargs=dict(
            n_concepts=n_concepts,
            action_dim=4,
            hidden=64,
            concept_encoder=shared_concept_encoder,
            safety_gate=shared_safety_gate,
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

    fe = model.policy.features_extractor
    assert fe.concept_encoder is shared_concept_encoder
    assert fe.safety_gate is shared_safety_gate
    concept_encoder = shared_concept_encoder
    safety_gate = shared_safety_gate

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

    concept_gate_cb = ConceptGateSupervisionCallback(
        concept_encoder=concept_encoder,
        safety_gate=safety_gate,
        total_timesteps=algo_cfg.timesteps,
        gate_alpha=gate_alpha,
        gate_beta=gate_beta,
        gate_loss_weight=gate_weight,
        concept_initial_weight=1.0,
        concept_decay_to=0.1,
        supervision_freq=int(getattr(algo_cfg, "supervision_freq", 2048)),
        batch_size=256,
        lr=1e-3,
        verbose=1,
    )

    # ── Train ────────────────────────────────────────────────────
    model.learn(
        total_timesteps=algo_cfg.timesteps,
        callback=[checkpoint_cb, eval_cb, metrics_cb, concept_gate_cb],
        tb_log_name=cfg.experiment.name,
        reset_num_timesteps=True,
    )

    # ── Save ─────────────────────────────────────────────────────
    model.save(str(out_dir / "final_model"))
    vec_env.save(str(out_dir / "vec_normalize.pkl"))
    torch.save(concept_encoder.state_dict(),
               str(out_dir / "concept_encoder.pt"))
    torch.save(safety_gate.state_dict(),
               str(out_dir / "safety_gate.pt"))

    if concept_gate_cb.concept_losses:
        final_cl = np.mean(concept_gate_cb.concept_losses[-10:])
        final_gl = np.mean(concept_gate_cb.gate_losses[-10:])
        print(f"\n[CBM+Gate] Final concept_loss={final_cl:.6f}, "
              f"gate_loss={final_gl:.6f}")

    print(f"\n[CBM+Gate] Training complete → {out_dir.resolve()}")


if __name__ == "__main__":
    train()
