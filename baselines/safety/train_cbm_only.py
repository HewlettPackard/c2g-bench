"""
baselines/safety/train_cbm_only.py  —  CBM-Only Ablation (Tier 3 Ablation)
=====================================================================
PPO with Concept Bottleneck feature extractor but NO safety gate
and NO physics shield.  The concept encoder is trained with decaying
supervision loss, but the policy receives [obs; concepts] as features
without any action filtering.

This ablation answers: "Does interpretability alone improve safety?"

Usage
-----
  python baselines/safety/train_cbm_only.py algo=cbm_only
  python baselines/safety/train_cbm_only.py algo=cbm_only scenario=scenario_b
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
import torch.nn.functional as F

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback, CheckpointCallback, EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from c2g_env import C2GFastEnv
from baselines.safety.concept_bottleneck import (
    C2GConcepts, C2GConceptEncoder, C2GConceptFeatureExtractor,
)
from baselines.metrics_callback import C2GMetricsCallback

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# SB3 CALLBACK: Concept Supervision Only
# ═══════════════════════════════════════════════════════════════════

class ConceptSupervisionCallback(BaseCallback):
    """
    Trains the concept encoder with decaying MSE supervision loss
    against ground-truth concepts.  No gate, no shield.
    """

    def __init__(
        self,
        concept_encoder: C2GConceptEncoder,
        total_timesteps: int,
        concept_initial_weight: float = 1.0,
        concept_decay_to: float = 0.1,
        supervision_freq: int = 2048,
        batch_size: int = 256,
        lr: float = 1e-3,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.concept_encoder = concept_encoder
        self.total_timesteps = total_timesteps
        self.concept_initial_weight = concept_initial_weight
        self.concept_decay_to = concept_decay_to
        self.supervision_freq = supervision_freq
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(concept_encoder.parameters(), lr=lr)
        self.concept_losses: list[float] = []
        self._last_log = 0

    def _on_step(self) -> bool:
        if self.num_timesteps % self.supervision_freq != 0:
            return True
        if self.num_timesteps == 0:
            return True

        progress = min(1.0, self.num_timesteps / self.total_timesteps)
        concept_weight = (
            self.concept_initial_weight -
            (self.concept_initial_weight - self.concept_decay_to) * progress
        )

        buf = self.model.rollout_buffer
        if buf.pos <= 0:
            return True

        obs = buf.observations[:buf.pos].reshape(-1, buf.observations.shape[-1])
        idx = np.random.choice(len(obs), min(self.batch_size, len(obs)), replace=False)
        obs_batch = torch.FloatTensor(obs[idx]).to(self.model.device)

        targets = []
        for i in range(obs_batch.shape[0]):
            c = C2GConcepts.from_obs(obs_batch[i].detach().cpu().numpy())
            targets.append(c.to_vector())
        targets = torch.FloatTensor(np.array(targets)).to(self.model.device)

        pred_concepts = self.concept_encoder(obs_batch)
        n_targets = min(pred_concepts.shape[1], targets.shape[1])

        concept_loss = F.mse_loss(
            pred_concepts[:, :n_targets], targets[:, :n_targets]
        ) * concept_weight

        self.optimizer.zero_grad()
        concept_loss.backward()
        self.optimizer.step()

        self.concept_losses.append(concept_loss.item())

        if self.num_timesteps - self._last_log >= 50_000:
            self._last_log = self.num_timesteps
            avg_cl = np.mean(self.concept_losses[-20:])
            if self.verbose:
                log.info(
                    f"[CBM-Only] step={self.num_timesteps:,} "
                    f"concept_loss={avg_cl:.6f} "
                    f"concept_weight={concept_weight:.3f}")
        return True


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

    n_concepts = int(getattr(algo_cfg, "n_concepts", 10))

    print(f"[CBM-Only] scenario={scenario}  seed={seed}  "
          f"concepts={n_concepts}  timesteps={algo_cfg.timesteps:,}")

    # ── Environments (NO shield wrapper) ─────────────────────────
    vec_env = make_vec_env(
        lambda: C2GFastEnv(scenario=scenario),
        n_envs=algo_cfg.n_envs, seed=seed)
    vec_env = VecNormalize(
        vec_env,
        norm_obs=algo_cfg.norm_obs, norm_reward=algo_cfg.norm_reward,
        clip_obs=algo_cfg.clip_obs, clip_reward=algo_cfg.clip_reward)

    eval_env = make_vec_env(
        lambda: C2GFastEnv(scenario=scenario),
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
        print(f"\n[CBM-Only] Final concept_loss={final_cl:.6f}")

    print(f"\n[CBM-Only] Training complete → {out_dir.resolve()}")


if __name__ == "__main__":
    train()
