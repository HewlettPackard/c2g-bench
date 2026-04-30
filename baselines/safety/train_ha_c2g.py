"""
baselines/safety/train_ha_c2g.py  —  High-Assurance C2G (Neuro-Symbolic 3-Layer)
==========================================================================
Trains PPO with the full HA-C2G neuro-symbolic architecture, adapted from
the SC26 HA-CompOpt paper:

  Layer 1 — Concept Bottleneck: obs(16/17) → concepts(10)
      Differentiable concept encoder with decaying supervision loss.
      Concepts: thermal_margin_A/B, soc_health, freq_stability,
      voltage_margin, cooling_demand_A/B, grid_urgency,
      batch_pressure, bess_headroom.

  Layer 2 — Safe Projection: concept-conditioned gate (ACTIVELY TRAINED)
      Gate target: g* = 1 − α · max(cooling_demand_A, cooling_demand_B)
      The gate learns to attenuate batch throttle when cooling demand is
      high. Unlike the default near-pass-through init, this gate receives
      explicit supervision via an auxiliary MSE loss on gate targets.
      This is the SC26 "gate supervision experiment" integrated into the
      main training loop.

  Layer 3 — Physics Rule Shield (shield-in-the-loop)
      Simplex safety shield applied DURING training. Actions the agent
      proposes that would violate hard constraints are overridden.
      A shield penalty reward (−0.5 per override) incentivises the
      policy to propose safe actions natively.

  Proof Trees — Generated at evaluation for audit.

Training regime:
  - Joint optimizer for concept encoder + safety gate
  - Concept supervision: decaying weight 1.0 → 0.1 over training
  - Gate supervision: fixed weight 0.1
  - Shield-in-the-loop: shield applied during rollouts
  - Shield penalty: −0.5 per override

Usage
-----
  uv run python baselines/safety/train_ha_c2g.py algo=ha_c2g
  uv run python baselines/safety/train_ha_c2g.py algo=ha_c2g scenario=scenario_b
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
import torch.nn as nn
import torch.nn.functional as F

import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback, CheckpointCallback, EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from c2g_env import C2GFastEnv
from baselines.safety.safety_shield import SafetyShield
from baselines.safety.concept_bottleneck import (
    C2GConcepts, C2GConceptEncoder, C2GGatedConceptFeatureExtractor,
)
from baselines.safety.safe_projection import (
    SafeProjectionGate, ConceptAndGateSupervisionCallback as _SupCallback,
    build_gate_targets,
    compute_layer2_action,
)
from baselines.safety.proof_tree import ProofTree
from baselines.metrics_callback import C2GMetricsCallback

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# SHIELD-IN-THE-LOOP WRAPPER WITH PROOF TREES
# ═══════════════════════════════════════════════════════════════════

class HAC2GShieldWrapper(gym.Wrapper):
    """
    Wraps C2GFastEnv with the full HA-C2G 3-layer action pipeline:

      Layer 2 — Safe Projection Gate (concept-conditioned action attenuation)
        a_gated = a_raw ⊙ g(concepts)
        The gate is trained via auxiliary supervision, and here it is
        **applied** to the action, not just used as a feature.

      Layer 3 — Simplex Safety Shield (hard constraint enforcement)
        a_safe = shield(a_gated, obs)
        Actions violating hard constraints are overridden.

    Plus: shield penalty reward and optional proof tree generation.

    This is the "shield-in-the-loop" training paradigm from the SC26
    paper, adapted for C2G-Bench.
    """

    def __init__(
        self,
        env: gym.Env,
        shield: SafetyShield | None = None,
        shield_penalty: float = 0.5,
        generate_proof_trees: bool = False,
        concept_encoder: "C2GConceptEncoder | None" = None,
        safety_gate: "SafeProjectionGate | None" = None,
    ):
        super().__init__(env)
        self.shield = shield or SafetyShield()
        self.shield_penalty = shield_penalty
        self.generate_proof_trees = generate_proof_trees
        self.concept_encoder = concept_encoder
        self.safety_gate = safety_gate
        self._last_obs = None

    def reset(self, **kwargs):
        self.shield.reset()
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        return obs, info

    def _apply_gate(self, action: np.ndarray, obs: np.ndarray) -> np.ndarray:
        """Apply concept-guided Layer 2 blending.

        Device-safe: builds tensors on the same device as the encoder
        parameters (CPU or CUDA), then brings results back to numpy.
        """
        if self.concept_encoder is None or self.safety_gate is None:
            return action
        with torch.no_grad():
            device = next(self.concept_encoder.parameters()).device
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            action_t = torch.as_tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
            concepts = self.concept_encoder(obs_t)
            gated_action, gate, action_priors = compute_layer2_action(
                action_t,
                concepts,
                obs=obs_t,
                safety_gate=self.safety_gate,
            )
            self._last_gate = gate.squeeze(0).detach().cpu().numpy()
            self._last_action_prior = action_priors.squeeze(0).detach().cpu().numpy()
        return gated_action.squeeze(0).detach().cpu().numpy()

    def step(self, action):
        obs_prev = self._last_obs if self._last_obs is not None else np.zeros(18, dtype=np.float32)

        # Layer 2: concept-conditioned gate
        raw_action = action.copy() if hasattr(action, 'copy') else np.array(action)
        gated_action = self._apply_gate(action, obs_prev)

        # Layer 3: safety shield
        safe_action, was_modified, shield_info = self.shield.filter(gated_action, obs_prev)

        obs, reward, terminated, truncated, info = self.env.step(safe_action)
        self._last_obs = obs

        # Shield penalty
        if was_modified and self.shield_penalty > 0:
            reward -= self.shield_penalty

        info.update(shield_info)
        info["shield_stats"] = self.shield.stats.as_dict()
        info["shield_active"] = was_modified
        info["gate_applied"] = self.concept_encoder is not None
        if hasattr(self, "_last_gate"):
            info["layer2_gate"] = self._last_gate.copy()
        if hasattr(self, "_last_action_prior"):
            info["layer2_action_prior"] = self._last_action_prior.copy()

        # Generate proof tree (expensive, only for evaluation)
        if self.generate_proof_trees:
            tree = ProofTree.from_step(
                obs=obs_prev,
                raw_action=raw_action,
                safe_action=safe_action,
                shield_info={"shield_type": "simplex"},
            )
            info["proof_tree"] = tree.to_dict()
            info["proof_tree_depth"] = tree.depth

        return obs, reward, terminated, truncated, info


# ═══════════════════════════════════════════════════════════════════
# SB3 CALLBACK: Concept + Gate Joint Supervision
# ═══════════════════════════════════════════════════════════════════

class ConceptGateSupervisionCallback(BaseCallback):
    """
    SB3 callback that jointly trains:
      1. Concept encoder (obs → concepts, MSE against ground-truth)
      2. Safety gate (concepts → per-action gate, MSE against targets)

    The concept supervision weight decays from 1.0 → 0.1 over training.
    The gate supervision weight is fixed (default 0.1).
    Both are optimised with a single Adam optimiser.
    """

    def __init__(
        self,
        concept_encoder: C2GConceptEncoder,
        safety_gate: SafeProjectionGate,
        total_timesteps: int,
        gate_alpha: float = 0.5,
        gate_beta: float = 0.3,
        gate_loss_weight: float = 0.1,
        concept_initial_weight: float = 1.0,
        concept_decay_to: float = 0.1,
        supervision_freq: int = 2048,
        batch_size: int = 256,
        lr: float = 1e-3,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.concept_encoder = concept_encoder
        self.safety_gate = safety_gate
        self.total_timesteps = total_timesteps
        self.gate_alpha = gate_alpha
        self.gate_beta = gate_beta
        self.gate_loss_weight = gate_loss_weight
        self.concept_initial_weight = concept_initial_weight
        self.concept_decay_to = concept_decay_to
        self.supervision_freq = supervision_freq
        self.batch_size = batch_size

        # Joint optimiser
        self.optimizer = torch.optim.Adam(
            list(concept_encoder.parameters()) +
            list(safety_gate.parameters()),
            lr=lr,
        )

        self.concept_losses: list[float] = []
        self.gate_losses: list[float] = []
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

        # ── Compute ground-truth concept targets ──────────────────
        targets = []
        for i in range(obs_batch.shape[0]):
            c = C2GConcepts.from_obs(obs_batch[i].detach().cpu().numpy())
            targets.append(c.to_vector())
        targets = torch.FloatTensor(np.array(targets)).to(self.model.device)

        # ── Forward pass ──────────────────────────────────────────
        pred_concepts = self.concept_encoder(obs_batch)
        n_targets = min(pred_concepts.shape[1], targets.shape[1])

        # ── Concept alignment loss (decaying) ─────────────────────
        concept_loss = F.mse_loss(
            pred_concepts[:, :n_targets],
            targets[:, :n_targets],
        ) * concept_weight

        # ── Gate supervision loss (fixed weight) ──────────────────
        gate_values = self.safety_gate(pred_concepts.detach())

        # Compute gate targets
        gate_target = build_gate_targets(
            targets,
            gate_alpha=self.gate_alpha,
            gate_beta=self.gate_beta,
        )

        gate_loss = F.mse_loss(gate_values, gate_target) * self.gate_loss_weight

        # ── Joint backward ────────────────────────────────────────
        total_loss = concept_loss + gate_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        self.concept_losses.append(concept_loss.item())
        self.gate_losses.append(gate_loss.item())

        # ── Logging ──────────────────────────────────────────────
        if self.num_timesteps - self._last_log >= 50_000:
            self._last_log = self.num_timesteps
            avg_cl = np.mean(self.concept_losses[-20:])
            avg_gl = np.mean(self.gate_losses[-20:])
            avg_gate = gate_values.mean().item()
            if self.verbose:
                log.info(
                    f"[HA-C2G] step={self.num_timesteps:,} "
                    f"concept_loss={avg_cl:.6f} gate_loss={avg_gl:.6f} "
                    f"avg_gate={avg_gate:.3f} concept_weight={concept_weight:.3f}")
        return True


# ═══════════════════════════════════════════════════════════════════
# ENV FACTORY
# ═══════════════════════════════════════════════════════════════════

def make_ha_env_fn(
    scenario: str,
    seed: int,
    shield_penalty: float = 0.5,
    concept_encoder: "C2GConceptEncoder | None" = None,
    safety_gate: "SafeProjectionGate | None" = None,
):
    def _init():
        base_env = C2GFastEnv(scenario=scenario)
        env = HAC2GShieldWrapper(
            base_env,
            shield=SafetyShield(),
            shield_penalty=shield_penalty,
            generate_proof_trees=False,  # off during training
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

    scenario  = cfg.scenario.env_id
    seed      = cfg.experiment.seed
    algo_cfg  = cfg.algo
    log_cfg   = cfg.logging

    n_concepts   = int(getattr(algo_cfg, "n_concepts", 10))
    gate_alpha   = float(getattr(algo_cfg, "gate_alpha", 0.5))
    gate_beta    = float(getattr(algo_cfg, "gate_beta", 0.3))
    gate_weight  = float(getattr(algo_cfg, "gate_loss_weight", 0.1))
    shield_pen   = float(getattr(algo_cfg, "shield_penalty", 0.5))

    print(f"[HA-C2G] scenario={scenario}  seed={seed}  "
          f"concepts={n_concepts}  gate_α={gate_alpha}  "
          f"shield_penalty={shield_pen}  timesteps={algo_cfg.timesteps:,}")

    # ── Create shared concept encoder & gate FIRST ───────────────
    # These are shared between the feature extractor and the env
    # wrapper, so the gate is both visible to the policy (as a feature)
    # AND applied to actions (in the wrapper).
    obs_dim = 18  # C2GFastEnv obs dim
    shared_concept_encoder = C2GConceptEncoder(
        obs_dim=obs_dim, n_concepts=n_concepts, hidden=64)
    shared_safety_gate = SafeProjectionGate(
        concept_dim=n_concepts, action_dim=4)

    # ── Environments (with gate applied to actions) ──────────────
    vec_env = make_vec_env(
        make_ha_env_fn(scenario, seed, shield_pen,
                       concept_encoder=shared_concept_encoder,
                       safety_gate=shared_safety_gate),
        n_envs=algo_cfg.n_envs, seed=seed)
    vec_env = VecNormalize(
        vec_env,
        norm_obs=algo_cfg.norm_obs, norm_reward=algo_cfg.norm_reward,
        clip_obs=algo_cfg.clip_obs, clip_reward=algo_cfg.clip_reward)

    eval_env = make_vec_env(
        make_ha_env_fn(scenario, seed + 999, shield_penalty=0.0,
                       concept_encoder=shared_concept_encoder,
                       safety_gate=shared_safety_gate),
        n_envs=1, seed=seed + 999)
    eval_env = VecNormalize(
        eval_env, norm_obs=True, norm_reward=False,
        clip_obs=algo_cfg.clip_obs, training=False)

    # ── Model with Gated Concept Feature Extractor ───────────────
    #    The feature extractor uses the SAME encoder & gate objects,
    #    so gradients from the auxiliary loss update them in-place and
    #    the env wrapper sees the updated parameters.
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

    # ── Verify shared encoder & gate are in the model ──────────────
    fe = model.policy.features_extractor
    assert fe.concept_encoder is shared_concept_encoder, \
        "Feature extractor must share the same concept encoder"
    assert fe.safety_gate is shared_safety_gate, \
        "Feature extractor must share the same safety gate"
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

    # ── Report concept & gate quality ────────────────────────────
    if concept_gate_cb.concept_losses:
        final_cl = np.mean(concept_gate_cb.concept_losses[-10:])
        final_gl = np.mean(concept_gate_cb.gate_losses[-10:])
        print(f"\n[HA-C2G] Final concept_loss={final_cl:.6f}, "
              f"gate_loss={final_gl:.6f}")

    print(f"\n[HA-C2G] Training complete → {out_dir.resolve()}")


if __name__ == "__main__":
    train()
