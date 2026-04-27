"""
baselines/train_rule_macro_ppo.py  —  PPO Low-Level + Rule-Based Macro
=======================================================================
Trains a PPO low-level controller (4-D: throttle, pump, hvac, bess) on
``C2GFastEnv`` while a frozen ``RuleBasedMacroController`` drives the
15-minute market bidding decisions that determine ``committed_mw``.

This gives the low-level agent a realistic training distribution: the
grid commitment varies over time (as the macro controller bids), so the
agent learns to track *dynamic* regulation signals — not just the fixed
``dr_baseline_mw`` default.

Architecture
------------
  ┌──────────────────────────────────────────────┐
  │  RuleMacroWrappedEnv  (gymnasium.Wrapper)    │
  │                                              │
  │  every 180 steps (15 min):                   │
  │    ① aggregate sub-obs → macro obs (19-D)    │
  │    ② RuleBasedMacroController.predict(obs)   │
  │    ③ grid.step_rmcp() → clear_bid()          │
  │    ④ env.committed_mw = cleared MW           │
  │                                              │
  │  every step (5 s):                           │
  │    PPO action (4-D) → C2GFastEnv.step()      │
  └──────────────────────────────────────────────┘

Usage
-----
  # Single run with defaults
  python baselines/train_rule_macro_ppo.py

  # Override scenario
  python baselines/train_rule_macro_ppo.py scenario=scenario_a

  # Sweep
  python baselines/train_rule_macro_ppo.py --multirun \\
      scenario=default,scenario_a,scenario_b,scenario_c \\
      experiment.seed=1,2,3

Outputs (Hydra-managed)
-----------------------
  outputs/<algo>_<scenario>/seed_<N>/<timestamp>/
      .hydra/               — config snapshot
      episode_metrics.csv
      checkpoints/
      best_model/
      tensorboard/
"""
from __future__ import annotations
from pathlib import Path
from typing import Any

import baselines._hydra_compat  # noqa: F401  # Hydra 1.3.x + Python ≥3.14 fix

import numpy as np
import yaml
import gymnasium as gym
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, sync_envs_normalization

from c2g_env import C2GFastEnv
from c2g_env.env_high_level import C2GMacroEnv
from baselines.rule_based_macro import RuleBasedMacroController
from baselines.metrics_callback import C2GMetricsCallback


# Number of FastEnv steps per macro decision (180 × 5 s = 900 s = 15 min)
_SUBSTEPS = 180

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "c2g_env" / "config.yaml"


class RuleMacroWrappedEnv(gym.Wrapper):
    """
    Wraps ``C2GFastEnv`` so that a rule-based macro controller
    periodically updates ``committed_mw`` via market handshake.

    The low-level action/observation spaces are unchanged (4-D / 18-D).
    Every ``_SUBSTEPS`` steps the wrapper:
      1. Aggregates recent sub-observations into a 19-D macro obs
      2. Calls ``RuleBasedMacroController.predict()``
      3. Runs the 3-phase market handshake (RMCP → bid → clear)
      4. Sets ``env.committed_mw`` to the cleared value

    Reuses ``C2GMacroEnv.aggregate_macro_obs``, ``C2GMacroEnv.run_handshake``,
    and ``C2GMacroEnv.macro_obs_from_reset`` so that observation construction
    and market logic are defined in exactly one place.
    """

    def __init__(self, env: C2GFastEnv) -> None:
        super().__init__(env)

        self._macro_ctrl = RuleBasedMacroController()

        # Load handshake config from the same config file the env uses
        cfg_path = _CONFIG_PATH
        with open(cfg_path, encoding="utf-8") as fh:
            full_cfg = yaml.safe_load(fh)
        hs_cfg = full_cfg.get("handshake", {})
        self._rmcp_max = float(hs_cfg.get("rmcp_max", 100.0))

        scfg = env._scfg
        self._committed_max_mw = float(scfg["committed_mw_max"])
        self._dr_baseline_mw = float(scfg.get("dr_baseline_mw", 5.0))
        self._dr_rate_usd_mw = float(scfg.get("dr_rate_usd_mw", 5.0))

        # Tracking state
        self._sub_tick = 0
        self._obs_buffer: list[np.ndarray] = []
        self._info_buffer: list[dict] = []
        self._prev_bid_mw_norm = 0.5
        self._prev_bid_price_norm = 0.5
        self._last_rmcp_norm = 0.0
        self._last_reg_need_norm = 0.0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict]:
        obs, info = self.env.reset(seed=seed, options=options)
        self._sub_tick = 0
        self._obs_buffer = []
        self._info_buffer = []
        self._prev_bid_mw_norm = 0.5
        self._prev_bid_price_norm = 0.5
        self._last_rmcp_norm = 0.0
        self._last_reg_need_norm = 0.0

        # Perform initial macro decision to set committed_mw from rule-based bid
        self._do_macro_handshake()
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        # At the start of each new macro window (after the first), update commitment
        if self._sub_tick > 0 and self._sub_tick % _SUBSTEPS == 0:
            self._do_macro_handshake()

        obs, reward, terminated, truncated, info = self.env.step(action)
        self._obs_buffer.append(obs)
        self._info_buffer.append(info)
        self._sub_tick += 1
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Market handshake
    # ------------------------------------------------------------------

    def _do_macro_handshake(self) -> None:
        """Run rule-based macro bid + grid clearing to update committed_mw."""
        macro_obs = self._build_macro_obs()
        bid_action, _ = self._macro_ctrl.predict(macro_obs)

        bid_mw_norm = float(bid_action[0])
        bid_price_norm = float(bid_action[1])

        hs = C2GMacroEnv.run_handshake(
            self.env._grid,  # type: ignore[attr-defined]
            bid_mw_norm=bid_mw_norm,
            bid_price_norm=bid_price_norm,
            committed_max_mw=self._committed_max_mw,
            rmcp_max=self._rmcp_max,
            dr_baseline_mw=self._dr_baseline_mw,
            dr_rate_usd_mw=self._dr_rate_usd_mw,
        )

        self.env.committed_mw = hs["committed_mw"]  # type: ignore[attr-defined]
        self._prev_bid_mw_norm = bid_mw_norm
        self._prev_bid_price_norm = bid_price_norm
        self._last_rmcp_norm = hs["rmcp_norm"]
        self._last_reg_need_norm = hs["reg_need_norm"]

        # Reset buffer for next macro window
        self._obs_buffer = []
        self._info_buffer = []

    def _build_macro_obs(self) -> np.ndarray:
        """
        Build a 19-D macro observation from buffered sub-step data.

        Delegates to ``C2GMacroEnv.aggregate_macro_obs`` (with buffer) or
        ``C2GMacroEnv.macro_obs_from_reset`` (at reset, no buffer).
        """
        if not self._obs_buffer:
            # At reset — build from the env's current state
            inner_obs = np.zeros(18, dtype=np.float32)
            inner_obs[7] = 0.5  # grid_load_norm (triggers mid-range bid)
            inner_obs[15] = 1.0  # v_pcc_pu (nominal)
            T_safe = self.env._thermal.T_safe  # type: ignore[attr-defined]
            return C2GMacroEnv.macro_obs_from_reset(
                inner_obs,
                T_safe=T_safe,
                temp_A=self.env._thermal.temp_A,  # type: ignore[attr-defined]
                temp_B=self.env._thermal.temp_B,  # type: ignore[attr-defined]
                prev_bid_mw_norm=self._prev_bid_mw_norm,
                prev_bid_price_norm=self._prev_bid_price_norm,
            )

        # Inject p_flex_max_kw so aggregate_macro_obs can compute backlog_norms
        p_flex_max_kw = self.env._workload.p_flex_max_kw  # type: ignore[attr-defined]
        for info in self._info_buffer:
            info["p_flex_max_kw"] = p_flex_max_kw

        committed_mw = self.env.committed_mw  # type: ignore[attr-defined]
        return C2GMacroEnv.aggregate_macro_obs(
            self._obs_buffer, self._info_buffer,
            T_safe=self.env._thermal.T_safe,  # type: ignore[attr-defined]
            committed_mw=committed_mw,
            bid_mw_norm=self._prev_bid_mw_norm,
            bid_price_norm=self._prev_bid_price_norm,
            rmcp_norm=self._last_rmcp_norm,
            reg_need_norm=self._last_reg_need_norm,
        )


# ======================================================================
# SyncNormEvalCallback (shared with train_ppo.py)
# ======================================================================

class SyncNormEvalCallback(EvalCallback):
    """EvalCallback that syncs VecNormalize stats before evaluation."""

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            sync_envs_normalization(self.training_env, self.eval_env)
        return super()._on_step()


# ======================================================================
# Env factory
# ======================================================================

def make_env_fn(scenario: str, seed: int):
    def _init():
        fast_env = C2GFastEnv(scenario=scenario)
        env = RuleMacroWrappedEnv(fast_env)
        env.reset(seed=seed)
        return env
    return _init


# ======================================================================
# Hydra entry-point
# ======================================================================

@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def train(cfg: DictConfig) -> None:
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    print(OmegaConf.to_yaml(cfg))

    scenario = cfg.scenario.env_id
    seed = cfg.experiment.seed
    algo_cfg = cfg.algo
    log_cfg = cfg.logging

    print(f"[RuleMacro+PPO] scenario={scenario}  seed={seed}  "
          f"timesteps={algo_cfg.timesteps:,}  n_envs={algo_cfg.n_envs}")

    # ── Environments ──────────────────────────────────────────────────
    vec_env = make_vec_env(make_env_fn(scenario, seed),
                           n_envs=algo_cfg.n_envs, seed=seed)
    vec_env = VecNormalize(
        vec_env,
        norm_obs=algo_cfg.norm_obs,
        norm_reward=algo_cfg.norm_reward,
        clip_obs=algo_cfg.clip_obs,
        clip_reward=algo_cfg.clip_reward,
    )

    eval_env = make_vec_env(make_env_fn(scenario, seed + 999),
                            n_envs=1, seed=seed + 999)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False,
                            clip_obs=algo_cfg.clip_obs, training=False)

    # ── Callbacks ─────────────────────────────────────────────────────
    checkpoint_cb = CheckpointCallback(
        save_freq=max(algo_cfg.eval_freq, 1),
        save_path=str(out_dir / "checkpoints"),
        name_prefix="ckpt",
    )
    eval_cb = SyncNormEvalCallback(
        eval_env,
        best_model_save_path=str(out_dir / "best_model"),
        log_path=str(out_dir / "tensorboard"),
        eval_freq=algo_cfg.eval_freq,
        n_eval_episodes=algo_cfg.n_eval_episodes,
        deterministic=True,
        verbose=0,
    )
    metrics_cb = C2GMetricsCallback(
        print_freq=log_cfg.console_freq,
        csv_path=out_dir / "episode_metrics.csv" if log_cfg.csv else None,
        verbose=1,
    )

    # ── Model ─────────────────────────────────────────────────────────
    net_arch = OmegaConf.to_container(algo_cfg.net_arch, resolve=True)

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
        policy_kwargs=dict(net_arch=net_arch),
        tensorboard_log=str(out_dir / "tensorboard") if log_cfg.tensorboard else None,
        verbose=0,
        seed=seed,
        device=cfg.device,
    )

    # ── Train ─────────────────────────────────────────────────────────
    model.learn(
        total_timesteps=algo_cfg.timesteps,
        callback=[checkpoint_cb, eval_cb, metrics_cb],
        tb_log_name=cfg.experiment.name,
        reset_num_timesteps=True,
    )

    model.save(str(out_dir / "final_model"))
    vec_env.save(str(out_dir / "vec_normalize.pkl"))
    print(f"\n[RuleMacro+PPO] Training complete → {out_dir.resolve()}")


if __name__ == "__main__":
    train()
