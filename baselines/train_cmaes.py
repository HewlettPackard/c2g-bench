"""
baselines/train_cmaes.py  —  CMA-ES Policy Search
===================================================
Optimises a parameterised linear policy via Covariance Matrix Adaptation
Evolution Strategy (CMA-ES), the gold-standard gradient-free optimizer
for continuous search spaces.

Policy parameterisation:
    action = clip(W @ obs + b, action_low, action_high)
    where W ∈ R^{4×17}, b ∈ R^4  (fast env, 72 parameters)
    or    W ∈ R^{2×17}, b ∈ R^2  (macro env, 36 parameters)

Fitness: mean total episode reward over ``n_rollouts`` episodes.

Requires: ``cma``  (pip install cma).

Usage
-----
  # Train on fast env
  uv run python baselines/train_cmaes.py

  # Train on macro env
  uv run python baselines/train_cmaes.py algo=cmaes scenario=scenario_b

  # Override CMA-ES params
  uv run python baselines/train_cmaes.py algo.popsize=40 algo.generations=300
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import baselines._hydra_compat  # noqa: F401

import hydra
from omegaconf import DictConfig

import numpy as np

log = logging.getLogger(__name__)


class LinearPolicy:
    """Linear obs→action policy with clip."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        act_low: np.ndarray,
        act_high: np.ndarray,
    ) -> None:
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_low = act_low
        self.act_high = act_high
        self.n_params = act_dim * obs_dim + act_dim
        self.W = np.zeros((act_dim, obs_dim))
        self.b = np.zeros(act_dim)

    def set_params(self, params: np.ndarray) -> None:
        n = self.act_dim * self.obs_dim
        self.W = params[:n].reshape(self.act_dim, self.obs_dim)
        self.b = params[n:]

    def get_params(self) -> np.ndarray:
        return np.concatenate([self.W.ravel(), self.b])

    def predict(
        self,
        obs: np.ndarray,
        state=None,
        episode_start=None,
        deterministic: bool = True,
    ) -> tuple[np.ndarray, None]:
        if obs.ndim == 1:
            action = np.clip(self.W @ obs + self.b, self.act_low, self.act_high)
        else:
            action = np.clip(obs @ self.W.T + self.b, self.act_low, self.act_high)
        return action.astype(np.float32), None


def evaluate_policy(
    policy: LinearPolicy,
    env_cls,
    env_kwargs: dict,
    n_rollouts: int = 3,
    seed_base: int = 0,
) -> float:
    """Return mean total reward over n_rollouts episodes."""
    total_rewards = []
    for i in range(n_rollouts):
        env = env_cls(**env_kwargs)
        obs, _ = env.reset(seed=seed_base + i)
        total_reward = 0.0
        done = False
        while not done:
            action, _ = policy.predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        total_rewards.append(total_reward)
        env.close()
    return float(np.mean(total_rewards))


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def train(cfg: DictConfig) -> None:
    import cma

    scenario = cfg.scenario.name
    algo_cfg = cfg.algo
    seed     = cfg.experiment.seed

    # Determine environment
    env_name = getattr(algo_cfg, "env", "fast")
    if env_name == "macro":
        from c2g_env import C2GMacroEnv as env_cls
    else:
        from c2g_env import C2GFastEnv as env_cls

    env_kwargs = {"scenario": scenario}
    tmp_env = env_cls(**env_kwargs)
    obs_dim = tmp_env.observation_space.shape[0]
    act_dim = tmp_env.action_space.shape[0]
    act_low = tmp_env.action_space.low
    act_high = tmp_env.action_space.high
    tmp_env.close()

    policy = LinearPolicy(obs_dim, act_dim, act_low, act_high)

    popsize     = int(getattr(algo_cfg, "popsize", 20))
    generations = int(getattr(algo_cfg, "generations", 200))
    sigma0      = float(getattr(algo_cfg, "sigma0", 0.5))
    n_rollouts  = int(getattr(algo_cfg, "n_rollouts", 3))

    log.info(
        f"CMA-ES: {policy.n_params} params, popsize={popsize}, "
        f"generations={generations}, scenario={scenario}"
    )

    x0 = np.zeros(policy.n_params)
    es = cma.CMAEvolutionStrategy(
        x0, sigma0,
        {"popsize": popsize, "seed": seed, "maxiter": generations, "verbose": -1},
    )

    best_fitness = -np.inf
    best_params = x0.copy()

    gen = 0
    while not es.stop():
        solutions = es.ask()
        fitnesses = []
        for params in solutions:
            policy.set_params(params)
            fit = evaluate_policy(
                policy, env_cls, env_kwargs,
                n_rollouts=n_rollouts, seed_base=seed,
            )
            fitnesses.append(-fit)  # CMA-ES minimises; negate reward

        es.tell(solutions, fitnesses)
        gen += 1

        # Track best
        gen_best_idx = int(np.argmin(fitnesses))
        gen_best_reward = -fitnesses[gen_best_idx]
        if gen_best_reward > best_fitness:
            best_fitness = gen_best_reward
            best_params = solutions[gen_best_idx].copy()

        if gen % 10 == 0:
            log.info(
                f"Gen {gen}/{generations}: best_reward={best_fitness:.1f}, "
                f"gen_best={gen_best_reward:.1f}"
            )

    # Save best policy
    out_dir = Path(".")
    policy.set_params(best_params)
    np.savez(
        out_dir / "cmaes_policy.npz",
        W=policy.W,
        b=policy.b,
        obs_dim=obs_dim,
        act_dim=act_dim,
        act_low=act_low,
        act_high=act_high,
    )

    results = {
        "best_fitness": best_fitness,
        "generations": gen,
        "n_params": policy.n_params,
        "scenario": scenario,
        "seed": seed,
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    log.info(f"CMA-ES done. Best reward: {best_fitness:.1f}")


if __name__ == "__main__":
    train()
