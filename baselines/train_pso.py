"""
baselines/train_pso.py  —  Particle Swarm Optimization Policy Search
=====================================================================
Optimises a parameterised linear policy via PSO, the most popular
swarm intelligence method in the energy systems literature.

Policy parameterisation (identical to CMA-ES):
    action = clip(W @ obs + b, action_low, action_high)

Fitness: mean total episode reward over ``n_rollouts`` episodes.

Requires: ``pymoo``  (pip install pymoo).

Usage
-----
  uv run python baselines/train_pso.py algo=pso
  uv run python baselines/train_pso.py algo=pso scenario=scenario_b
  uv run python baselines/train_pso.py algo.n_particles=40 algo.generations=300
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
    """Linear obs→action policy with clip (shared with CMA-ES)."""

    def __init__(self, obs_dim: int, act_dim: int, act_low, act_high):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_low = np.asarray(act_low)
        self.act_high = np.asarray(act_high)
        self.n_params = act_dim * obs_dim + act_dim
        self.W = np.zeros((act_dim, obs_dim))
        self.b = np.zeros(act_dim)

    def set_params(self, params: np.ndarray) -> None:
        n = self.act_dim * self.obs_dim
        self.W = params[:n].reshape(self.act_dim, self.obs_dim)
        self.b = params[n:]

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        if obs.ndim == 1:
            action = np.clip(self.W @ obs + self.b, self.act_low, self.act_high)
        else:
            action = np.clip(obs @ self.W.T + self.b, self.act_low, self.act_high)
        return action.astype(np.float32), None


def evaluate_policy(policy, env_cls, env_kwargs, n_rollouts=3, seed_base=0):
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
    from pymoo.algorithms.soo.nonconvex.pso import PSO
    from pymoo.core.problem import Problem
    from pymoo.optimize import minimize as pymoo_minimize
    from pymoo.termination import get_termination

    scenario = cfg.scenario.name
    algo_cfg = cfg.algo
    seed     = cfg.experiment.seed

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

    n_particles = int(getattr(algo_cfg, "n_particles", 20))
    generations = int(getattr(algo_cfg, "generations", 200))
    n_rollouts  = int(getattr(algo_cfg, "n_rollouts", 3))

    log.info(
        f"PSO: {policy.n_params} params, n_particles={n_particles}, "
        f"generations={generations}, scenario={scenario}"
    )

    class PolicyOptProblem(Problem):
        def __init__(self):
            super().__init__(
                n_var=policy.n_params,
                n_obj=1,
                xl=-2.0 * np.ones(policy.n_params),
                xu=2.0 * np.ones(policy.n_params),
            )

        def _evaluate(self, X, out, *args, **kwargs):
            F = np.zeros(len(X))
            for i, params in enumerate(X):
                policy.set_params(params)
                reward = evaluate_policy(
                    policy, env_cls, env_kwargs,
                    n_rollouts=n_rollouts, seed_base=seed,
                )
                F[i] = -reward  # pymoo minimises
            out["F"] = F.reshape(-1, 1)

    problem = PolicyOptProblem()
    algorithm = PSO(pop_size=n_particles)
    termination = get_termination("n_gen", generations)

    result = pymoo_minimize(
        problem, algorithm,
        termination=termination,
        seed=seed,
        verbose=True,
    )

    # Save best policy
    best_params = result.X
    best_reward = -float(result.F[0])
    policy.set_params(best_params)

    out_dir = Path(".")
    np.savez(
        out_dir / "pso_policy.npz",
        W=policy.W,
        b=policy.b,
        obs_dim=obs_dim,
        act_dim=act_dim,
        act_low=act_low,
        act_high=act_high,
    )

    results = {
        "best_fitness": best_reward,
        "generations": generations,
        "n_params": policy.n_params,
        "scenario": scenario,
        "seed": seed,
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    log.info(f"PSO done. Best reward: {best_reward:.1f}")


if __name__ == "__main__":
    train()
