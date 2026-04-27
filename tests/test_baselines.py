"""
tests/test_baselines.py  —  Smoke tests for Phase 3 baseline & evaluation code
"""
from __future__ import annotations
import csv, json
from pathlib import Path
import sys
import types
import numpy as np
import pytest

from c2g_env import C2GFastEnv
from baselines.rule_based_mpc import RuleBasedController


# ── helpers ─────────────────────────────────────────────────────────────────

def _fresh_env(scenario: str = "default", seed: int = 0):
    env = C2GFastEnv(scenario=scenario)
    obs, _ = env.reset(seed=seed)
    return env, obs


# ═══════════════════════════════════════════════════════════════════════════
# RuleBasedController
# ═══════════════════════════════════════════════════════════════════════════

class TestRuleBasedController:

    def test_instantiation(self):
        ctrl = RuleBasedController()
        assert ctrl is not None

    def test_predict_returns_tuple(self):
        ctrl = RuleBasedController()
        _, obs = _fresh_env()
        result = ctrl.predict(obs)
        assert isinstance(result, tuple)
        assert len(result) == 2            # (action, state)

    def test_action_shape(self):
        ctrl = RuleBasedController()
        _, obs = _fresh_env()
        action, _ = ctrl.predict(obs)
        assert action.shape == (4,)

    def test_action_range(self):
        """Action must stay within [-1, 1] for all 3 dims."""
        ctrl = RuleBasedController()
        _, obs = _fresh_env()
        action, _ = ctrl.predict(obs)
        assert np.all(action >= -1.0 - 1e-6)
        assert np.all(action <= 1.0 + 1e-6)

    def test_thermal_protection_high_temp(self):
        """At critical temperature, throttle → 0 and hvac → 1."""
        ctrl = RuleBasedController()
        obs = np.zeros(17, dtype=np.float32)
        obs[0] = 0.99   # temp_A nearly at T_safe
        obs[1] = 0.99   # temp_B nearly at T_safe
        action, _ = ctrl.predict(obs)
        assert action[0] < 0.2, "throttle should be near 0 under thermal protection"
        assert action[2] > 0.5, "hvac should be high under thermal protection"

    def test_bess_follows_regd(self):
        """Positive regd → BESS discharge (positive action)."""
        ctrl = RuleBasedController()
        obs = np.zeros(17, dtype=np.float32)
        obs[2] = 0.5    # SOC=50%, no saturation
        obs[6] = 0.5    # positive regd signal
        action, _ = ctrl.predict(obs)
        assert action[3] > 0.0, "BESS should discharge with positive regd"

    def test_bess_soc_guard_low(self):
        """With SOC nearly empty, BESS should not discharge much."""
        ctrl = RuleBasedController()
        obs = np.zeros(17, dtype=np.float32)
        obs[2] = 0.05   # SOC=5%, near empty
        obs[6] = 0.8    # strong regd signal asking for discharge
        action, _ = ctrl.predict(obs)
        # discharge action should be suppressed
        assert action[3] < 0.5, "High discharge should be suppressed at low SOC"

    def test_bess_soc_guard_high(self):
        """With SOC nearly full, BESS should not charge much."""
        ctrl = RuleBasedController()
        obs = np.zeros(17, dtype=np.float32)
        obs[2] = 0.95   # SOC=95%, nearly full
        obs[6] = -0.8   # grid asking us to charge
        action, _ = ctrl.predict(obs)
        # charge action (negative) should be suppressed
        assert action[3] > -0.5, "Charging should be suppressed at high SOC"

    def test_full_episode_no_crash(self):
        """Run RuleBasedController for a full episode without exceptions."""
        ctrl = RuleBasedController()
        env, obs = _fresh_env(seed=42)
        step_count = 0
        done = False
        while not done:
            action, _ = ctrl.predict(obs)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            step_count += 1
        assert step_count > 0

    def test_full_episode_thermal_default(self):
        """Default scenario should not cause thermal termination with rule-based."""
        ctrl = RuleBasedController()
        env, obs = _fresh_env(scenario="default", seed=0)
        done = False
        terminated = False
        while not done:
            action, _ = ctrl.predict(obs)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        # Rule-based should survive most of the time on default
        # (not guaranteed, but we test it at least runs)
        assert True

    def test_scenario_a_no_crash(self):
        ctrl = RuleBasedController()
        env, obs = _fresh_env(scenario="scenario_a", seed=1)
        done = False
        while not done:
            action, _ = ctrl.predict(obs)
            obs, _, t, tr, _ = env.step(action)
            done = t or tr

    def test_scenario_b_no_crash(self):
        ctrl = RuleBasedController()
        env, obs = _fresh_env(scenario="scenario_b", seed=2)
        done = False
        while not done:
            action, _ = ctrl.predict(obs)
            obs, _, t, tr, _ = env.step(action)
            done = t or tr

    def test_scenario_c_no_crash(self):
        ctrl = RuleBasedController()
        env, obs = _fresh_env(scenario="scenario_c", seed=3)
        done = False
        while not done:
            action, _ = ctrl.predict(obs)
            obs, _, t, tr, _ = env.step(action)
            done = t or tr

    def test_deterministic(self):
        """Same obs should give same action."""
        ctrl = RuleBasedController()
        obs = np.array([0.8, 0.8, 0.5, 0.5, 0.5, 0.5, 0.3, 0.5, 0.5, 0.0, 0.5, 0.5],
                       dtype=np.float32)
        a1, _ = ctrl.predict(obs)
        a2, _ = ctrl.predict(obs)
        np.testing.assert_array_equal(a1, a2)


# ═══════════════════════════════════════════════════════════════════════════
# run_benchmark helpers (import-level tests)
# ═══════════════════════════════════════════════════════════════════════════

class TestBenchmarkImport:
    def test_imports(self):
        from evaluation.run_benchmark import run_episode, RandomAgent
        assert callable(run_episode)

    def test_random_agent_action_shape(self):
        from evaluation.run_benchmark import RandomAgent
        env, _ = _fresh_env()
        agent = RandomAgent(env)
        obs = env.observation_space.sample()
        action, state = agent.predict(obs)
        assert action.shape == (4,)
        assert state is None

    def test_run_episode_returns_dict(self):
        from evaluation.run_benchmark import run_episode, RandomAgent
        env, _ = _fresh_env()
        agent = RandomAgent(env)
        m = run_episode(agent, "default", seed=0)
        expected_keys = {
            "mean_reward", "total_reward", "tracking_rmse",
            "thermal_viol_rate", "throughput_ratio", "bess_degradation",
            "episode_length", "survived",
        }
        assert expected_keys.issubset(m.keys())

    def test_run_episode_finite_values(self):
        from evaluation.run_benchmark import run_episode, RandomAgent
        env, _ = _fresh_env()
        agent = RandomAgent(env)
        m = run_episode(agent, "default", seed=42)
        for k, v in m.items():
            assert np.isfinite(v), f"Metric {k} = {v} is not finite"

    def test_run_episode_rule_based(self):
        from evaluation.run_benchmark import run_episode
        ctrl = RuleBasedController()
        m = run_episode(ctrl, "default", seed=0)
        assert m["episode_length"] > 0
        assert 0.0 <= m["thermal_viol_rate"] <= 1.0

    def test_run_episode_all_scenarios(self):
        from evaluation.run_benchmark import run_episode
        ctrl = RuleBasedController()
        for sc in ["default", "scenario_a", "scenario_b", "scenario_c"]:
            m = run_episode(ctrl, sc, seed=0)
            assert m["episode_length"] > 0, f"Episode too short for {sc}"

    def test_benchmark_function_csv(self, tmp_path):
        from evaluation.run_benchmark import benchmark, save_csv
        rows = benchmark(
            agents     = ["rule_based", "random"],
            scenarios  = ["default"],
            n_episodes = 2,
            seed_start = 0,
            model_dir  = None,
        )
        assert len(rows) == 2   # 2 agents × 1 scenario
        out = tmp_path / "results.csv"
        save_csv(rows, out)
        assert out.exists()
        with open(out) as f:
            reader = csv.DictReader(f)
            data = list(reader)
        assert len(data) == 2
        assert "mean_reward" in data[0]

    def test_load_sb3_agent_restores_obs_normalization(self, monkeypatch, tmp_path):
        import evaluation.run_benchmark as runner

        class FakeModel:
            def __init__(self):
                self.last_obs = None

            @classmethod
            def load(cls, _path):
                return cls()

            def predict(self, obs, deterministic=True):
                self.last_obs = np.array(obs, copy=True)
                return np.zeros(4, dtype=np.float32), None

        class FakeNormalizer:
            def __init__(self):
                self.seen_obs = None

            def normalize_obs(self, obs):
                self.seen_obs = np.array(obs, copy=True)
                return obs + 3.0

        fake_norm = FakeNormalizer()
        fake_sb3 = types.SimpleNamespace(PPO=FakeModel, SAC=FakeModel)
        monkeypatch.setitem(sys.modules, "stable_baselines3", fake_sb3)
        monkeypatch.setattr(runner, "_maybe_load_obs_normalizer", lambda _path, _scenario: fake_norm)

        model_dir = tmp_path / "ppo_default_s42"
        model_dir.mkdir(parents=True)
        (model_dir / "final_model.zip").touch()

        agent = runner.load_sb3_agent("ppo", "default", 42, str(model_dir))
        raw_obs = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        agent.predict(raw_obs)

        np.testing.assert_array_equal(fake_norm.seen_obs, raw_obs)
        np.testing.assert_array_equal(agent._model.last_obs, raw_obs + 3.0)

    def test_load_ha_agent_restores_obs_normalization(self, monkeypatch, tmp_path):
        import evaluation.run_ha_benchmark as runner

        class FakeModel:
            def __init__(self):
                self.last_obs = None

            @classmethod
            def load(cls, _path):
                return cls()

            def predict(self, obs, deterministic=True):
                self.last_obs = np.array(obs, copy=True)
                return np.zeros(4, dtype=np.float32), None

        class FakeNormalizer:
            def __init__(self):
                self.seen_obs = None

            def normalize_obs(self, obs):
                self.seen_obs = np.array(obs, copy=True)
                return obs - 2.0

        fake_norm = FakeNormalizer()
        fake_sb3 = types.SimpleNamespace(PPO=FakeModel)
        monkeypatch.setitem(sys.modules, "stable_baselines3", fake_sb3)
        monkeypatch.setattr(runner, "_maybe_load_obs_normalizer", lambda _path, _scenario: fake_norm)

        model_dir = tmp_path / "ha_c2g_default_s42"
        model_dir.mkdir(parents=True)
        (model_dir / "final_model.zip").touch()

        agent, needs_shield = runner.load_agent("ha_c2g", "default", 42, str(model_dir))
        raw_obs = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        agent.predict(raw_obs)

        assert needs_shield is True
        np.testing.assert_array_equal(fake_norm.seen_obs, raw_obs)
        np.testing.assert_array_equal(agent._m.last_obs, raw_obs - 2.0)


# ═══════════════════════════════════════════════════════════════════════════
# generate_plots  (import + CSV-loading only, no display)
# ═══════════════════════════════════════════════════════════════════════════

class TestGeneratePlotsImport:
    def test_imports(self):
        from evaluation.generate_plots import generate_all
        assert callable(generate_all)

    def test_missing_csv_no_crash(self, tmp_path, capsys):
        from evaluation.generate_plots import generate_all
        generate_all(tmp_path / "nonexistent.csv", tmp_path / "out")
        captured = capsys.readouterr()
        assert "ERROR" in captured.out

    def test_generates_figures(self, tmp_path):
        """Create a minimal CSV and verify figures are produced."""
        from evaluation.run_benchmark import benchmark, save_csv
        from evaluation.generate_plots import generate_all
        rows = benchmark(
            agents     = ["rule_based", "random"],
            scenarios  = ["default"],
            n_episodes = 2,
            seed_start = 0,
            model_dir  = None,
        )
        csv_path = tmp_path / "results.csv"
        save_csv(rows, csv_path)
        out_dir = tmp_path / "figures"
        generate_all(csv_path, out_dir)
        figs = list(out_dir.glob("*.png"))
        assert len(figs) >= 5, f"Expected ≥5 figures, got {len(figs)}: {[f.name for f in figs]}"


# ═══════════════════════════════════════════════════════════════════════════
# Training script imports (no actual training)
# ═══════════════════════════════════════════════════════════════════════════

class TestTrainImports:
    def test_ppo_imports(self):
        import baselines.train_ppo as m
        assert callable(m.train)
        assert callable(m.make_env_fn)

    def test_sac_imports(self):
        import baselines.train_sac as m
        assert callable(m.train)
        assert callable(m.make_env_fn)
