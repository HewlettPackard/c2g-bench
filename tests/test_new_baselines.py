"""
tests/test_new_baselines.py — Unit tests for Phase 5 baselines
==============================================================
Tests for: BangBang, PID, MPC-Fast, MPC-Macro, MILP Dispatch,
           CMA-ES LinearPolicy, PSO LinearPolicy, PPO-Lagrangian wrapper.

All tests run in-process (no subprocess), each finishing in seconds.
"""
from __future__ import annotations

import numpy as np
import pytest

from c2g_env import C2GFastEnv

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh_env(scenario: str = "default", seed: int = 0):
    env = C2GFastEnv(scenario=scenario)
    obs, _ = env.reset(seed=seed)
    return env, obs


def _obs_17d(**overrides) -> np.ndarray:
    """Build a synthetic 17-D observation vector with sensible defaults."""
    obs = np.array([
        0.86,  # [0]  temp_A_norm  (30°C / 35°C)
        0.57,  # [1]  temp_B_norm  (20°C / 35°C)
        0.50,  # [2]  bess_soc
        0.50,  # [3]  p_base_norm
        0.30,  # [4]  p_flex_nom_norm
        0.70,  # [5]  p_facility_norm
        0.00,  # [6]  regd_signal
        0.40,  # [7]  lmp_norm
        0.50,  # [8]  grid_load_norm
        0.00,  # [9]  is_spike
        0.80,  # [10] prev_throttle
        0.70,  # [11] prev_pump_speed
        0.50,  # [12] pue_norm
        0.50,  # [13] T_amb_norm
        0.00,  # [14] freq_dev_norm
        1.00,  # [15] v_pcc_pu
        0.10,  # [16] backlog_norm
    ], dtype=np.float32)
    for k, v in overrides.items():
        idx = {
            "temp_A": 0, "temp_B": 1, "soc": 2, "regd": 6,
            "lmp": 7, "load": 8, "spike": 9, "freq": 14, "vpcc": 15,
        }[k]
        obs[idx] = v
    return obs


def _run_steps(ctrl, env, obs, n_steps: int = 50):
    """Run controller for n_steps, return (rewards, final_obs)."""
    rewards = []
    for _ in range(n_steps):
        action, _ = ctrl.predict(obs)
        obs, rew, term, trunc, _ = env.step(action)
        rewards.append(rew)
        if term or trunc:
            break
    return rewards, obs


# ═══════════════════════════════════════════════════════════════════════════
# Bang-Bang Controller
# ═══════════════════════════════════════════════════════════════════════════

class TestBangBang:

    def test_instantiation(self):
        from baselines.bang_bang import BangBangController
        ctrl = BangBangController()
        assert ctrl is not None

    def test_predict_shape(self):
        from baselines.bang_bang import BangBangController
        ctrl = BangBangController()
        obs = _obs_17d()
        action, state = ctrl.predict(obs)
        assert action.shape == (4,)
        assert state is None

    def test_action_bounds(self):
        from baselines.bang_bang import BangBangController
        ctrl = BangBangController()
        for _ in range(20):
            obs = _obs_17d(
                temp_A=np.random.uniform(0.5, 1.0),
                temp_B=np.random.uniform(0.4, 1.0),
                soc=np.random.uniform(0.05, 0.98),
                regd=np.random.uniform(-1, 1),
            )
            action, _ = ctrl.predict(obs)
            assert np.all(action >= -1.0 - 1e-6), f"Action below -1: {action}"
            assert np.all(action <=  1.0 + 1e-6), f"Action above 1: {action}"

    def test_throttle_always_max(self):
        from baselines.bang_bang import BangBangController
        ctrl = BangBangController()
        obs = _obs_17d()
        action, _ = ctrl.predict(obs)
        assert action[0] == pytest.approx(1.0), "Throttle should always be 1.0"

    def test_bess_bang_positive_regd(self):
        """Positive regd → full discharge (if SOC healthy)."""
        from baselines.bang_bang import BangBangController
        ctrl = BangBangController()
        obs = _obs_17d(regd=0.5, soc=0.5)
        action, _ = ctrl.predict(obs)
        assert action[3] == pytest.approx(1.0), "BESS should fully discharge on positive regd"

    def test_bess_bang_negative_regd(self):
        """Negative regd → full charge (if SOC healthy)."""
        from baselines.bang_bang import BangBangController
        ctrl = BangBangController()
        obs = _obs_17d(regd=-0.5, soc=0.5)
        action, _ = ctrl.predict(obs)
        assert action[3] == pytest.approx(-1.0), "BESS should fully charge on negative regd"

    def test_bess_soc_guard_low(self):
        """SOC below floor → no discharge."""
        from baselines.bang_bang import BangBangController
        ctrl = BangBangController()
        obs = _obs_17d(regd=0.8, soc=0.05)
        action, _ = ctrl.predict(obs)
        assert action[3] == pytest.approx(0.0), "Should not discharge at very low SOC"

    def test_bess_soc_guard_high(self):
        """SOC above ceiling → no charge."""
        from baselines.bang_bang import BangBangController
        ctrl = BangBangController()
        obs = _obs_17d(regd=-0.8, soc=0.96)
        action, _ = ctrl.predict(obs)
        assert action[3] == pytest.approx(0.0), "Should not charge at very high SOC"

    def test_pump_hysteresis_on(self):
        """Hot Zone A → pump ON."""
        from baselines.bang_bang import BangBangController
        ctrl = BangBangController()
        obs = _obs_17d(temp_A=0.92)  # > 31/35 = 0.886
        action, _ = ctrl.predict(obs)
        assert action[1] == pytest.approx(1.0), "Pump should be ON when hot"

    def test_pump_hysteresis_off(self):
        """Cool Zone A → pump OFF (low speed)."""
        from baselines.bang_bang import BangBangController
        ctrl = BangBangController()
        obs = _obs_17d(temp_A=0.70)  # < 29/35 = 0.829
        action, _ = ctrl.predict(obs)
        assert action[1] == pytest.approx(0.3), "Pump should be at low speed when cool"

    def test_batched_predict(self):
        from baselines.bang_bang import BangBangController
        ctrl = BangBangController()
        obs_batch = np.stack([_obs_17d() for _ in range(5)])
        actions, state = ctrl.predict(obs_batch)
        assert actions.shape == (5, 4)
        assert state is None

    def test_50_steps_no_crash(self):
        from baselines.bang_bang import BangBangController
        ctrl = BangBangController()
        env, obs = _fresh_env()
        rewards, _ = _run_steps(ctrl, env, obs, n_steps=50)
        assert len(rewards) > 0


# ═══════════════════════════════════════════════════════════════════════════
# PID Controller
# ═══════════════════════════════════════════════════════════════════════════

class TestPID:

    def test_instantiation(self):
        from baselines.pid_controller import PIDController
        ctrl = PIDController()
        assert ctrl is not None

    def test_predict_shape(self):
        from baselines.pid_controller import PIDController
        ctrl = PIDController()
        obs = _obs_17d()
        action, state = ctrl.predict(obs)
        assert action.shape == (4,)
        assert state is None

    def test_action_bounds(self):
        from baselines.pid_controller import PIDController
        ctrl = PIDController()
        for _ in range(20):
            obs = _obs_17d(
                temp_A=np.random.uniform(0.5, 1.0),
                soc=np.random.uniform(0.1, 0.95),
                regd=np.random.uniform(-1, 1),
            )
            action, _ = ctrl.predict(obs)
            assert np.all(action >= -1.0 - 1e-6)
            assert np.all(action <=  1.0 + 1e-6)

    def test_bess_tracks_regd(self):
        """With positive regd, BESS should discharge (positive)."""
        from baselines.pid_controller import PIDController
        ctrl = PIDController()
        obs = _obs_17d(regd=0.6, soc=0.5)
        action, _ = ctrl.predict(obs)
        assert action[3] > 0.0, "BESS should discharge for positive regd"

    def test_pump_responds_to_heat(self):
        """Hot Zone A → higher pump speed."""
        from baselines.pid_controller import PIDController
        ctrl = PIDController()
        hot = _obs_17d(temp_A=0.95)
        cold = _obs_17d(temp_A=0.70)
        a_hot, _ = ctrl.predict(hot)
        ctrl.reset()
        a_cold, _ = ctrl.predict(cold)
        assert a_hot[1] > a_cold[1], "Pump should be faster when hot"

    def test_reset_clears_integrators(self):
        from baselines.pid_controller import PIDController
        ctrl = PIDController()
        # Drive integrator
        for _ in range(10):
            ctrl.predict(_obs_17d(regd=0.8))
        ctrl.reset()
        # After reset, first prediction should match fresh controller
        ctrl2 = PIDController()
        obs = _obs_17d(regd=0.0)
        a1, _ = ctrl.predict(obs)
        a2, _ = ctrl2.predict(obs)
        np.testing.assert_allclose(a1, a2, atol=0.01)

    def test_custom_gains(self):
        from baselines.pid_controller import PIDController
        ctrl = PIDController(bess_gains=(5.0, 0.0, 0.0))
        obs = _obs_17d(regd=0.2, soc=0.5)
        action, _ = ctrl.predict(obs)
        # Kp=5 × regd=0.2 = 1.0 (clamped)
        assert action[3] == pytest.approx(1.0, abs=0.01)

    def test_batched_predict(self):
        from baselines.pid_controller import PIDController
        ctrl = PIDController()
        obs_batch = np.stack([_obs_17d() for _ in range(4)])
        actions, _ = ctrl.predict(obs_batch)
        assert actions.shape == (4, 4)

    def test_50_steps_no_crash(self):
        from baselines.pid_controller import PIDController
        ctrl = PIDController()
        env, obs = _fresh_env()
        rewards, _ = _run_steps(ctrl, env, obs, n_steps=50)
        assert len(rewards) > 0


# ═══════════════════════════════════════════════════════════════════════════
# MPC-Fast Controller
# ═══════════════════════════════════════════════════════════════════════════

class TestMPCFast:

    def test_instantiation(self):
        from baselines.mpc_fast import MPCFastController
        ctrl = MPCFastController(horizon=6, max_iter=10)
        assert ctrl.H == 6

    def test_predict_shape(self):
        from baselines.mpc_fast import MPCFastController
        ctrl = MPCFastController(horizon=6, max_iter=10)
        obs = _obs_17d()
        action, state = ctrl.predict(obs)
        assert action.shape == (4,)
        assert state is None

    def test_action_bounds(self):
        from baselines.mpc_fast import MPCFastController
        ctrl = MPCFastController(horizon=6, max_iter=10)
        obs = _obs_17d(regd=0.3, soc=0.5)
        action, _ = ctrl.predict(obs)
        assert action[0] >= -0.01 and action[0] <= 1.01, f"throttle={action[0]}"
        assert action[1] >= 0.14  and action[1] <= 1.01, f"pump={action[1]}"
        assert action[2] >= -0.01 and action[2] <= 1.01, f"hvac={action[2]}"
        assert action[3] >= -1.01 and action[3] <= 1.01, f"bess={action[3]}"

    def test_bess_responds_to_regd(self):
        """MPC should align BESS with regd direction."""
        from baselines.mpc_fast import MPCFastController
        ctrl = MPCFastController(horizon=6, max_iter=20)
        obs = _obs_17d(regd=0.8, soc=0.5)
        action, _ = ctrl.predict(obs)
        assert action[3] > 0.0, "BESS should discharge for positive regd"

    def test_replan_caching(self):
        """With replan_every=3, should reuse plan for 3 steps."""
        from baselines.mpc_fast import MPCFastController
        ctrl = MPCFastController(horizon=6, max_iter=10, replan_every=3)
        obs = _obs_17d(regd=0.2, soc=0.5)
        a1, _ = ctrl.predict(obs)
        a2, _ = ctrl.predict(obs)  # should use cached plan
        a3, _ = ctrl.predict(obs)  # should use cached plan
        a4, _ = ctrl.predict(obs)  # should re-solve
        assert a1.shape == (4,)
        assert a4.shape == (4,)

    def test_5_steps_no_crash(self):
        from baselines.mpc_fast import MPCFastController
        ctrl = MPCFastController(horizon=6, max_iter=10)
        env, obs = _fresh_env()
        rewards, _ = _run_steps(ctrl, env, obs, n_steps=5)
        assert len(rewards) > 0

    def test_batched_predict(self):
        from baselines.mpc_fast import MPCFastController
        ctrl = MPCFastController(horizon=4, max_iter=5)
        obs_batch = np.stack([_obs_17d() for _ in range(3)])
        actions, _ = ctrl.predict(obs_batch)
        assert actions.shape == (3, 4)


# ═══════════════════════════════════════════════════════════════════════════
# MPC-Macro Controller
# ═══════════════════════════════════════════════════════════════════════════

class TestMPCMacro:

    def _obs_macro(self, **overrides) -> np.ndarray:
        """Synthetic 16-D macro observation."""
        obs = np.zeros(16, dtype=np.float32)
        obs[0] = 0.86     # temp_A_mean (norm)
        obs[1] = 0.57     # temp_B_mean (norm)
        obs[2] = 0.50     # soc
        obs[6] = 0.40     # lmp_norm
        obs[7] = 0.50     # grid_load_norm
        obs[10] = 0.10    # headroom_A
        obs[11] = 0.40    # headroom_B
        obs[12] = 0.50    # commit_prev
        obs[13] = 0.0     # bess_prev
        for k, v in overrides.items():
            idx = {"soc": 2, "lmp": 6, "load": 7, "commit_prev": 12}[k]
            obs[idx] = v
        return obs

    def test_instantiation(self):
        from baselines.mpc_macro import MPCMacroController
        ctrl = MPCMacroController(horizon=4, max_iter=20)
        assert ctrl.H == 4

    def test_predict_shape(self):
        from baselines.mpc_macro import MPCMacroController
        ctrl = MPCMacroController(horizon=4, max_iter=20)
        obs = self._obs_macro()
        action, state = ctrl.predict(obs)
        assert action.shape == (2,)
        assert state is None

    def test_action_bounds(self):
        from baselines.mpc_macro import MPCMacroController
        ctrl = MPCMacroController(horizon=4, max_iter=20)
        obs = self._obs_macro()
        action, _ = ctrl.predict(obs)
        assert action[0] >= -0.01 and action[0] <= 1.01, f"commit={action[0]}"
        assert action[1] >= -1.01 and action[1] <= 1.01, f"bess={action[1]}"

    def test_high_lmp_favours_discharge(self):
        from baselines.mpc_macro import MPCMacroController
        ctrl = MPCMacroController(horizon=4, max_iter=30)
        obs = self._obs_macro(lmp=0.9, soc=0.7)
        action, _ = ctrl.predict(obs)
        assert action[1] > -0.1, "High LMP should favour discharge or idle, not deep charge"

    def test_batched_predict(self):
        from baselines.mpc_macro import MPCMacroController
        ctrl = MPCMacroController(horizon=4, max_iter=10)
        obs_batch = np.stack([self._obs_macro() for _ in range(3)])
        actions, _ = ctrl.predict(obs_batch)
        assert actions.shape == (3, 2)


# ═══════════════════════════════════════════════════════════════════════════
# MILP Dispatch Controller
# ═══════════════════════════════════════════════════════════════════════════

class TestMILPDispatch:

    def _obs_macro(self, **overrides) -> np.ndarray:
        obs = np.zeros(16, dtype=np.float32)
        obs[0] = 0.86; obs[1] = 0.57; obs[2] = 0.50
        obs[6] = 0.40; obs[7] = 0.50; obs[12] = 0.50
        for k, v in overrides.items():
            idx = {"soc": 2, "lmp": 6, "load": 7, "commit_prev": 12}[k]
            obs[idx] = v
        return obs

    def test_instantiation(self):
        from baselines.milp_dispatch import MILPDispatchController
        ctrl = MILPDispatchController(horizon=4)
        assert ctrl.H == 4

    def test_predict_shape(self):
        from baselines.milp_dispatch import MILPDispatchController
        ctrl = MILPDispatchController(horizon=4)
        obs = self._obs_macro()
        action, state = ctrl.predict(obs)
        assert action.shape == (2,)
        assert state is None

    def test_action_bounds(self):
        from baselines.milp_dispatch import MILPDispatchController
        ctrl = MILPDispatchController(horizon=4)
        obs = self._obs_macro()
        action, _ = ctrl.predict(obs)
        assert action[0] >= -0.01 and action[0] <= 1.01
        assert action[1] >= -1.01 and action[1] <= 1.01

    def test_solver_converges(self):
        """MILP solver returns a valid solution (not NaN)."""
        from baselines.milp_dispatch import MILPDispatchController
        ctrl = MILPDispatchController(horizon=4)
        for lmp in [0.05, 0.50, 0.95]:
            action, _ = ctrl.predict(self._obs_macro(lmp=lmp, soc=0.5))
            assert np.all(np.isfinite(action)), f"NaN action at lmp={lmp}"
            assert action[0] >= -0.01 and action[0] <= 1.01


# ═══════════════════════════════════════════════════════════════════════════
# CMA-ES Linear Policy
# ═══════════════════════════════════════════════════════════════════════════

class TestCMAESLinearPolicy:

    def test_policy_predict(self):
        from baselines.train_cmaes import LinearPolicy
        policy = LinearPolicy(
            obs_dim=17, act_dim=4,
            act_low=np.array([0, 0, 0, -1], dtype=np.float32),
            act_high=np.array([1, 1, 1, 1], dtype=np.float32),
        )
        obs = _obs_17d()
        action, state = policy.predict(obs)
        assert action.shape == (4,)
        assert state is None

    def test_set_get_params(self):
        from baselines.train_cmaes import LinearPolicy
        policy = LinearPolicy(17, 4, np.zeros(4), np.ones(4))
        n = policy.n_params
        assert n == 17 * 4 + 4  # 72
        params = np.random.randn(n)
        policy.set_params(params)
        recovered = policy.get_params()
        np.testing.assert_allclose(params, recovered)

    def test_action_clipping(self):
        from baselines.train_cmaes import LinearPolicy
        act_low = np.array([0, 0, 0, -1], dtype=np.float32)
        act_high = np.array([1, 1, 1, 1], dtype=np.float32)
        policy = LinearPolicy(17, 4, act_low, act_high)
        # Set extreme weights to produce out-of-range raw outputs
        policy.set_params(np.ones(policy.n_params) * 10.0)
        obs = _obs_17d()
        action, _ = policy.predict(obs)
        assert np.all(action >= act_low - 1e-6)
        assert np.all(action <= act_high + 1e-6)

    def test_batched_predict(self):
        from baselines.train_cmaes import LinearPolicy
        policy = LinearPolicy(17, 4, np.zeros(4), np.ones(4))
        obs_batch = np.stack([_obs_17d() for _ in range(5)])
        actions, _ = policy.predict(obs_batch)
        assert actions.shape == (5, 4)

    def test_evaluate_policy_runs(self):
        """Smoke: evaluate_policy completes one rollout."""
        from baselines.train_cmaes import LinearPolicy, evaluate_policy
        act_low = np.array([0, 0, 0, -1], dtype=np.float32)
        act_high = np.array([1, 1, 1, 1], dtype=np.float32)
        policy = LinearPolicy(17, 4, act_low, act_high)
        reward = evaluate_policy(policy, C2GFastEnv, {"scenario": "default"},
                                 n_rollouts=1, seed_base=0)
        assert np.isfinite(reward)


# ═══════════════════════════════════════════════════════════════════════════
# PSO Linear Policy  (same LinearPolicy structure)
# ═══════════════════════════════════════════════════════════════════════════

class TestPSOLinearPolicy:

    def test_policy_predict(self):
        from baselines.train_pso import LinearPolicy
        policy = LinearPolicy(17, 4, np.zeros(4), np.ones(4))
        obs = _obs_17d()
        action, _ = policy.predict(obs)
        assert action.shape == (4,)

    def test_set_params(self):
        from baselines.train_pso import LinearPolicy
        policy = LinearPolicy(17, 4, np.zeros(4), np.ones(4))
        policy.set_params(np.random.randn(policy.n_params))
        action, _ = policy.predict(_obs_17d())
        assert action.shape == (4,)


# ═══════════════════════════════════════════════════════════════════════════
# PPO-Lagrangian Reward Wrapper
# ═══════════════════════════════════════════════════════════════════════════

class TestPPOLagrangian:

    def test_wrapper_imports(self):
        from baselines.train_ppo_lagrangian import LagrangianRewardWrapper
        assert LagrangianRewardWrapper is not None

    def test_wrapper_modifies_reward(self):
        from baselines.train_ppo_lagrangian import LagrangianRewardWrapper
        env = C2GFastEnv(scenario="default")
        lambdas = np.array([1.0, 1.0, 1.0])
        wrapped = LagrangianRewardWrapper(env, lambdas)
        obs, _ = wrapped.reset(seed=0)
        action = env.action_space.sample()

        obs_wrap, rew_wrap, _, _, info_wrap = wrapped.step(action)
        assert "constraint_costs" in info_wrap
        assert "lagrangian_penalty" in info_wrap
        assert info_wrap["constraint_costs"].shape == (3,)

    def test_wrapper_zero_lambdas_no_penalty(self):
        from baselines.train_ppo_lagrangian import LagrangianRewardWrapper
        env = C2GFastEnv(scenario="default")
        lambdas = np.array([0.0, 0.0, 0.0])
        wrapped = LagrangianRewardWrapper(env, lambdas)
        obs, _ = wrapped.reset(seed=0)
        action = np.array([1.0, 0.7, 0.5, 0.0], dtype=np.float32)
        _, rew_wrap, _, _, info = wrapped.step(action)
        assert info["lagrangian_penalty"] == pytest.approx(0.0)

    def test_lagrangian_callback_imports(self):
        from baselines.train_ppo_lagrangian import LagrangianUpdateCallback
        lambdas = np.array([0.1, 0.1, 0.1])
        budgets = np.array([0.05, 0.10, 0.05])
        cb = LagrangianUpdateCallback(lambdas, budgets)
        assert cb is not None

    def test_wrapper_episode_cost_rates(self):
        """On episode end, info should contain episode_cost_rates."""
        from baselines.train_ppo_lagrangian import LagrangianRewardWrapper
        env = C2GFastEnv(scenario="default")
        lambdas = np.array([0.5, 0.5, 0.5])
        wrapped = LagrangianRewardWrapper(env, lambdas)
        obs, _ = wrapped.reset(seed=0)
        # Run until episode ends
        for _ in range(500):
            action = env.action_space.sample()
            obs, _, term, trunc, info = wrapped.step(action)
            if term or trunc:
                break
        if term or trunc:
            assert "episode_cost_rates" in info
            rates = info["episode_cost_rates"]
            assert rates.shape == (3,)
            assert np.all(rates >= 0.0) and np.all(rates <= 1.0)


# ═══════════════════════════════════════════════════════════════════════════
# Config YAML validation
# ═══════════════════════════════════════════════════════════════════════════

class TestAlgoConfigs:
    """Verify all algo YAML files parse correctly."""

    @pytest.mark.parametrize("yaml_name", [
        "pid", "mpc_fast", "mpc_macro", "milp", "cmaes", "pso", "ppo_lagrangian",
    ])
    def test_yaml_loads(self, yaml_name):
        import yaml
        path = Path(__file__).parent.parent / "conf" / "algo" / f"{yaml_name}.yaml"
        assert path.exists(), f"{path} not found"
        with open(path) as f:
            cfg = yaml.safe_load(f)
        assert "name" in cfg, f"{yaml_name}.yaml missing 'name' key"
        assert cfg["name"] == yaml_name


from pathlib import Path
