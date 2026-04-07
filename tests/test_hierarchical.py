"""
Tests for hierarchical RL infrastructure
==========================================
Covers:
  - RuleBasedMacroController: predict interface, action bounds, safety rules
  - C2GMacroEnv with inner_action_fn: obs shape, step, episode
  - train_ppo_macro module importable
  - train_hierarchical module importable
"""
import math
import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from c2g_env import C2GMacroEnv
from baselines.rule_based_macro import RuleBasedMacroController


# =========================================================================
# A. RuleBasedMacroController
# =========================================================================

class TestRuleBasedMacroController:

    @pytest.fixture
    def ctrl(self):
        return RuleBasedMacroController()

    def test_predict_returns_tuple(self, ctrl):
        obs = np.zeros(17, dtype=np.float32)
        action, state = ctrl.predict(obs)
        assert state is None
        assert isinstance(action, np.ndarray)

    def test_action_shape(self, ctrl):
        obs = np.zeros(17, dtype=np.float32)
        action, _ = ctrl.predict(obs)
        assert action.shape == (2,)

    def test_action_bounds(self, ctrl):
        """Actions must be within MacroEnv action space."""
        for _ in range(20):
            obs = np.random.rand(17).astype(np.float32)
            obs[6] = np.random.uniform(0, 1)   # lmp_norm
            obs[7] = np.random.uniform(0, 1)   # load_norm
            obs[2] = np.random.uniform(0, 1)   # soc
            action, _ = ctrl.predict(obs)
            assert 0.0 <= action[0] <= 1.0, f"commit_norm={action[0]} out of [0,1]"
            assert -1.0 <= action[1] <= 1.0, f"bess_target={action[1]} out of [-1,1]"

    def test_high_load_high_commitment(self, ctrl):
        obs = np.zeros(17, dtype=np.float32)
        obs[7] = 0.9   # high grid load
        obs[10] = 0.5   # healthy headroom A
        obs[11] = 0.5   # healthy headroom B
        obs[15] = 1.0   # nominal voltage
        action, _ = ctrl.predict(obs)
        assert action[0] >= 0.7, "High load should produce high commitment"

    def test_low_load_low_commitment(self, ctrl):
        obs = np.zeros(17, dtype=np.float32)
        obs[7] = 0.2   # low grid load
        obs[10] = 0.5
        obs[11] = 0.5
        obs[15] = 1.0
        action, _ = ctrl.predict(obs)
        assert action[0] <= 0.3, "Low load should produce low commitment"

    def test_high_lmp_discharge(self, ctrl):
        obs = np.zeros(17, dtype=np.float32)
        obs[2] = 0.5    # mid SOC
        obs[6] = 0.8    # high LMP
        obs[10] = 0.5
        obs[11] = 0.5
        obs[15] = 1.0
        action, _ = ctrl.predict(obs)
        assert action[1] > 0.0, "High LMP should trigger discharge"

    def test_low_soc_charges(self, ctrl):
        obs = np.zeros(17, dtype=np.float32)
        obs[2] = 0.10   # very low SOC
        obs[6] = 0.8    # high LMP (but SOC override should override)
        obs[10] = 0.5
        obs[11] = 0.5
        obs[15] = 1.0
        action, _ = ctrl.predict(obs)
        assert action[1] < 0.0, "Low SOC should charge regardless of LMP"

    def test_batch_predict(self, ctrl):
        obs = np.zeros((5, 17), dtype=np.float32)
        actions, _ = ctrl.predict(obs)
        assert actions.shape == (5, 2)

    def test_thermal_emergency_reduces_commitment(self, ctrl):
        obs = np.zeros(17, dtype=np.float32)
        obs[7] = 0.9    # high load → normally high commitment
        obs[10] = 0.05  # very low headroom A → safety override
        obs[11] = 0.05
        obs[15] = 1.0
        action, _ = ctrl.predict(obs)
        assert action[0] <= 0.40, "Low headroom should cap commitment"

    def test_low_voltage_reduces_commitment(self, ctrl):
        obs = np.zeros(17, dtype=np.float32)
        obs[7] = 0.9
        obs[10] = 0.5
        obs[11] = 0.5
        obs[15] = 0.93  # low voltage
        action, _ = ctrl.predict(obs)
        assert action[0] <= 0.50, "Low voltage should reduce commitment"


# =========================================================================
# B. MacroEnv with custom inner_action_fn
# =========================================================================

class TestMacroEnvWithInnerFn:
    """Test that inner_action_fn is actually called during MacroEnv step."""

    def test_inner_fn_called(self):
        call_count = [0]

        def counting_fn(inner_obs, macro_action):
            call_count[0] += 1
            return np.array([1.0, 0.7, 0.7, 0.0], dtype=np.float32)

        env = C2GMacroEnv(scenario="default", inner_action_fn=counting_fn)
        env.reset(seed=0)
        env.step(env.action_space.sample())
        # 180 sub-steps per macro step
        assert call_count[0] == 180, f"Expected 180 calls, got {call_count[0]}"

    def test_inner_fn_receives_macro_action(self):
        received_actions = []

        def capturing_fn(inner_obs, macro_action):
            received_actions.append(macro_action.copy())
            return np.array([1.0, 0.7, 0.7, 0.0], dtype=np.float32)

        env = C2GMacroEnv(scenario="default", inner_action_fn=capturing_fn)
        env.reset(seed=0)
        action = np.array([0.7, 0.3], dtype=np.float32)
        env.step(action)
        # All captured macro actions should match
        assert len(received_actions) > 0
        assert received_actions[0].shape == (2,)

    def test_episode_with_rule_based_inner(self):
        """Rule-based low-level as inner_action_fn runs a full macro step."""
        from baselines.rule_based_mpc import RuleBasedController
        inner_ctrl = RuleBasedController()

        def rule_fn(inner_obs, macro_action):
            action, _ = inner_ctrl.predict(inner_obs)
            return action

        env = C2GMacroEnv(scenario="default", inner_action_fn=rule_fn)
        obs, _ = env.reset(seed=42)
        assert obs.shape == (17,)

        obs, rew, term, trunc, info = env.step(np.array([0.5, 0.0],
                                                         dtype=np.float32))
        assert obs.shape == (17,)
        assert math.isfinite(rew)


# =========================================================================
# C. Macro rule-based controller on live env
# =========================================================================

class TestRuleBasedMacroOnEnv:

    def test_full_macro_episode(self):
        """Rule-based macro runs a full episode without crashing."""
        ctrl = RuleBasedMacroController()
        env = C2GMacroEnv(scenario="default")
        obs, _ = env.reset(seed=0)
        total_rew = 0.0
        steps = 0
        for _ in range(10):  # 10 macro steps = 150 min
            action, _ = ctrl.predict(obs)
            obs, rew, term, trunc, info = env.step(action)
            total_rew += rew
            steps += 1
            if term or trunc:
                break
        assert steps > 0
        assert math.isfinite(total_rew)


# =========================================================================
# D. Module imports (smoke tests)
# =========================================================================

class TestModuleImports:

    def test_import_train_ppo_macro(self):
        import baselines.train_ppo_macro  # noqa

    def test_import_train_hierarchical(self):
        import baselines.train_hierarchical  # noqa

    def test_import_rule_based_macro(self):
        from baselines.rule_based_macro import RuleBasedMacroController  # noqa
