"""
Tests for baselines/safety_shield.py
======================================
Covers:
  - SafetyShield: thermal, SOC, frequency, voltage overrides
  - SafetyShield: no intervention when obs is safe
  - SafetyShield: stats tracking
  - ShieldedEnv: Gymnasium wrapper API
  - ShieldedAgent: predict interface
  - Integration: shielded episode survives where unshielded may not
"""
import math
import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from baselines.safety.safety_shield import SafetyShield, ShieldedEnv, ShieldedAgent, ShieldStats
from c2g_env import C2GFastEnv
from c2g_env.thermal_limits import T_SAFE


# =========================================================================
# A. SafetyShield unit tests
# =========================================================================

class TestSafetyShieldBasic:

    @pytest.fixture
    def shield(self):
        return SafetyShield()

    def _safe_obs(self):
        """Obs representing a completely safe state."""
        obs = np.zeros(17, dtype=np.float32)
        obs[0] = 28.0 / T_SAFE   # temp_A_norm ~ 0.8 (safe)
        obs[1] = 27.0 / T_SAFE   # temp_B_norm
        obs[2] = 0.5            # soc
        obs[14] = 0.0           # freq_dev_norm (nominal)
        obs[15] = 1.0           # v_pcc_pu (nominal)
        return obs

    def test_safe_action_unchanged(self, shield):
        """When obs is safe, action should pass through unmodified."""
        obs = self._safe_obs()
        action = np.array([0.8, 0.6, 0.6, 0.2], dtype=np.float32)
        safe, modified, info = shield.filter(action, obs)
        assert not modified
        np.testing.assert_array_equal(safe, action)

    def test_returns_correct_types(self, shield):
        obs = self._safe_obs()
        action = np.array([0.5, 0.5, 0.5, 0.0], dtype=np.float32)
        safe, modified, info = shield.filter(action, obs)
        assert isinstance(safe, np.ndarray)
        assert isinstance(modified, bool)
        assert isinstance(info, dict)
        assert safe.shape == (4,)

    def test_stats_incremented(self, shield):
        obs = self._safe_obs()
        action = np.array([0.5, 0.5, 0.5, 0.0], dtype=np.float32)
        shield.filter(action, obs)
        shield.filter(action, obs)
        assert shield.stats.total_steps == 2

    def test_reset_clears_stats(self, shield):
        obs = self._safe_obs()
        shield.filter(np.zeros(4, dtype=np.float32), obs)
        shield.reset()
        assert shield.stats.total_steps == 0


class TestThermalOverride:

    def test_high_temp_reduces_throttle(self):
        shield = SafetyShield()
        obs = np.zeros(17, dtype=np.float32)
        obs[0] = 34.5 / T_SAFE  # temp_A very close to T_safe
        obs[1] = 28.0 / T_SAFE
        obs[2] = 0.5
        obs[15] = 1.0
        action = np.array([1.0, 0.3, 0.3, 0.0], dtype=np.float32)
        safe, modified, info = shield.filter(action, obs)
        assert modified
        assert safe[0] < 1.0, "Throttle should be reduced near T_safe"
        assert safe[1] > 0.3, "Pump should be increased near T_safe"
        assert safe[2] > 0.3, "HVAC should be increased near T_safe"

    def test_at_tsafe_full_override(self):
        shield = SafetyShield()
        obs = np.zeros(17, dtype=np.float32)
        obs[0] = T_SAFE / T_SAFE  # exactly at T_safe
        obs[1] = 28.0 / T_SAFE
        obs[2] = 0.5
        obs[15] = 1.0
        action = np.array([1.0, 0.3, 0.3, 0.0], dtype=np.float32)
        safe, modified, info = shield.filter(action, obs)
        assert modified
        assert safe[0] == pytest.approx(0.0, abs=0.01), "Full thermal emergency → zero throttle"
        assert safe[1] == pytest.approx(1.0, abs=0.01), "Full thermal emergency → max pump"

    def test_cool_temp_no_thermal_override(self):
        shield = SafetyShield()
        obs = np.zeros(17, dtype=np.float32)
        obs[0] = 25.0 / T_SAFE  # well below T_shield
        obs[1] = 25.0 / T_SAFE
        obs[2] = 0.5
        obs[15] = 1.0
        action = np.array([1.0, 0.3, 0.3, 0.0], dtype=np.float32)
        safe, modified, _ = shield.filter(action, obs)
        assert not modified
        assert shield.stats.thermal_overrides == 0


class TestSOCOverride:

    def test_low_soc_blocks_discharge(self):
        shield = SafetyShield()
        obs = np.zeros(17, dtype=np.float32)
        obs[0] = 25.0 / T_SAFE
        obs[1] = 25.0 / T_SAFE
        obs[2] = 0.12  # near soc_min + guard
        obs[15] = 1.0
        action = np.array([0.8, 0.5, 0.5, 0.8], dtype=np.float32)  # wants to discharge
        safe, modified, _ = shield.filter(action, obs)
        assert modified
        assert safe[3] <= 0.0, "Should block discharge at low SOC"
        assert shield.stats.soc_overrides == 1

    def test_high_soc_blocks_charge(self):
        shield = SafetyShield()
        obs = np.zeros(17, dtype=np.float32)
        obs[0] = 25.0 / T_SAFE
        obs[1] = 25.0 / T_SAFE
        obs[2] = 0.93  # near soc_max - guard
        obs[15] = 1.0
        action = np.array([0.8, 0.5, 0.5, -0.8], dtype=np.float32)  # wants to charge
        safe, modified, _ = shield.filter(action, obs)
        assert modified
        assert safe[3] >= 0.0, "Should block charge at high SOC"

    def test_mid_soc_no_override(self):
        shield = SafetyShield()
        obs = np.zeros(17, dtype=np.float32)
        obs[0] = 25.0 / T_SAFE
        obs[1] = 25.0 / T_SAFE
        obs[2] = 0.5  # healthy SOC
        obs[15] = 1.0
        action = np.array([0.8, 0.5, 0.5, 0.5], dtype=np.float32)
        safe, modified, _ = shield.filter(action, obs)
        assert not modified
        assert shield.stats.soc_overrides == 0


class TestFrequencyOverride:

    def test_under_freq_blocks_charge(self):
        """Under-frequency + agent wants to charge → override to discharge."""
        shield = SafetyShield()
        obs = np.zeros(17, dtype=np.float32)
        obs[0] = 25.0 / T_SAFE
        obs[1] = 25.0 / T_SAFE
        obs[2] = 0.5
        obs[14] = -0.9  # severe under-frequency deviation
        obs[15] = 1.0
        action = np.array([0.8, 0.5, 0.5, -0.5], dtype=np.float32)  # wants to charge
        safe, modified, _ = shield.filter(action, obs)
        assert modified
        assert safe[3] > 0.0, "Should switch to discharge under under-frequency"
        assert shield.stats.freq_overrides == 1

    def test_over_freq_blocks_discharge(self):
        """Over-frequency + agent wants to discharge → override to charge."""
        shield = SafetyShield()
        obs = np.zeros(17, dtype=np.float32)
        obs[0] = 25.0 / T_SAFE
        obs[1] = 25.0 / T_SAFE
        obs[2] = 0.5
        obs[14] = 0.9  # severe over-frequency deviation
        obs[15] = 1.0
        action = np.array([0.8, 0.5, 0.5, 0.5], dtype=np.float32)  # wants to discharge
        safe, modified, _ = shield.filter(action, obs)
        assert modified
        assert safe[3] < 0.0, "Should switch to charge under over-frequency"

    def test_nominal_freq_no_override(self):
        shield = SafetyShield()
        obs = np.zeros(17, dtype=np.float32)
        obs[0] = 25.0 / T_SAFE
        obs[1] = 25.0 / T_SAFE
        obs[2] = 0.5
        obs[14] = 0.1  # slight deviation, within threshold
        obs[15] = 1.0
        action = np.array([0.8, 0.5, 0.5, 0.5], dtype=np.float32)
        safe, modified, _ = shield.filter(action, obs)
        assert not modified


class TestVoltageOverride:

    def test_low_voltage_reduces_throttle(self):
        shield = SafetyShield()
        obs = np.zeros(17, dtype=np.float32)
        obs[0] = 25.0 / T_SAFE
        obs[1] = 25.0 / T_SAFE
        obs[2] = 0.5
        obs[14] = 0.0
        obs[15] = 0.91  # below v_min_shield (0.92)
        action = np.array([1.0, 0.5, 0.5, 0.0], dtype=np.float32)
        safe, modified, _ = shield.filter(action, obs)
        assert modified
        assert safe[0] < 1.0, "Low voltage should reduce throttle"
        assert shield.stats.voltage_overrides == 1

    def test_nominal_voltage_no_override(self):
        shield = SafetyShield()
        obs = np.zeros(17, dtype=np.float32)
        obs[0] = 25.0 / T_SAFE
        obs[1] = 25.0 / T_SAFE
        obs[2] = 0.5
        obs[14] = 0.0
        obs[15] = 1.0
        action = np.array([1.0, 0.5, 0.5, 0.0], dtype=np.float32)
        safe, modified, _ = shield.filter(action, obs)
        assert not modified


# =========================================================================
# B. ShieldedEnv wrapper
# =========================================================================

class TestShieldedEnv:

    @pytest.fixture
    def senv(self):
        base = C2GFastEnv(scenario="default")
        return ShieldedEnv(base, shield=SafetyShield())

    def test_reset_returns_obs_info(self, senv):
        obs, info = senv.reset(seed=0)
        assert obs.shape == (18,)

    def test_step_returns_5_tuple(self, senv):
        senv.reset(seed=0)
        action = senv.action_space.sample()
        result = senv.step(action)
        assert len(result) == 5

    def test_info_has_shield_keys(self, senv):
        senv.reset(seed=0)
        _, _, _, _, info = senv.step(senv.action_space.sample())
        assert "shield_modified" in info
        assert "shield_stats" in info

    def test_spaces_match_base(self, senv):
        assert senv.observation_space.shape == (18,)
        assert senv.action_space.shape == (4,)

    def test_shield_stats_accessible(self, senv):
        senv.reset(seed=0)
        senv.step(senv.action_space.sample())
        stats = senv.shield.stats.as_dict()
        assert stats["shield_total_steps"] == 1


class TestShieldedAgent:

    def test_wraps_rule_based(self):
        from baselines.rule_based_mpc import RuleBasedController
        agent = RuleBasedController()
        safe_agent = ShieldedAgent(agent)
        obs = np.zeros(17, dtype=np.float32)
        obs[0] = 25.0 / T_SAFE
        obs[1] = 25.0 / T_SAFE
        obs[2] = 0.5
        obs[15] = 1.0
        action, state = safe_agent.predict(obs)
        assert action.shape == (4,)
        assert state is None


# =========================================================================
# C. Integration: shielded episode
# =========================================================================

class TestShieldedEpisode:

    def test_shielded_random_survives_longer(self):
        """A shielded random agent should survive at least as long as unshielded."""
        # Unshielded random
        env_raw = C2GFastEnv(scenario="default")
        obs, _ = env_raw.reset(seed=42)
        raw_steps = 0
        for _ in range(200):
            obs, _, term, trunc, _ = env_raw.step(env_raw.action_space.sample())
            raw_steps += 1
            if term or trunc:
                break

        # Shielded random
        env_safe = ShieldedEnv(C2GFastEnv(scenario="default"))
        obs, _ = env_safe.reset(seed=42)
        safe_steps = 0
        for _ in range(200):
            obs, _, term, trunc, _ = env_safe.step(env_safe.action_space.sample())
            safe_steps += 1
            if term or trunc:
                break

        assert safe_steps >= raw_steps, (
            f"Shielded ({safe_steps}) should survive >= unshielded ({raw_steps})"
        )

    def test_shielded_episode_no_thermal_fault(self):
        """Shielded agent should not trigger thermal faults within 100 steps."""
        env = ShieldedEnv(C2GFastEnv(scenario="default"))
        obs, _ = env.reset(seed=0)
        for _ in range(100):
            action = np.array([1.0, 0.2, 0.2, 0.0], dtype=np.float32)  # aggressive
            obs, _, terminated, truncated, info = env.step(action)
            if terminated:
                # If terminated, it should NOT be from thermal fault
                # (shield should prevent that)
                assert not info.get("thermal_fault", False), (
                    "Shield should prevent thermal faults"
                )
                break


class TestShieldStats:

    def test_intervention_rate(self):
        s = ShieldStats(total_steps=100, interventions=25)
        assert s.intervention_rate == pytest.approx(0.25)

    def test_as_dict(self):
        s = ShieldStats(total_steps=10, interventions=3,
                        thermal_overrides=1, soc_overrides=2)
        d = s.as_dict()
        assert d["shield_total_steps"] == 10
        assert d["shield_interventions"] == 3
        assert d["shield_thermal_overrides"] == 1


class TestModuleImports:

    def test_import_safety_shield(self):
        from baselines.safety.safety_shield import SafetyShield, ShieldedEnv, ShieldedAgent  # noqa

    def test_import_train_shielded(self):
        import baselines.safety.train_shielded_ppo  # noqa
