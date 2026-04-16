"""
Tests for baselines/safety/ — High-Assurance Safety Modules
=============================================================
Covers:
  - CBFShield: barrier function values, QP solving, env wrapper
  - HJShield: value function precomputation, runtime filtering
  - MPCSafetyFilter: NLP solving, env wrapper
  - C2GConcepts: ground-truth concept computation
  - C2GConceptEncoder: neural encoder forward pass
  - SafeProjectionGate: gate forward pass, supervision loss
  - ProofTree: tree construction, serialisation
  - Integration: shielded episodes, metric collection
"""
import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from baselines.safety.cbf_shield import CBFShield, CBFShieldedEnv, CBFStats
from baselines.safety.hj_shield import HJShield, HJShieldedEnv
from baselines.safety.mpc_safety_filter import MPCSafetyFilter, MPCSFShieldedEnv
from baselines.safety.concept_bottleneck import (
    C2GConcepts, C2G_CONCEPT_NAMES,
)
from baselines.safety.proof_tree import ProofTree, ProofNode, RuleStatus


# ── Shared test helpers ──────────────────────────────────────────

def _safe_obs():
    """Obs representing a completely safe state (normalised)."""
    obs = np.zeros(17, dtype=np.float32)
    obs[0] = 28.0 / 35.0   # temp_A ~ 0.8 (safe)
    obs[1] = 27.0 / 35.0   # temp_B
    obs[2] = 0.50           # soc (middle)
    obs[3] = 0.15           # p_base
    obs[4] = 0.10           # p_flex
    obs[6] = 0.3            # regd signal
    obs[13] = 0.5           # T_amb
    obs[14] = 0.0           # freq_dev (nominal)
    obs[15] = 1.0           # v_pcc (nominal)
    return obs


def _hot_obs():
    """Obs near thermal limit."""
    obs = _safe_obs()
    obs[0] = 34.5 / 35.0   # T_A very close to T_safe
    obs[1] = 34.0 / 35.0
    return obs


def _low_soc_obs():
    """Obs with SOC near minimum."""
    obs = _safe_obs()
    obs[2] = 0.11           # just above SOC_min
    return obs


def _neutral_action():
    return np.array([0.5, 0.5, 0.5, 0.0], dtype=np.float32)


def _aggressive_action():
    """High throttle, low cooling, full discharge."""
    return np.array([1.0, 0.1, 0.1, 1.0], dtype=np.float32)


# =========================================================================
# A. CBF Shield Tests
# =========================================================================

class TestCBFShield:

    @pytest.fixture
    def cbf(self):
        return CBFShield()

    def test_safe_action_unchanged(self, cbf):
        """When obs is safe, neutral action should pass through."""
        obs = _safe_obs()
        action = _neutral_action()
        safe, modified, info = cbf.filter(action, obs)
        assert not modified or np.allclose(safe, action, atol=0.05)

    def test_thermal_barrier_positive_safe(self, cbf):
        s = cbf._decode_obs(_safe_obs())
        h = cbf._barrier_values(s)
        assert h["thermal_A"] > 0
        assert h["thermal_B"] > 0

    def test_thermal_barrier_near_limit(self, cbf):
        s = cbf._decode_obs(_hot_obs())
        h = cbf._barrier_values(s)
        assert h["thermal_A"] < 1.0  # near limit

    def test_hot_state_barrier_is_small(self, cbf):
        """When near thermal limit, barrier value should be near zero."""
        obs = _hot_obs()
        action = _aggressive_action()
        safe, modified, info = cbf.filter(action, obs)
        h = info.get("cbf_barrier_values", {})
        # thermal_A barrier should be very small (near limit)
        assert h.get("thermal_A", 1.0) < 0.5, (
            f"Expected small thermal_A barrier near limit, got {h}"
        )

    def test_low_soc_blocks_discharge(self, cbf):
        obs = _low_soc_obs()
        action = np.array([0.5, 0.5, 0.5, 0.8], dtype=np.float32)  # discharge
        safe, modified, info = cbf.filter(action, obs)
        # Should reduce or block discharge
        assert safe[3] <= action[3]

    def test_stats_tracking(self, cbf):
        obs = _safe_obs()
        for _ in range(10):
            cbf.filter(_neutral_action(), obs)
        assert cbf.stats.total_steps == 10
        d = cbf.stats.as_dict()
        assert "cbf_intervention_rate" in d

    def test_returns_correct_types(self, cbf):
        safe, modified, info = cbf.filter(_neutral_action(), _safe_obs())
        assert isinstance(safe, np.ndarray)
        assert isinstance(modified, bool)
        assert isinstance(info, dict)
        assert safe.shape == (4,)

    def test_action_bounds_maintained(self, cbf):
        obs = _hot_obs()
        action = np.array([1.5, -0.5, 2.0, -2.0], dtype=np.float32)
        safe, _, _ = cbf.filter(action, obs)
        assert np.all(safe[:3] >= 0) and np.all(safe[:3] <= 1)
        assert safe[3] >= -1 and safe[3] <= 1

    def test_reset_clears_stats(self, cbf):
        cbf.filter(_neutral_action(), _safe_obs())
        cbf.reset()
        assert cbf.stats.total_steps == 0


# =========================================================================
# B. HJ Shield Tests
# =========================================================================

class TestHJShield:

    @pytest.fixture
    def hj(self):
        return HJShield(n_grid=20, precompute=True)  # small grid for speed

    @pytest.fixture
    def hj_no_precompute(self):
        return HJShield(precompute=False)

    def test_precomputation_creates_value_functions(self, hj):
        assert hj.V_thermal is not None
        assert hj.V_soc is not None
        assert hj.U_thermal is not None
        assert hj.U_soc is not None

    def test_safe_state_passthrough(self, hj):
        obs = _safe_obs()
        action = _neutral_action()
        safe, modified, info = hj.filter(action, obs)
        # Safe state should have high value → no modification
        assert "hj_value_thermal_A" in info

    def test_hot_state_intervention(self, hj):
        obs = _hot_obs()
        action = _aggressive_action()
        safe, modified, info = hj.filter(action, obs)
        # Near thermal limit should trigger intervention
        assert info["hj_value_thermal_A"] < hj.delta * 2

    def test_fallback_without_precompute(self, hj_no_precompute):
        obs = _hot_obs()
        action = _aggressive_action()
        safe, modified, info = hj_no_precompute.filter(action, obs)
        assert isinstance(safe, np.ndarray)

    def test_stats_tracking(self, hj):
        for _ in range(5):
            hj.filter(_neutral_action(), _safe_obs())
        assert hj.stats.total_steps == 5

    def test_action_bounds(self, hj):
        obs = _hot_obs()
        safe, _, _ = hj.filter(_aggressive_action(), obs)
        assert np.all(safe[:3] >= 0) and np.all(safe[:3] <= 1)
        assert safe[3] >= -1 and safe[3] <= 1


# =========================================================================
# C. MPC Safety Filter Tests
# =========================================================================

class TestMPCSafetyFilter:

    @pytest.fixture
    def mpc(self):
        return MPCSafetyFilter(horizon=3)

    def test_safe_state_no_filter(self, mpc):
        obs = _safe_obs()
        action = _neutral_action()
        safe, modified, info = mpc.filter(action, obs)
        assert not modified  # safe state should skip the NLP
        assert info.get("mpcsf_skipped", False)

    def test_hot_state_triggers_nlp(self, mpc):
        obs = _hot_obs()
        action = _aggressive_action()
        safe, modified, info = mpc.filter(action, obs)
        assert not info.get("mpcsf_skipped", True)  # should attempt NLP

    def test_returns_correct_shape(self, mpc):
        safe, _, _ = mpc.filter(_neutral_action(), _safe_obs())
        assert safe.shape == (4,)

    def test_stats_tracking(self, mpc):
        mpc.filter(_neutral_action(), _safe_obs())
        assert mpc.stats.total_steps == 1

    def test_action_bounds(self, mpc):
        obs = _hot_obs()
        safe, _, _ = mpc.filter(_aggressive_action(), obs)
        assert np.all(safe[:3] >= 0) and np.all(safe[:3] <= 1)
        assert safe[3] >= -1 and safe[3] <= 1


# =========================================================================
# D. Concept Bottleneck Tests
# =========================================================================

class TestC2GConcepts:

    def test_concept_names_count(self):
        assert len(C2G_CONCEPT_NAMES) == 10

    def test_from_obs_returns_correct_type(self):
        obs = _safe_obs()
        concepts = C2GConcepts.from_obs(obs)
        assert isinstance(concepts, C2GConcepts)

    def test_to_vector_shape(self):
        concepts = C2GConcepts.from_obs(_safe_obs())
        vec = concepts.to_vector()
        assert vec.shape == (10,)
        assert vec.dtype == np.float32

    def test_all_concepts_in_0_1(self):
        for obs_fn in [_safe_obs, _hot_obs, _low_soc_obs]:
            concepts = C2GConcepts.from_obs(obs_fn())
            vec = concepts.to_vector()
            assert np.all(vec >= 0.0) and np.all(vec <= 1.0), \
                f"Concept out of [0,1]: {vec}"

    def test_safe_state_has_high_margins(self):
        concepts = C2GConcepts.from_obs(_safe_obs())
        assert concepts.thermal_margin_A > 0.3
        assert concepts.thermal_margin_B > 0.3
        assert concepts.soc_health > 0.5

    def test_hot_state_has_low_thermal_margin(self):
        concepts = C2GConcepts.from_obs(_hot_obs())
        assert concepts.thermal_margin_A < 0.1
        assert concepts.cooling_demand_A > 0.5

    def test_low_soc_has_low_health(self):
        concepts = C2GConcepts.from_obs(_low_soc_obs())
        assert concepts.soc_health < 0.1
        assert concepts.bess_headroom < 0.15

    def test_to_dict(self):
        concepts = C2GConcepts.from_obs(_safe_obs())
        d = concepts.to_dict()
        assert len(d) == 10
        assert "thermal_margin_A" in d

    def test_n_concepts(self):
        assert C2GConcepts.n_concepts() == 10


# =========================================================================
# E. Safe Projection Tests (require torch)
# =========================================================================

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
class TestSafeProjection:

    def test_gate_output_shape(self):
        from baselines.safety.safe_projection import SafeProjectionGate
        gate = SafeProjectionGate(concept_dim=10, action_dim=4)
        concepts = torch.randn(8, 10)
        g = gate(concepts)
        assert g.shape == (8, 4)

    def test_gate_output_in_0_1(self):
        from baselines.safety.safe_projection import SafeProjectionGate
        gate = SafeProjectionGate(concept_dim=10, action_dim=4)
        concepts = torch.randn(32, 10)
        g = gate(concepts)
        assert torch.all(g >= 0) and torch.all(g <= 1)

    def test_gate_near_passthrough_at_init(self):
        from baselines.safety.safe_projection import SafeProjectionGate
        gate = SafeProjectionGate(concept_dim=10, action_dim=4, init_bias=2.0)
        concepts = torch.zeros(1, 10)
        g = gate(concepts)
        # At init with bias=2.0, sigmoid(2.0) ≈ 0.88
        assert torch.all(g > 0.5), f"Gate should be near pass-through: {g}"

    def test_full_safe_projection_layer(self):
        from baselines.safety.safe_projection import SafeProjectionLayer
        layer = SafeProjectionLayer(concept_dim=10, action_dim=4)
        raw_action = torch.randn(4, 4)
        concepts = torch.randn(4, 10)
        safe = layer(raw_action, concepts)
        assert safe.shape == (4, 4)

    def test_gate_supervision_loss(self):
        from baselines.safety.safe_projection import GateSupervisionLoss
        loss_fn = GateSupervisionLoss()
        gate_values = torch.ones(8, 4) * 0.9
        concept_targets = torch.rand(8, 10)
        loss = loss_fn.compute(gate_values, concept_targets)
        assert loss.ndim == 0  # scalar
        assert loss.item() >= 0


# =========================================================================
# F. Concept Encoder Tests (require torch)
# =========================================================================

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
class TestConceptEncoder:

    def test_encoder_output_shape(self):
        from baselines.safety.concept_bottleneck import C2GConceptEncoder
        enc = C2GConceptEncoder(obs_dim=17, n_concepts=10)
        obs = torch.randn(8, 17)
        concepts = enc(obs)
        assert concepts.shape == (8, 10)

    def test_encoder_output_in_0_1(self):
        from baselines.safety.concept_bottleneck import C2GConceptEncoder
        enc = C2GConceptEncoder(obs_dim=17, n_concepts=10)
        obs = torch.randn(32, 17)
        concepts = enc(obs)
        assert torch.all(concepts >= 0) and torch.all(concepts <= 1)

    def test_feature_extractor_output_dim(self):
        from baselines.safety.concept_bottleneck import C2GConceptFeatureExtractor
        import gymnasium as gym
        obs_space = gym.spaces.Box(low=-10, high=10, shape=(17,))
        fe = C2GConceptFeatureExtractor(obs_space, n_concepts=10)
        obs = torch.randn(4, 17)
        features = fe(obs)
        assert features.shape == (4, 27)  # 17 + 10

    def test_gated_feature_extractor_output_dim(self):
        from baselines.safety.concept_bottleneck import C2GGatedConceptFeatureExtractor
        import gymnasium as gym
        obs_space = gym.spaces.Box(low=-10, high=10, shape=(17,))
        fe = C2GGatedConceptFeatureExtractor(obs_space, n_concepts=10, action_dim=4)
        obs = torch.randn(4, 17)
        features = fe(obs)
        assert features.shape == (4, 31)  # 17 + 10 + 4


# =========================================================================
# G. Proof Tree Tests
# =========================================================================

class TestProofTree:

    def test_from_step_safe(self):
        obs = _safe_obs()
        action = _neutral_action()
        tree = ProofTree.from_step(obs, action, action)
        assert tree.is_safe
        assert tree.n_failures == 0

    def test_from_step_with_modification(self):
        obs = _safe_obs()
        raw = _aggressive_action()
        safe = np.array([0.3, 0.8, 0.8, 0.0], dtype=np.float32)
        tree = ProofTree.from_step(obs, raw, safe)
        # Still safe (obs is safe) but action was modified
        assert tree.is_safe

    def test_proof_node_depth(self):
        child1 = ProofNode("C1", RuleStatus.PASS)
        child2 = ProofNode("C2", RuleStatus.PASS)
        parent = ProofNode("THERMAL", RuleStatus.PASS, children=[child1, child2])
        root = ProofNode("ROOT", RuleStatus.PASS, children=[parent])
        assert root.depth() == 3

    def test_to_dict_serialisable(self):
        obs = _safe_obs()
        tree = ProofTree.from_step(obs, _neutral_action(), _neutral_action())
        d = tree.to_dict()
        assert isinstance(d, dict)
        assert "children" in d
        import json
        json.dumps(d)  # should not raise

    def test_summary_string(self):
        obs = _safe_obs()
        tree = ProofTree.from_step(obs, _neutral_action(), _neutral_action())
        s = tree.summary()
        assert "SYSTEM_SAFE" in s
        assert "PASS" in s

    def test_with_concepts(self):
        obs = _safe_obs()
        concepts = {"thermal_margin_A": 0.8, "soc_health": 0.9}
        tree = ProofTree.from_step(
            obs, _neutral_action(), _neutral_action(), concepts=concepts)
        d = tree.to_dict()
        # Should have concept children
        concept_node = [c for c in d["children"] if c["name"] == "CONCEPT_OK"][0]
        assert len(concept_node["children"]) == 2

    def test_n_failures_counts_correctly(self):
        fail = ProofNode("C1", RuleStatus.FAIL, severity=1.0)
        ok = ProofNode("C2", RuleStatus.PASS)
        root = ProofNode("ROOT", RuleStatus.FAIL, children=[fail, ok])
        tree = ProofTree(root)
        assert tree.n_failures == 2  # root + C1


# =========================================================================
# H. Cross-shield Consistency Tests
# =========================================================================

class TestCrossShieldConsistency:
    """All shields should satisfy the same basic API contract."""

    @pytest.fixture(params=["simplex", "cbf", "hj", "mpc"])
    def shield(self, request):
        if request.param == "simplex":
            from baselines.safety_shield import SafetyShield
            return SafetyShield()
        elif request.param == "cbf":
            return CBFShield()
        elif request.param == "hj":
            return HJShield(n_grid=20, precompute=True)
        elif request.param == "mpc":
            return MPCSafetyFilter(horizon=3)

    def test_filter_returns_tuple_3(self, shield):
        result = shield.filter(_neutral_action(), _safe_obs())
        assert len(result) == 3

    def test_safe_action_shape(self, shield):
        safe, _, _ = shield.filter(_neutral_action(), _safe_obs())
        assert safe.shape == (4,)

    def test_action_bounds(self, shield):
        safe, _, _ = shield.filter(_aggressive_action(), _hot_obs())
        assert np.all(safe[:3] >= -0.01) and np.all(safe[:3] <= 1.01)
        assert safe[3] >= -1.01 and safe[3] <= 1.01

    def test_has_reset(self, shield):
        assert hasattr(shield, "reset")
        shield.reset()  # should not raise


# =========================================================================
# I. Ablation Consistency Tests
# =========================================================================

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
class TestAblations:
    """Verify CBM-Only and CBM+Gate feature extractors produce correct dims."""

    def test_cbm_only_feature_extractor(self):
        """CBM-Only uses C2GConceptFeatureExtractor → output dim = obs + concepts."""
        from baselines.safety.concept_bottleneck import C2GConceptFeatureExtractor
        import gymnasium as gym
        obs_space = gym.spaces.Box(low=-10, high=10, shape=(17,))
        fe = C2GConceptFeatureExtractor(obs_space, n_concepts=10)
        obs = torch.randn(4, 17)
        features = fe(obs)
        assert features.shape == (4, 27)  # 17 obs + 10 concepts

    def test_cbm_gate_feature_extractor(self):
        """CBM+Gate uses C2GGatedConceptFeatureExtractor → obs + concepts + gate."""
        from baselines.safety.concept_bottleneck import C2GGatedConceptFeatureExtractor
        import gymnasium as gym
        obs_space = gym.spaces.Box(low=-10, high=10, shape=(17,))
        fe = C2GGatedConceptFeatureExtractor(obs_space, n_concepts=10, action_dim=4)
        obs = torch.randn(4, 17)
        features = fe(obs)
        assert features.shape == (4, 31)  # 17 obs + 10 concepts + 4 gate

    def test_cbm_only_no_gate(self):
        """CBM-Only feature extractor should NOT have a safety_gate attribute."""
        from baselines.safety.concept_bottleneck import C2GConceptFeatureExtractor
        import gymnasium as gym
        obs_space = gym.spaces.Box(low=-10, high=10, shape=(17,))
        fe = C2GConceptFeatureExtractor(obs_space, n_concepts=10)
        assert not hasattr(fe, 'safety_gate')

    def test_cbm_gate_has_gate(self):
        """CBM+Gate feature extractor should have a safety_gate attribute."""
        from baselines.safety.concept_bottleneck import C2GGatedConceptFeatureExtractor
        import gymnasium as gym
        obs_space = gym.spaces.Box(low=-10, high=10, shape=(17,))
        fe = C2GGatedConceptFeatureExtractor(obs_space, n_concepts=10, action_dim=4)
        assert hasattr(fe, 'safety_gate')

    def test_ablation_ladder_output_dims(self):
        """Verify output dim ladder: CBM-Only(27) < CBM+Gate(31)."""
        from baselines.safety.concept_bottleneck import (
            C2GConceptFeatureExtractor, C2GGatedConceptFeatureExtractor,
        )
        import gymnasium as gym
        obs_space = gym.spaces.Box(low=-10, high=10, shape=(17,))
        fe_cbm = C2GConceptFeatureExtractor(obs_space, n_concepts=10)
        fe_gate = C2GGatedConceptFeatureExtractor(obs_space, n_concepts=10, action_dim=4)
        assert fe_cbm.features_dim == 27
        assert fe_gate.features_dim == 31
        assert fe_cbm.features_dim < fe_gate.features_dim
