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

    def test_gate_targets_are_not_all_passthrough(self):
        from baselines.safety.safe_projection import build_gate_targets

        concepts = torch.tensor([[0.2, 0.3, 0.7, 0.9, 0.9, 0.9, 0.8, 1.0, 0.1, 0.6]])
        gate_targets = build_gate_targets(concepts, gate_alpha=0.5, gate_beta=0.3)

        assert gate_targets.shape == (1, 4)
        assert torch.all(gate_targets < 1.0)
        assert gate_targets[0, 0] < gate_targets[0, 2]
        assert gate_targets[0, 3] < 1.0

    def test_layer2_blends_toward_priors_not_pure_attenuation(self):
        from baselines.safety.safe_projection import compute_layer2_action

        class HalfGate(torch.nn.Module):
            def forward(self, concepts):
                return torch.full((concepts.shape[0], 4), 0.5, dtype=concepts.dtype, device=concepts.device)

        obs = torch.tensor([[34.5 / 35.0, 33.8 / 35.0, 0.5, 0.0, 0.0, 0.0, 0.8,
                             0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.2, 0.0]], dtype=torch.float32)
        concepts = torch.tensor([[0.1, 0.1, 0.8, 1.0, 1.0, 0.9, 0.7, 0.8, 0.2, 0.4]], dtype=torch.float32)
        policy_action = torch.tensor([[0.05, 0.10, 0.10, 0.0]], dtype=torch.float32)

        blended, gate_values, priors = compute_layer2_action(
            policy_action,
            concepts,
            obs=obs,
            safety_gate=HalfGate(),
        )
        attenuated = policy_action * 0.5

        assert torch.allclose(gate_values, torch.full_like(gate_values, 0.5))
        assert blended.shape == policy_action.shape
        assert torch.any(torch.abs(blended - attenuated) > 1e-6)
        assert blended[0, 0] > policy_action[0, 0]
        assert blended[0, 1] > attenuated[0, 1]
        assert blended[0, 2] > attenuated[0, 2]


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
        obs_space = gym.spaces.Box(low=-10, high=10, shape=(18,))
        fe = C2GConceptFeatureExtractor(obs_space, n_concepts=10)
        obs = torch.randn(4, 18)
        features = fe(obs)
        assert features.shape == (4, 28)  # 18 + 10

    def test_gated_feature_extractor_output_dim(self):
        from baselines.safety.concept_bottleneck import C2GGatedConceptFeatureExtractor
        import gymnasium as gym
        obs_space = gym.spaces.Box(low=-10, high=10, shape=(18,))
        fe = C2GGatedConceptFeatureExtractor(obs_space, n_concepts=10, action_dim=4)
        obs = torch.randn(4, 18)
        features = fe(obs)
        assert features.shape == (4, 32)  # 18 + 10 + 4


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
        obs_space = gym.spaces.Box(low=-10, high=10, shape=(18,))
        fe = C2GConceptFeatureExtractor(obs_space, n_concepts=10)
        obs = torch.randn(4, 18)
        features = fe(obs)
        assert features.shape == (4, 28)  # 18 obs + 10 concepts

    def test_cbm_gate_feature_extractor(self):
        """CBM+Gate uses C2GGatedConceptFeatureExtractor → obs + concepts + gate."""
        from baselines.safety.concept_bottleneck import C2GGatedConceptFeatureExtractor
        import gymnasium as gym
        obs_space = gym.spaces.Box(low=-10, high=10, shape=(18,))
        fe = C2GGatedConceptFeatureExtractor(obs_space, n_concepts=10, action_dim=4)
        obs = torch.randn(4, 18)
        features = fe(obs)
        assert features.shape == (4, 32)  # 18 obs + 10 concepts + 4 gate

    def test_cbm_only_no_gate(self):
        """CBM-Only feature extractor should NOT have a safety_gate attribute."""
        from baselines.safety.concept_bottleneck import C2GConceptFeatureExtractor
        import gymnasium as gym
        obs_space = gym.spaces.Box(low=-10, high=10, shape=(18,))
        fe = C2GConceptFeatureExtractor(obs_space, n_concepts=10)
        assert not hasattr(fe, 'safety_gate')

    def test_cbm_gate_has_gate(self):
        """CBM+Gate feature extractor should have a safety_gate attribute."""
        from baselines.safety.concept_bottleneck import C2GGatedConceptFeatureExtractor
        import gymnasium as gym
        obs_space = gym.spaces.Box(low=-10, high=10, shape=(18,))
        fe = C2GGatedConceptFeatureExtractor(obs_space, n_concepts=10, action_dim=4)
        assert hasattr(fe, 'safety_gate')

    def test_ablation_ladder_output_dims(self):
        """Verify output dim ladder: CBM-Only(27) < CBM+Gate(31)."""
        from baselines.safety.concept_bottleneck import (
            C2GConceptFeatureExtractor, C2GGatedConceptFeatureExtractor,
        )
        import gymnasium as gym
        obs_space = gym.spaces.Box(low=-10, high=10, shape=(18,))
        fe_cbm = C2GConceptFeatureExtractor(obs_space, n_concepts=10)
        fe_gate = C2GGatedConceptFeatureExtractor(obs_space, n_concepts=10, action_dim=4)
        assert fe_cbm.features_dim == 28
        assert fe_gate.features_dim == 32
        assert fe_cbm.features_dim < fe_gate.features_dim

    def test_cbm_shield_uses_same_extractor_as_cbm_only(self):
        """CBM+Shield uses the same non-gated extractor as CBM-Only (dim=28).
        The safety comes from the shield wrapper, not the feature extractor."""
        from baselines.safety.concept_bottleneck import C2GConceptFeatureExtractor
        import gymnasium as gym
        obs_space = gym.spaces.Box(low=-10, high=10, shape=(18,))
        fe = C2GConceptFeatureExtractor(obs_space, n_concepts=10)
        assert fe.features_dim == 28
        assert not hasattr(fe, 'safety_gate')


class TestGateBehavioral:
    """Behavioral tests verifying gate actually modifies actions."""

    def test_gate_attenuates_action_on_high_cooling_demand(self):
        """When cooling demand is high, a trained gate should reduce
        the throttle action (action 0)."""
        import torch
        from baselines.safety.safe_projection import SafeProjectionGate

        gate = SafeProjectionGate(concept_dim=10, action_dim=4, init_bias=2.0)
        # Manually set gate weights so high cooling_demand → low throttle gate
        # This simulates what the supervision loss achieves.
        with torch.no_grad():
            # Zero out the first layer, set bias of output to respond to concept 5 (cooling_demand_A)
            gate.gate[0].weight.zero_()
            gate.gate[0].bias.zero_()
            # Make hidden unit 0 respond to concept 5 (cooling_demand_A)
            gate.gate[0].weight[0, 5] = 5.0
            # Make output action 0 (throttle) respond negatively to hidden 0
            gate.gate[2].weight.zero_()
            gate.gate[2].bias.fill_(3.0)  # baseline: sigmoid(3) ≈ 0.95
            gate.gate[2].weight[0, 0] = -6.0  # throttle responds to cooling

        # Low cooling demand → gate ≈ 1 (pass-through)
        low_demand = torch.zeros(1, 10)
        low_demand[0, 5] = 0.1  # cooling_demand_A = low
        g_low = gate(low_demand)
        throttle_gate_low = g_low[0, 0].item()

        # High cooling demand → gate < 1 (attenuated)
        high_demand = torch.zeros(1, 10)
        high_demand[0, 5] = 0.9  # cooling_demand_A = high
        g_high = gate(high_demand)
        throttle_gate_high = g_high[0, 0].item()

        assert throttle_gate_low > throttle_gate_high, \
            f"Gate should attenuate throttle when cooling demand is high: " \
            f"low_demand_gate={throttle_gate_low:.3f}, high_demand_gate={throttle_gate_high:.3f}"
        assert throttle_gate_high < 0.7, \
            f"Gate should meaningfully reduce throttle: got {throttle_gate_high:.3f}"

    def test_gate_applied_in_wrapper(self):
        """HAC2GShieldWrapper.step() applies gate to actions in the real runtime path."""
        import torch
        import numpy as np
        import gymnasium as gym
        import importlib
        import sys
        from baselines.safety.concept_bottleneck import C2GConceptEncoder
        from baselines.safety.safe_projection import SafeProjectionGate

        # Import HAC2GShieldWrapper despite hydra dependency at module level
        # by mocking hydra/omegaconf if not installed
        for mod_name in ("hydra", "hydra.core", "hydra.core.hydra_config", "omegaconf"):
            if mod_name not in sys.modules:
                sys.modules[mod_name] = type(sys)("mock_" + mod_name)
        if not hasattr(sys.modules["omegaconf"], "DictConfig"):
            sys.modules["omegaconf"].DictConfig = type("DictConfig", (), {})
            sys.modules["omegaconf"].OmegaConf = type("OmegaConf", (), {})
        if not hasattr(sys.modules["hydra"], "main"):
            sys.modules["hydra"].main = lambda **kw: (lambda f: f)
        if not hasattr(sys.modules["hydra.core.hydra_config"], "HydraConfig"):
            sys.modules["hydra.core.hydra_config"].HydraConfig = type("HydraConfig", (), {})

        from baselines.train_ha_c2g import HAC2GShieldWrapper

        # Build a tiny 4-action dummy env with 17-D obs
        base_env = gym.make("MountainCarContinuous-v0")
        # Monkey-patch obs/action spaces to match C2G dimensions
        base_env.observation_space = gym.spaces.Box(
            low=-10, high=10, shape=(18,), dtype=np.float32)
        base_env.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(4,), dtype=np.float32)
        # Monkey-patch step/reset to return correct shapes
        _original_reset = base_env.reset
        _original_step = base_env.step
        def _fake_reset(**kw):
            _original_reset(**kw)
            obs = np.random.randn(18).astype(np.float32)
            return obs, {}
        def _fake_step(action):
            # Record what action the env actually received
            _fake_step.last_action = action.copy()
            obs = np.random.randn(18).astype(np.float32)
            return obs, 1.0, False, False, {}
        base_env.reset = _fake_reset
        base_env.step = _fake_step

        # Create encoder + gate with init_bias=0 → gate ≈ 0.5
        encoder = C2GConceptEncoder(obs_dim=18, n_concepts=10)
        gate = SafeProjectionGate(concept_dim=10, action_dim=4, init_bias=0.0)

        # Build a passthrough mock shield
        class PassthroughShield:
            def reset(self): pass
            def filter(self, action, obs):
                self.received_action = action.copy()
                return action, False, {}
            class stats:
                @staticmethod
                def as_dict(): return {}

        mock_shield = PassthroughShield()

        wrapper = HAC2GShieldWrapper(
            base_env,
            shield=mock_shield,
            shield_penalty=0.5,
            concept_encoder=encoder,
            safety_gate=gate,
        )
        wrapper.reset()

        raw_action = np.array([0.8, 0.7, 0.6, 0.5], dtype=np.float32)
        obs, reward, term, trunc, info = wrapper.step(raw_action)

        # The shield should have received the GATED action, not the raw one
        received = mock_shield.received_action
        assert not np.allclose(received, raw_action, atol=0.01), \
            f"Shield received raw action unchanged — gate not applied! " \
            f"raw={raw_action}, received={received}"
        # With init_bias=0 → gate ≈ 0.5, so received ≈ raw * 0.5
        assert np.all(np.abs(received) < np.abs(raw_action) + 1e-6), \
            "Gated action magnitude should be ≤ raw action magnitude"
        assert info["gate_applied"] is True
        assert "shield_active" in info


# ═══════════════════════════════════════════════════════════════════
# Statistical Analysis Tests
# ═══════════════════════════════════════════════════════════════════

class TestStatisticalAnalysis:
    """Tests for multi-seed CI and significance infrastructure."""

    def test_bootstrap_ci_basic(self):
        from evaluation.statistical_analysis import bootstrap_ci
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        lo, hi = bootstrap_ci(values, confidence=0.95)
        mean = np.mean(values)
        assert lo < mean < hi, f"CI should bracket mean: [{lo}, {hi}], mean={mean}"
        assert hi - lo > 0.5, "CI should have non-trivial width"
        assert hi - lo < 8.0, "CI should not be wider than the range"

    def test_bootstrap_ci_single_value(self):
        from evaluation.statistical_analysis import bootstrap_ci
        lo, hi = bootstrap_ci(np.array([42.0]))
        assert lo == 42.0 and hi == 42.0

    def test_summarise_metric(self):
        from evaluation.statistical_analysis import summarise_metric
        values = np.random.RandomState(42).randn(20) * 0.1 + 0.95
        s = summarise_metric(values, "survival_rate", "ha_c2g", "default")
        assert s.n_seeds == 20
        assert s.ci_lower < s.mean < s.ci_upper
        assert s.iqr_lower <= s.median <= s.iqr_upper
        assert s.min_val <= s.mean <= s.max_val

    def test_cohens_d(self):
        from evaluation.statistical_analysis import cohens_d, effect_label
        a = np.array([10.0, 11.0, 12.0, 10.5, 11.5])
        b = np.array([5.0, 6.0, 5.5, 6.5, 5.5])
        d = cohens_d(a, b)
        assert d > 2.0, f"Large effect expected, got d={d}"
        assert effect_label(d) == "large"
        # Small effect
        c = np.array([10.0, 10.1, 10.2, 10.0, 10.1])
        d_small = cohens_d(a, c)
        assert abs(d_small) < abs(d)

    def test_pairwise_test(self):
        from evaluation.statistical_analysis import pairwise_test
        np.random.seed(42)
        a = np.random.randn(20) + 5.0  # clearly different
        b = np.random.randn(20) + 0.0
        comp = pairwise_test(a, b, "reward", "default", "agent_a", "agent_b")
        assert comp.significant, f"Should be significant, p={comp.p_value}"
        assert comp.p_value < 0.001
        assert comp.effect_label == "large"

    def test_pairwise_test_not_significant(self):
        from evaluation.statistical_analysis import pairwise_test
        np.random.seed(42)
        a = np.random.randn(10) + 5.0
        b = np.random.randn(10) + 5.0  # same distribution
        comp = pairwise_test(a, b, "reward", "default", "agent_a", "agent_b")
        # May or may not be significant, but p should not be tiny
        assert comp.p_value > 0.001

    def test_latex_table_generation(self):
        from evaluation.statistical_analysis import (
            summarise_metric, generate_latex_table,
        )
        sums = []
        for agent in ["ha_c2g", "cbm_only", "random"]:
            vals = np.random.RandomState(hash(agent) % 2**31).randn(10) + 0.5
            s = summarise_metric(vals, "mean_reward", agent, "default")
            sums.append(s)
        latex = generate_latex_table(sums, metrics=["mean_reward"], scenario="default")
        assert "\\begin{table}" in latex
        assert "\\end{table}" in latex
        assert "HA-C2G" in latex
        assert "\\pm" in latex


# ═══════════════════════════════════════════════════════════════════
# Failure Analysis Tests
# ═══════════════════════════════════════════════════════════════════

class TestFailureAnalysis:
    """Tests for failure-case analysis infrastructure."""

    def test_per_constraint_margins(self):
        from evaluation.failure_analysis import compute_per_constraint_margins
        obs = np.zeros(17, dtype=np.float32)
        obs[0] = 30.0 / 35.0  # T_A = 30°C → margin = 5°C
        obs[1] = 34.0 / 35.0  # T_B = 34°C → margin = 1°C
        obs[2] = 0.5           # SOC = 0.5
        obs[14] = 0.0          # freq_dev = 0
        obs[15] = 1.0          # v_pcc = 1.0 pu
        margins = compute_per_constraint_margins(obs)
        assert margins["C1:T_A<35"] == pytest.approx(5.0, abs=0.1)
        assert margins["C2:T_B<35"] == pytest.approx(1.0, abs=0.1)
        assert margins["C3:SOC_lo"] > 0
        assert margins["C5:voltage"] > 0

    def test_identify_violations(self):
        from evaluation.failure_analysis import identify_violations
        margins = {"C1:T_A<35": 2.0, "C2:T_B<35": -0.5, "C3:SOC_lo": 0.3,
                   "C3:SOC_hi": 0.4, "C4:freq": -0.1, "C5:voltage": 0.05}
        viols = identify_violations(margins)
        assert "C2:T_B<35" in viols
        assert "C4:freq" in viols
        assert "C1:T_A<35" not in viols

    def test_closest_constraint(self):
        from evaluation.failure_analysis import closest_constraint
        margins = {"C1:T_A<35": 2.0, "C2:T_B<35": 0.1, "C5:voltage": 0.5}
        name, val = closest_constraint(margins)
        assert name == "C2:T_B<35"
        assert val == pytest.approx(0.1)

    def test_episode_trace_properties(self):
        from evaluation.failure_analysis import (
            EpisodeTrace, ConstraintViolation,
        )
        trace = EpisodeTrace(
            agent="test", scenario="default", seed=42,
            total_reward=100.0, episode_length=288, survived=True,
        )
        trace.violations = [
            ConstraintViolation(10, "C1:T_A<35", -0.5, [0]*17),
            ConstraintViolation(20, "C2:T_B<35", -0.3, [0]*17),
            ConstraintViolation(30, "C1:T_A<35", -0.1, [0]*17),
        ]
        trace.margin_trajectory = [1.0, 0.5, -0.5, -0.3, 0.2]
        assert trace.n_violations == 3
        assert trace.violation_rate == pytest.approx(3 / 288)
        assert trace.worst_margin == -0.5
        assert trace.violated_constraints == {"C1:T_A<35", "C2:T_B<35"}
        assert trace.first_violation_step == 10

    def test_build_failure_profile(self):
        from evaluation.failure_analysis import (
            EpisodeTrace, ConstraintViolation, ShieldIntervention,
            build_failure_profile, CONSTRAINT_NAMES,
        )
        traces = []
        for seed in range(5):
            t = EpisodeTrace("agent", "default", seed, 100.0, 288, True)
            t.margin_trajectory = [1.0, 0.5, 0.1]
            t.per_constraint_margins = {c: [1.0] for c in CONSTRAINT_NAMES}
            if seed % 2 == 0:
                t.violations.append(
                    ConstraintViolation(50, "C1:T_A<35", -0.1, [0]*17))
            t.interventions.append(
                ShieldIntervention(10, [0.5]*4, [0.4]*4, 0.1, "C1:T_A<35"))
            traces.append(t)
        profile = build_failure_profile(traces)
        assert profile.n_seeds == 5
        assert profile.total_violations == 3  # seeds 0, 2, 4
        assert profile.total_interventions == 5
        assert profile.per_constraint_violation_counts["C1:T_A<35"] == 3
        assert profile.survival_rate == 1.0

    def test_comparative_failures(self):
        from evaluation.failure_analysis import (
            EpisodeTrace, ConstraintViolation, find_comparative_failures,
        )
        # Agent A fails on seed 0, agent B doesn't
        ta = EpisodeTrace("A", "default", 0, 90.0, 288, True)
        ta.violations = [ConstraintViolation(10, "C1:T_A<35", -0.2, [0]*17)]
        ta.margin_trajectory = [-0.2]

        tb = EpisodeTrace("B", "default", 0, 95.0, 288, True)
        tb.violations = []
        tb.margin_trajectory = [0.5]

        cfs = find_comparative_failures([ta], [tb], "A", "B")
        assert len(cfs) == 1
        assert cfs[0].failing_agent == "A"
        assert cfs[0].succeeding_agent == "B"
        assert "C1:T_A<35" in cfs[0].constraints_only_failing_violates
