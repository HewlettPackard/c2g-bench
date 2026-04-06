"""
Tests for c2g_env/simulators/renewable.py
==========================================
Covers:
  - Normal operation: get_generation returns dict with all keys
  - Value ranges: all power outputs in [0, capacity]
  - Wind power curve edge cases: cut-in, rated, cut-out
  - Solar power: zero at night (GHI=0), max at STC (GHI=1000)
  - Renewable total = wind + solar
  - Tick looping: tick wraps correctly without crash
  - wind_power / solar_power called directly with edge values
  - Negative GHI clamped to zero
  - Data loading: both wind and solar traces load correctly
  - Missing data gracefully degrades to zero generation
"""
import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from c2g_env.simulators.renewable import RenewableGen

RENEWABLE_DIR = "data/processed/renewable"
EXPECTED_KEYS = {"p_wind_mw", "p_solar_mw", "p_renewable_mw",
                 "wind_speed_ms", "ghi_wm2"}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rgen():
    return RenewableGen(
        renewable_dir=RENEWABLE_DIR,
        wind_capacity_mw=100.0,
        solar_capacity_mw=75.0,
    )


# ---------------------------------------------------------------------------
# 1. Return format
# ---------------------------------------------------------------------------

def test_get_generation_returns_dict(rgen):
    assert isinstance(rgen.get_generation(0), dict)


def test_get_generation_has_all_keys(rgen):
    result = rgen.get_generation(0)
    assert EXPECTED_KEYS.issubset(set(result.keys()))


def test_all_values_finite(rgen):
    r = rgen.get_generation(10)
    for k, v in r.items():
        assert np.isfinite(v), f"Key '{k}' is not finite: {v}"


# ---------------------------------------------------------------------------
# 2. Power output ranges
# ---------------------------------------------------------------------------

def test_wind_power_bounded(rgen):
    for tick in range(0, rgen.len_wind, 10):
        r = rgen.get_generation(tick)
        assert 0.0 <= r["p_wind_mw"] <= rgen.wind_capacity_mw, (
            f"tick={tick}: wind {r['p_wind_mw']} MW outside [0, {rgen.wind_capacity_mw}]"
        )


def test_solar_power_bounded(rgen):
    for tick in range(0, rgen.len_solar, 10):
        r = rgen.get_generation(tick)
        assert 0.0 <= r["p_solar_mw"] <= rgen.solar_capacity_mw, (
            f"tick={tick}: solar {r['p_solar_mw']} MW outside [0, {rgen.solar_capacity_mw}]"
        )


def test_renewable_total_equals_sum(rgen):
    for tick in range(0, min(rgen.len_wind, 50)):
        r = rgen.get_generation(tick)
        assert r["p_renewable_mw"] == pytest.approx(
            r["p_wind_mw"] + r["p_solar_mw"], rel=1e-9
        )


def test_renewable_total_non_negative(rgen):
    for tick in range(100):
        r = rgen.get_generation(tick)
        assert r["p_renewable_mw"] >= 0.0


# ---------------------------------------------------------------------------
# 3. Wind power curve
# ---------------------------------------------------------------------------

def test_wind_zero_below_cut_in(rgen):
    """Below cut-in speed (3 m/s), wind turbines produce nothing."""
    assert rgen.wind_power(0.0) == pytest.approx(0.0)
    assert rgen.wind_power(rgen.v_cut_in - 0.01) == pytest.approx(0.0)


def test_wind_zero_at_cut_out(rgen):
    """At or above cut-out speed (25 m/s), wind turbines shut down."""
    assert rgen.wind_power(rgen.v_cut_out) == pytest.approx(0.0)
    assert rgen.wind_power(rgen.v_cut_out + 10.0) == pytest.approx(0.0)


def test_wind_near_rated_close_to_capacity(rgen):
    """At rated speed (12 m/s), output should approach capacity."""
    p = rgen.wind_power(rgen.v_rated)
    assert p >= 0.8 * rgen.wind_capacity_mw, (
        f"Wind at rated speed {p:.1f} MW < 80% of capacity"
    )


def test_wind_monotone_below_rated(rgen):
    """Wind power must be monotonically increasing between cut-in and rated."""
    speeds = np.linspace(rgen.v_cut_in, rgen.v_rated, 20)
    powers = [rgen.wind_power(v) for v in speeds]
    for i in range(len(powers) - 1):
        assert powers[i] <= powers[i + 1] + 1e-9, (
            f"Wind not monotone at {speeds[i]:.1f}→{speeds[i+1]:.1f} m/s"
        )


def test_wind_negative_speed_returns_zero(rgen):
    assert rgen.wind_power(-5.0) == pytest.approx(0.0)


def test_wind_extreme_speed_returns_zero(rgen):
    assert rgen.wind_power(100.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 4. Solar PV model
# ---------------------------------------------------------------------------

def test_solar_zero_at_night(rgen):
    """GHI=0 (night) must produce zero solar power."""
    assert rgen.solar_power(0.0) == pytest.approx(0.0)


def test_solar_at_stc(rgen):
    """At standard test conditions (1000 W/m²), output = capacity × η_system."""
    expected = rgen.solar_capacity_mw * rgen.eta_system
    assert rgen.solar_power(1000.0) == pytest.approx(expected, rel=1e-6)


def test_solar_bounded_above(rgen):
    """Solar output must never exceed capacity, even at extreme GHI."""
    assert rgen.solar_power(2000.0) <= rgen.solar_capacity_mw * 1.001


def test_solar_monotone_with_ghi(rgen):
    """Higher GHI → higher (or equal) solar output."""
    ghis = np.linspace(0.0, 1500.0, 30)
    powers = [rgen.solar_power(g) for g in ghis]
    for i in range(len(powers) - 1):
        assert powers[i] <= powers[i + 1] + 1e-9


def test_solar_negative_ghi_clamped(rgen):
    """Negative GHI must be treated as zero."""
    assert rgen.solar_power(-100.0) == pytest.approx(rgen.solar_power(0.0))


# ---------------------------------------------------------------------------
# 5. Tick looping
# ---------------------------------------------------------------------------

def test_tick_wraps_without_crash(rgen):
    """Ticks beyond the data length should wrap via modulo, not crash."""
    big_tick = rgen.len_wind * 3 + 7
    r = rgen.get_generation(big_tick)
    r_wrapped = rgen.get_generation(big_tick % rgen.len_wind)
    assert r["p_wind_mw"] == pytest.approx(r_wrapped["p_wind_mw"])


def test_tick_zero_and_horizon_same(rgen):
    """Tick 0 and tick horizon should produce identical results."""
    r0  = rgen.get_generation(0)
    r_h = rgen.get_generation(rgen.len_wind)
    assert r_h["p_wind_mw"] == pytest.approx(r0["p_wind_mw"])


# ---------------------------------------------------------------------------
# 6. Data loading
# ---------------------------------------------------------------------------

def test_wind_data_loaded(rgen):
    assert rgen.wind_speeds is not None
    assert rgen.len_wind > 0


def test_solar_data_loaded(rgen):
    assert rgen.ghi_values is not None
    assert rgen.len_solar > 0


def test_wind_speeds_non_negative(rgen):
    assert np.all(rgen.wind_speeds >= 0.0)


def test_ghi_non_negative(rgen):
    assert np.all(rgen.ghi_values >= 0.0)


# ---------------------------------------------------------------------------
# 7. Graceful degradation: missing data directory
# ---------------------------------------------------------------------------

def test_missing_dir_gracefully_returns_zero():
    """When data files are missing, generation should degrade to zero."""
    rgen_no_data = RenewableGen(
        renewable_dir="/nonexistent/path",
        wind_capacity_mw=100.0,
        solar_capacity_mw=75.0,
    )
    r = rgen_no_data.get_generation(0)
    assert r["p_wind_mw"]  == pytest.approx(0.0)
    assert r["p_solar_mw"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 8. Capacity parametrisation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("wind_cap, solar_cap", [
    (50.0,  25.0),
    (200.0, 150.0),
    (0.0,   100.0),   # wind-only site
    (100.0, 0.0),     # solar-only site
])
def test_capacity_respected(wind_cap, solar_cap):
    rg = RenewableGen(
        renewable_dir=RENEWABLE_DIR,
        wind_capacity_mw=wind_cap,
        solar_capacity_mw=solar_cap,
    )
    for tick in range(0, 50):
        r = rg.get_generation(tick)
        assert r["p_wind_mw"]  <= wind_cap  + 1e-6
        assert r["p_solar_mw"] <= solar_cap + 1e-6
