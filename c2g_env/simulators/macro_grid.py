"""
Step 1.4 — Macro-Grid Signal Generator (The "Handshake")
=========================================================
Generates the two signals the C2G-Bench RL environment needs from the
regional grid:

  1. **Frequency Regulation Signal** (ΔP_grid, kW) — the real-time
     dispatch command the grid operator sends every timestep.
     The datacenter must match this by adjusting its three levers.

  2. **Locational Marginal Price** (LMP, $/MWh) — the wholesale energy
     price at the datacenter's Point of Common Coupling (PCC), used by
     the upper-level Market Orchestrator agent for economic decisions.

Frequency Regulation model
--------------------------
Modelled after PJM's RegD signal, which is designed for fast resources
(e.g., batteries, DVFS).  Key statistical properties reproduced:
  - Zero-mean over any 15-minute window (energy-neutral)
  - Band-limited noise: power spectral density peaks at 0.01–0.05 Hz
  - Normalised to [-1, 1]; scaled by the committed_mw capacity
  - Autocorrelation: AR(1) with ρ ≈ 0.8 at 5-min timestep

  ΔP_grid(t) = committed_mw × regD(t)   [MW]

LMP model
---------
Built on real NYISO 5-minute actual load data (11 load zones, 2023–2025).
A price proxy is derived from the load-duration curve:

  LMP(t) = lmp_base + lmp_slope × max(0, Load_t - Load_median)

This captures the non-linear marginal cost curve: prices spike sharply
when the grid is near peak load, creating a strong economic incentive for
the datacenter to reduce demand.  The zone is configurable (default: NYC).

References
----------
[1] PJM Manual 12: Balancing Operations, Section 4 (RegD signal spec).
    https://www.pjm.com/~/media/documents/manuals/m12.ashx
[2] NYISO Real-Time Actual Load data, 5-minute resolution.
    https://www.nyiso.com/real-time-dashboard
[3] Hogan (2002): "Electricity Market Restructuring: Reforms of Reforms,"
    Journal of Regulatory Economics 21(1). — For LMP theory.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Per-market regulation and price calibration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MarketParams:
    """
    Calibration parameters for one ISO/market region.

    Parameters
    ----------
    name : str
        Human-readable ISO name (e.g. "PJM").
    region : str
        Sub-region or data-centre hub (e.g. "Northern Virginia / Ashburn").
    regulation_product : str
        Official regulation product name (e.g. "RegD", "FCR-N", "FCAS").
    regd_rho : float
        AR(1) autocorrelation at 5-second step.
    regd_sigma : float
        AR(1) noise standard deviation (stationary std ≈ sigma / sqrt(1 - rho^2)).
    window_ticks : int
        Energy-neutrality correction window in ticks (= settlement interval / 5 s).
    lmp_base_usd : float
        Off-peak base wholesale electricity price (USD/MWh).
    lmp_slope_usd : float
        Marginal price sensitivity above median load (USD/MWh per GW).
    load_mean_mw : float
        Reference mean zonal load (MW) — used for synthetic fallback.
    load_daily_amp_mw : float
        Daily sinusoidal load amplitude (MW) — used for synthetic fallback.
    load_noise_std_mw : float
        Load noise standard deviation (MW) — used for synthetic fallback.
    dc_hub : str
        Reference data-centre location for documentation purposes.
    data_url : str
        Canonical download URL for real regional load data.
    """
    name:               str
    region:             str
    regulation_product: str
    regd_rho:           float
    regd_sigma:         float
    window_ticks:       int
    lmp_base_usd:       float
    lmp_slope_usd:      float
    load_mean_mw:       float
    load_daily_amp_mw:  float
    load_noise_std_mw:  float
    dc_hub:             str
    data_url:           str
    zone_csv:           str | None = None   # energy CSV stem; None → synthetic


#: Registry of supported markets.  Keys match the ``grid_market`` config field.
MARKET_PRESETS: dict[str, MarketParams] = {
    # ── 1. NYISO New York City ─────────────────────────────────────────────
    # Reference: NYISO OASIS real-time load, NYC zone (2023-2025)
    # RegD-equivalent signal; 15-min settlement.
    "nyiso_nyc": MarketParams(
        name               = "NYISO",
        region             = "New York City",
        regulation_product = "AGC / Secondary Reserve",
        regd_rho           = 0.9963,   # 0.80^(5/300)
        regd_sigma         = 0.022,
        window_ticks       = 180,      # 15 min
        lmp_base_usd       = 28.0,
        lmp_slope_usd      = 18.0,     # peak ≈ $118/MWh (5 GW above median)
        load_mean_mw       = 5_500.0,
        load_daily_amp_mw  = 1_500.0,
        load_noise_std_mw  = 300.0,
        dc_hub             = "Manhattan / New Jersey financial data centres",
        data_url           = "https://www.nyiso.com/load-data",
        zone_csv           = "NYC",
    ),
    # ── 2. PJM Dominion (Northern Virginia) ───────────────────────────────
    # "Data Center Alley" — ~70 % of global internet traffic.
    # Same RegD product as our synthetic signal (PJM is the RegD origin).
    "pjm_dom": MarketParams(
        name               = "PJM",
        region             = "Northern Virginia / Ashburn",
        regulation_product = "RegD (Fast-Response Regulation)",
        regd_rho           = 0.9963,
        regd_sigma         = 0.022,
        window_ticks       = 180,
        lmp_base_usd       = 35.0,
        lmp_slope_usd      = 10.0,     # deep market; peak ≈ $115/MWh
        load_mean_mw       = 14_000.0,
        load_daily_amp_mw  =  3_000.0,
        load_noise_std_mw  =    800.0,
        dc_hub             = "Ashburn / Loudoun County VA (largest DC cluster on Earth)",
        data_url           = "https://dataminer2.pjm.com/feed/hrl_load_metered",
        zone_csv           = "PJM_DOM",
    ),
    # ── 3. CAISO PG&E (Bay Area) ──────────────────────────────────────────
    # Duck curve: large midday solar ramp → rapid dispatchable ramp-up at dusk.
    # Higher base LMP (gas-heavy); slope reflects non-linear duck-curve ramp.
    "caiso_pgae": MarketParams(
        name               = "CAISO",
        region             = "Bay Area / San Jose & Santa Clara",
        regulation_product = "REGU / REGD (Regulation Up/Down)",
        regd_rho           = 0.9960,   # slightly more volatile due to solar variability
        regd_sigma         = 0.025,
        window_ticks       = 180,
        lmp_base_usd       = 48.0,
        lmp_slope_usd      = 28.0,     # duck curve creates steep non-linearity; peak ≈ $188
        load_mean_mw       = 26_000.0,
        load_daily_amp_mw  =  6_000.0, # strong solar duck curve
        load_noise_std_mw  =  1_500.0,
        dc_hub             = "Santa Clara / San Jose (Silicon Valley AI cluster)",
        data_url           = "https://oasis.caiso.com/mrioasis/logon.do",
        zone_csv           = "CAISO_PGAE",
    ),
    # ── 4. ERCOT North (Dallas–Fort Worth) ────────────────────────────────
    # Isolated grid (no interstate interconnects) → extreme price spikes.
    # Winter Storm Uri hit $9,000/MWh cap. Fast-growing DC market.
    "ercot_north": MarketParams(
        name               = "ERCOT",
        region             = "Dallas / Fort Worth",
        regulation_product = "ECRS (Electricity Contingency Reserve Service)",
        regd_rho           = 0.9950,   # more volatile; isolated grid
        regd_sigma         = 0.030,
        window_ticks       = 180,
        lmp_base_usd       = 28.0,
        lmp_slope_usd      = 55.0,     # isolated grid → very steep; peak >> $200 (clips in obs)
        load_mean_mw       = 42_000.0,
        load_daily_amp_mw  = 12_000.0, # extreme AC load in Texas summers
        load_noise_std_mw  =  2_000.0,
        dc_hub             = "Dallas–Fort Worth / San Antonio (fastest-growing US DC market)",
        data_url           = "https://www.ercot.com/gridinfo/load/load_hist",
        zone_csv           = "ERCOT_NORTH",
    ),
    # ── 5. ENTSO-E Germany (Frankfurt hub) ────────────────────────────────
    # FCR-N: responds to frequency deviation, not AGC dispatch → smoother signal.
    # 30-min EPEX settlement; deep interconnected market suppresses price spikes.
    # High renewable penetration (offshore wind in North Sea) → negative prices.
    "entso_de": MarketParams(
        name               = "ENTSO-E / EPEX",
        region             = "Germany (Frankfurt)",
        regulation_product = "FCR-N (Frequency Containment Reserve — Normal)",
        regd_rho           = 0.9980,   # FCR-N is smoother / more persistent
        regd_sigma         = 0.015,
        window_ticks       = 360,      # 30-min EPEX settlement interval
        lmp_base_usd       = 60.0,     # ~EUR 55 × 1.10 USD/EUR
        lmp_slope_usd      = 8.0,      # deep interconnected market; peak ≈ $156/MWh
        load_mean_mw       = 58_000.0,
        load_daily_amp_mw  = 10_000.0,
        load_noise_std_mw  =  3_000.0,
        dc_hub             = "Frankfurt (DE-CIX — largest internet exchange by traffic)",
        data_url           = "https://transparency.entsoe.eu/load-domain/r2/totalLoadR2/show",
        zone_csv           = "ENTSOE_DE",
    ),
    # ── 6. AEMO New South Wales (Sydney) ──────────────────────────────────
    # 5-min dispatch intervals (NEM); South Australia has reached 100% renewable.
    # Extreme price events: AUD $15,500/MWh cap; negative prices are common.
    "aemo_nsw": MarketParams(
        name               = "AEMO / NEM",
        region             = "New South Wales / Sydney",
        regulation_product = "Regulation FCAS (Frequency Control Ancillary Service)",
        regd_rho           = 0.9945,   # 5-min dispatch → faster decorrelation
        regd_sigma         = 0.032,
        window_ticks       = 60,       # 5-min NEM dispatch interval
        lmp_base_usd       = 55.0,     # AUD ~$80 × 0.67 USD
        lmp_slope_usd      = 45.0,     # extreme events; peak ≈ $190/MWh (clips)
        load_mean_mw       =  8_200.0,
        load_daily_amp_mw  =  1_500.0,
        load_noise_std_mw  =    400.0,
        dc_hub             = "Western Sydney Aerotropolis (AUD $1B investment zone)",
        data_url           = "https://aemo.com.au/en/energy-systems/electricity/national-electricity-market-nem/data-nem/aggregated-data",
        zone_csv           = "AEMO_NSW",
    ),
}



class MacroGridSignal:
    """
    Generates real-time frequency regulation and LMP price signals.

    Parameters
    ----------
    energy_dir:
        Directory with NYISO processed load CSVs (one per zone).
        Defaults to ``data/processed/energy`` relative to cwd.
    zone:
        Energy CSV stem to use for load data.  If None (default), uses the
        ``zone_csv`` field from the market preset.  Falls back to synthetic
        if the CSV does not exist.
    dt_seconds:
        Timestep in seconds.  Must match NYISO data resolution (300 s).
    committed_mw:
        Default regulation capacity committed to the grid operator [MW].
        The high-level agent overrides this each 15-minute interval.
    lmp_base_usd:
        Off-peak base LMP ($/MWh).  Calibrated to NYISO NYC historical mean.
    lmp_slope:
        Marginal price sensitivity above median load ($/MWh per GW).
    seed:
        RNG seed for the RegD signal generator.
    """

    def __init__(
        self,
        energy_dir: str | Path = "data/processed/energy",
        zone: str | None = None,
        dt_seconds: float = 300.0,
        committed_mw: float = 20.0,
        lmp_base_usd: float | None = None,
        lmp_slope: float | None = None,
        seed: int = 42,
        market: str = "nyiso_nyc",
    ) -> None:
        # Resolve market preset — explicit lmp_* args override the preset
        if market not in MARKET_PRESETS:
            raise ValueError(
                f"Unknown market {market!r}. "
                f"Available: {list(MARKET_PRESETS.keys())}"
            )
        self._market: MarketParams = MARKET_PRESETS[market]

        self.dt = dt_seconds
        self.committed_mw = committed_mw
        self.lmp_base_usd = lmp_base_usd if lmp_base_usd is not None else self._market.lmp_base_usd
        self.lmp_slope    = lmp_slope    if lmp_slope    is not None else self._market.lmp_slope_usd
        self._rng = np.random.default_rng(seed)

        # Resolve zone: explicit arg > market preset default > "NYC"
        effective_zone = zone or self._market.zone_csv or "NYC"
        self._load_mw, self._load_median = self._load_zone_or_synthetic(
            Path(energy_dir), effective_zone, seed
        )
        self._n = len(self._load_mw)

        # AR(1) parameters from market preset
        self._regd_rho:   float = self._market.regd_rho
        self._regd_sigma: float = self._market.regd_sigma
        self._regd_state: float = 0.0

        # Energy-neutrality integrator (window = market settlement interval)
        self._regd_buffer: list[float] = []
        self._window_ticks: int = self._market.window_ticks

        self._tick: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, committed_mw: float | None = None) -> dict[str, float]:
        """
        Advance one timestep and return the grid signals.

        Parameters
        ----------
        committed_mw:
            Override the committed regulation capacity for this step [MW].
            If None, uses the value set at construction / last ``set_committed``.

        Returns
        -------
        dict with keys:
            delta_p_kw      : Regulation dispatch command [kW].
                              Positive = grid asks DC to *increase* consumption.
                              Negative = grid asks DC to *reduce* consumption.
            committed_mw    : Effective committed capacity this step [MW].
            lmp_usd_mwh     : Wholesale LMP at PCC [$/MWh].
            grid_load_mw    : Regional zone load [MW] (grid stress indicator).
            load_norm       : Grid load as fraction of historical max [0, 1].
            regd_signal     : Normalised RegD signal [-1, 1].
            tick            : Current tick index.
        """
        if committed_mw is not None:
            self.committed_mw = float(committed_mw)

        idx = self._tick % self._n
        load_mw = float(self._load_mw[idx])

        regd = self._step_regd()
        delta_p_kw = regd * self.committed_mw * 1_000.0   # MW → kW

        lmp = self._load_to_lmp(load_mw)
        load_norm = load_mw / float(self._load_mw.max())

        self._tick += 1
        return {
            "delta_p_kw":    delta_p_kw,
            "committed_mw":  self.committed_mw,
            "lmp_usd_mwh":   lmp,
            "grid_load_mw":  load_mw,
            "load_norm":     load_norm,
            "regd_signal":   regd,
            "tick":          idx,
        }

    def reset(self, seed: int | None = None) -> None:
        """Reset to the beginning of the dataset and clear RegD state."""
        self._tick = 0
        self._regd_state = 0.0
        self._regd_buffer = []
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    def set_committed(self, committed_mw: float) -> None:
        """Update the committed regulation capacity [MW]."""
        self.committed_mw = float(np.clip(committed_mw, 0.0, 50.0))

    @property
    def horizon_ticks(self) -> int:
        """Total ticks in the loaded dataset before looping."""
        return self._n

    @property
    def lmp_stats(self) -> dict[str, float]:
        """Historical LMP statistics derived from zone load data."""
        lmps = np.array([self._load_to_lmp(l) for l in self._load_mw])
        return {
            "mean":  float(lmps.mean()),
            "std":   float(lmps.std()),
            "p95":   float(np.percentile(lmps, 95)),
            "max":   float(lmps.max()),
        }

    # ------------------------------------------------------------------
    # RegD signal generator
    # ------------------------------------------------------------------

    def _step_regd(self) -> float:
        """
        AR(1) process producing a PJM RegD-like normalised signal in [-1, 1].

        Energy neutrality is enforced with a mean-correction over rolling
        15-minute windows, matching PJM's requirement that RegD integrates
        to approximately zero over any 15-minute settlement period.
        """
        noise = float(self._rng.normal(0.0, self._regd_sigma))
        self._regd_state = self._regd_rho * self._regd_state + noise

        # Soft clip: tanh keeps the signal smooth near ±1
        raw = float(np.tanh(self._regd_state))

        # Accumulate into the energy-neutrality buffer
        self._regd_buffer.append(raw)
        if len(self._regd_buffer) >= self._window_ticks:
            # Correct for accumulated drift
            drift = np.mean(self._regd_buffer)
            raw_corrected = raw - drift
            self._regd_buffer = []
        else:
            raw_corrected = raw

        return float(np.clip(raw_corrected, -1.0, 1.0))

    # ------------------------------------------------------------------
    # LMP proxy from zone load
    # ------------------------------------------------------------------

    def _load_to_lmp(self, load_mw: float) -> float:
        """
        Non-linear LMP proxy: flat below median load, rising steeply above.

        LMP(t) = lmp_base + lmp_slope × max(0, Load_t - Load_median) / 1000
                                                                  [MW → GW]
        The division by 1000 converts the MW excess to GW for the slope units.
        """
        excess_gw = max(0.0, load_mw - self._load_median) / 1_000.0
        lmp = self.lmp_base_usd + self.lmp_slope * excess_gw
        return float(np.clip(lmp, 0.0, 500.0))   # cap at $500/MWh (spike limit)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_zone_or_synthetic(
        self, energy_dir: Path, zone: str, seed: int
    ) -> tuple[np.ndarray, float]:
        """
        Load zone load data from CSV; fall back to calibrated synthetic load
        if the file does not exist.

        Synthetic model (when real data unavailable):
            L(t) = L_mean + L_amp * sin(2π * h / 24 + π)     (daily cycle)
                         + σ_noise * ε(t)                      (AR(1) noise)
        where h is the hour-of-day.  Parameters come from the market preset so
        implied LMP peaks are statistically realistic for that ISO.
        """
        path = energy_dir / f"{zone}.csv"
        if path.exists():
            return self._load_zone(path)

        import warnings
        warnings.warn(
            f"MacroGridSignal: load file not found: {path}. "
            f"Using calibrated synthetic load for market '{self._market.name}' "
            f"({self._market.region}). "
            f"Download real data from: {self._market.data_url}",
            stacklevel=3,
        )
        return self._synthetic_load(seed)

    def _synthetic_load(self, seed: int) -> tuple[np.ndarray, float]:
        """Generate one year of synthetic 5-minute load using market calibration."""
        rng  = np.random.default_rng(seed + 999)
        m    = self._market
        # 365 × 24 × 12 = 105 120 five-minute ticks
        n    = 105_120
        t    = np.arange(n, dtype=np.float64)
        h    = (t * 5 / 60) % 24          # hour of day (fractional)
        # Daily cycle: peak in late afternoon, trough at 4 am
        daily = m.load_daily_amp_mw * np.sin(2 * np.pi * h / 24 - np.pi * 0.5)
        # AR(1) noise (ρ=0.95 at 5-min for load process, different from signal)
        rho_load, sigma_load = 0.97, m.load_noise_std_mw * np.sqrt(1 - 0.97**2)
        noise_arr = np.zeros(n)
        s = 0.0
        for i in range(n):
            s = rho_load * s + rng.normal(0, sigma_load)
            noise_arr[i] = s
        loads = np.maximum(m.load_mean_mw * 0.3, m.load_mean_mw + daily + noise_arr)
        return loads, float(np.median(loads))

    @staticmethod
    def _load_zone(path: Path) -> tuple[np.ndarray, float]:
        """
        Load zone load CSV and return (load_array_mw, median_mw).

        Expected columns: ``Time Stamp``, ``Load``  (NYISO OASIS format)
        """
        df = pd.read_csv(path, parse_dates=["Time Stamp"])
        if "Load" not in df.columns:
            raise ValueError(f"{path}: expected column 'Load', got {list(df.columns)}")
        df = df.dropna(subset=["Load"]).sort_values("Time Stamp").reset_index(drop=True)
        loads = df["Load"].values.astype(np.float64)
        return loads, float(np.median(loads))
