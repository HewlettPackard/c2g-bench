# c2g_env/simulators/renewable.py
#
# On-site renewable generation models for a hyperscale datacenter campus.
#
# Wind model — IEC 61400 sigmoid power curve approximation:
#   P_wind(v) = capacity_mw × sigmoid((v - v_mid) / k)         v_cut_in < v < v_cut_out
#   P_wind    = 0                                                otherwise
#   where v_mid ≈ (v_rated + v_cut_in) / 2, k controls steepness
#
# Solar model — simple capacity-factor approach:
#   P_solar = capacity_mw × (GHI / GHI_stc) × η_system
#   GHI_stc = 1000 W/m² (standard test conditions)
#   η_system ≈ 0.85 (inverter + wiring + soiling derating)

import os

import numpy as np
import pandas as pd


class RenewableGen:
    """
    On-site wind + solar generation for the C2G-Bench datacenter campus.

    Loads pre-processed 5-minute resource data and converts to power output
    using physics-based generation models.
    """

    def __init__(
        self,
        renewable_dir: str = "data/processed/renewable",
        wind_capacity_mw: float = 100.0,
        solar_capacity_mw: float = 75.0,
    ):
        self.wind_capacity_mw = wind_capacity_mw
        self.solar_capacity_mw = solar_capacity_mw

        # --- Wind turbine parameters (generic 3 MW onshore IEC Class II) ---
        self.v_cut_in = 3.0       # m/s — below this, blades don't spin
        self.v_rated = 12.0       # m/s — at and above, full rated power
        self.v_cut_out = 25.0     # m/s — safety shutdown
        # Sigmoid steepness — fits typical power curves well
        self._v_mid = (self.v_cut_in + self.v_rated) / 2.0
        self._v_k = 1.5

        # --- Solar PV parameters ---
        self.ghi_stc = 1000.0     # W/m² at standard test conditions
        self.eta_system = 0.85    # system derating (inverter, wiring, soiling)
        self.panel_efficiency = 0.21  # modern mono-PERC / TOPCon

        # --- Load resource time series ---
        self.wind_speeds = self._load_series(renewable_dir, "wind_5min.csv", "wind_speed_100m_ms")
        self.ghi_values = self._load_series(renewable_dir, "solar_5min.csv", "ghi_wm2")

        self.len_wind = len(self.wind_speeds) if self.wind_speeds is not None else 0
        self.len_solar = len(self.ghi_values) if self.ghi_values is not None else 0

    @staticmethod
    def _load_series(directory: str, filename: str, column: str) -> np.ndarray | None:
        path = os.path.join(directory, filename)
        if not os.path.exists(path):
            return None
        try:
            df = pd.read_csv(path, usecols=[column])
            arr = pd.to_numeric(df[column], errors="coerce").dropna().to_numpy(dtype=float)
            return arr if arr.size > 0 else None
        except Exception:  # noqa: BLE001
            return None

    # ------------------------------------------------------------------ #
    #  Wind power curve                                                    #
    # ------------------------------------------------------------------ #
    def wind_power(self, wind_speed_ms: float) -> float:
        """
        Compute wind farm output (MW) for a given hub-height wind speed.

        Uses a sigmoid approximation of the IEC turbine power curve:
        smooth ramp from cut-in to rated, flat at rated, hard cut-out.
        """
        v = float(wind_speed_ms)
        if v < self.v_cut_in or v >= self.v_cut_out:
            return 0.0

        # Sigmoid: 0 near cut-in, ~1 near rated
        x = (v - self._v_mid) / self._v_k
        cf = 1.0 / (1.0 + np.exp(-x))
        return self.wind_capacity_mw * float(cf)

    # ------------------------------------------------------------------ #
    #  Solar PV model                                                      #
    # ------------------------------------------------------------------ #
    def solar_power(self, ghi_wm2: float) -> float:
        """
        Compute solar farm output (MW) for a given GHI irradiance.

        P = capacity × (GHI / GHI_stc) × η_system, clamped to [0, capacity].
        """
        ghi = max(0.0, float(ghi_wm2))
        cf = (ghi / self.ghi_stc) * self.eta_system
        return self.solar_capacity_mw * min(cf, 1.0)

    # ------------------------------------------------------------------ #
    #  Step interface (tick-aligned)                                        #
    # ------------------------------------------------------------------ #
    def get_generation(self, tick: int) -> dict:
        """
        Return renewable generation at a given simulation tick.

        Returns
        -------
        dict with keys:
            p_wind_mw        : wind farm output (MW)
            p_solar_mw       : solar farm output (MW)
            p_renewable_mw   : total renewable (MW)
            wind_speed_ms    : raw wind speed at tick
            ghi_wm2          : raw GHI at tick
        """
        # Wind
        if self.wind_speeds is not None and self.len_wind > 0:
            ws = float(self.wind_speeds[tick % self.len_wind])
        else:
            ws = 0.0
        p_wind = self.wind_power(ws)

        # Solar
        if self.ghi_values is not None and self.len_solar > 0:
            ghi = float(self.ghi_values[tick % self.len_solar])
        else:
            ghi = 0.0
        p_solar = self.solar_power(ghi)

        return {
            "p_wind_mw": p_wind,
            "p_solar_mw": p_solar,
            "p_renewable_mw": p_wind + p_solar,
            "wind_speed_ms": ws,
            "ghi_wm2": ghi,
        }
