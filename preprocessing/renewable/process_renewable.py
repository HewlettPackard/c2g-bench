"""
Preprocess hourly renewable resource data into 5-minute tick resolution
matching the C2G-Bench simulation timestep.

Input:
    data/processed/renewable/wind_hourly.csv   (timestamp_utc, wind_speed_100m_ms)
    data/processed/renewable/solar_hourly.csv  (timestamp_utc, ghi_wm2)

Output:
    data/processed/renewable/wind_5min.csv     (tick, wind_speed_100m_ms)
    data/processed/renewable/solar_5min.csv    (tick, ghi_wm2)

Interpolation: linear between hourly values (12 × 5-min ticks per hour).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

PROCESSED_ROOT = "data/processed/renewable"
TICKS_PER_HOUR = 12  # 60 min / 5 min


def interpolate_hourly_to_5min(hourly_values: np.ndarray) -> np.ndarray:
    """Linearly interpolate hourly data to 5-minute resolution."""
    n_hours = len(hourly_values)
    n_ticks = n_hours * TICKS_PER_HOUR
    hourly_indices = np.arange(n_hours) * TICKS_PER_HOUR
    tick_indices = np.arange(n_ticks)
    return np.interp(tick_indices, hourly_indices, hourly_values)


def process_wind(src_dir: str = PROCESSED_ROOT) -> str:
    """Interpolate hourly wind speed to 5-minute ticks."""
    df = pd.read_csv(os.path.join(src_dir, "wind_hourly.csv"))
    values = df["wind_speed_100m_ms"].to_numpy(dtype=float)
    ticks = interpolate_hourly_to_5min(values)

    out = pd.DataFrame({"tick": np.arange(len(ticks)), "wind_speed_100m_ms": ticks})
    path = os.path.join(src_dir, "wind_5min.csv")
    out.to_csv(path, index=False)
    print(f"Wind  5-min: {len(ticks)} ticks → {path}")
    return path


def process_solar(src_dir: str = PROCESSED_ROOT) -> str:
    """Interpolate hourly GHI to 5-minute ticks, floored at 0."""
    df = pd.read_csv(os.path.join(src_dir, "solar_hourly.csv"))
    values = df["ghi_wm2"].to_numpy(dtype=float)
    ticks = interpolate_hourly_to_5min(values)
    ticks = np.maximum(ticks, 0.0)  # GHI cannot be negative

    out = pd.DataFrame({"tick": np.arange(len(ticks)), "ghi_wm2": ticks})
    path = os.path.join(src_dir, "solar_5min.csv")
    out.to_csv(path, index=False)
    print(f"Solar 5-min: {len(ticks)} ticks → {path}")
    return path


def main():
    process_wind()
    process_solar()
    print("Done!")


if __name__ == "__main__":
    main()
