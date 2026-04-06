"""
Download hourly wind and solar resource data from the Open-Meteo Historical
Weather API for the C2G-Bench datacenter site.

Source:
    https://open-meteo.com/en/docs/historical-weather-api

Variables:
    wind_speed_100m  — 100 m hub-height wind speed (m/s)
    shortwave_radiation — Global Horizontal Irradiance (W/m²)

Usage:
    python preprocessing/renewable/download_renewable.py
    python preprocessing/renewable/download_renewable.py --start-date 2023-01-01 --end-date 2023-12-31

Outputs:
    data/processed/renewable/wind_hourly.csv
    data/processed/renewable/solar_hourly.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime

# --- Site configuration ---------------------------------------------------
# Default: North-Central NY near NYISO NORTH zone (Massena area, good wind).
# Adjust lat/lon if the datacenter site changes.
SITE_LAT = 44.93
SITE_LON = -74.89
SITE_NAME = "NYISO_NORTH"

API_BASE = "https://archive-api.open-meteo.com/v1/archive"
RAW_ROOT = "data/raw/renewable"
PROCESSED_ROOT = "data/processed/renewable"

# Open-Meteo limits ~10 000 hourly rows per request ≈ 416 days.
# For a full year we stay well within this limit.
MAX_DAYS_PER_REQUEST = 365


def _fetch_json(url: str) -> dict:
    """GET a URL and parse JSON response."""
    req = urllib.request.Request(url, headers={"User-Agent": "C2G-Bench/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def download_renewable(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    out_dir: str = PROCESSED_ROOT,
    raw_dir: str = RAW_ROOT,
) -> tuple[str, str]:
    """
    Download hourly wind speed and solar irradiance from Open-Meteo.

    Returns (wind_csv_path, solar_csv_path).
    """
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    url = (
        f"{API_BASE}?"
        f"latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&hourly=wind_speed_100m,shortwave_radiation"
        f"&timezone=UTC"
    )

    print(f"Fetching renewable data for {SITE_NAME} ({lat}, {lon})")
    print(f"  Period: {start_date} → {end_date}")
    print(f"  URL: {url[:120]}...")

    data = _fetch_json(url)

    hourly = data["hourly"]
    times = hourly["time"]
    wind_speeds = hourly["wind_speed_100m"]
    ghi_values = hourly["shortwave_radiation"]

    n = len(times)
    print(f"  Received {n} hourly records")

    # --- Save raw JSON for reproducibility ---
    raw_path = os.path.join(raw_dir, f"open_meteo_{start_date}_{end_date}.json")
    with open(raw_path, "w") as f:
        json.dump(data, f)
    print(f"  Raw JSON → {raw_path}")

    # --- Write wind CSV ---
    wind_path = os.path.join(out_dir, "wind_hourly.csv")
    n_wind_valid = 0
    with open(wind_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_utc", "wind_speed_100m_ms"])
        for i in range(n):
            ts = times[i]
            ws = wind_speeds[i]
            if ws is None:
                continue
            writer.writerow([ts, ws])
            n_wind_valid += 1
    print(f"  Wind   → {wind_path}  ({n_wind_valid} valid rows)")

    # --- Write solar CSV ---
    solar_path = os.path.join(out_dir, "solar_hourly.csv")
    n_solar_valid = 0
    with open(solar_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_utc", "ghi_wm2"])
        for i in range(n):
            ts = times[i]
            ghi = ghi_values[i]
            if ghi is None:
                continue
            writer.writerow([ts, ghi])
            n_solar_valid += 1
    print(f"  Solar  → {solar_path}  ({n_solar_valid} valid rows)")

    return wind_path, solar_path


def main():
    parser = argparse.ArgumentParser(
        description="Download wind & solar resource data from Open-Meteo"
    )
    parser.add_argument(
        "--start-date", default="2023-01-01",
        help="Start date YYYY-MM-DD (default: 2023-01-01)",
    )
    parser.add_argument(
        "--end-date", default="2023-12-31",
        help="End date YYYY-MM-DD (default: 2023-12-31)",
    )
    parser.add_argument("--lat", type=float, default=SITE_LAT)
    parser.add_argument("--lon", type=float, default=SITE_LON)
    args = parser.parse_args()

    # Validate dates
    try:
        datetime.strptime(args.start_date, "%Y-%m-%d")
        datetime.strptime(args.end_date, "%Y-%m-%d")
    except ValueError:
        print("Error: dates must be YYYY-MM-DD format", file=sys.stderr)
        sys.exit(1)

    download_renewable(args.lat, args.lon, args.start_date, args.end_date)
    print("\nDone!")


if __name__ == "__main__":
    main()
