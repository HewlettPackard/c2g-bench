"""Merge and summarize the full cross-scenario thermal sensitivity study."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = REPO_ROOT / "copilot" / "artifacts"
SCENARIOS = ("default", "scenario_a", "scenario_b", "scenario_c")
CONTROLLERS = ("bang_bang", "pid", "rule_based", "sac")
SEEDS = (100, 101, 102, 103, 104)
FAULT_COLUMNS = (
    "thermal_fault",
    "freq_fault",
    "voltage_fault",
    "soc_fault",
    "sla_fault",
)


def load_partitions(artifact_dir: Path) -> pd.DataFrame:
    paths = {
        scenario: artifact_dir / f"thermal_sensitivity_cross_scenario_{suffix}.csv"
        for scenario, suffix in {
            "default": "default",
            "scenario_a": "a",
            "scenario_b": "b",
            "scenario_c": "c",
        }.items()
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing scenario partitions: {missing}")

    frames = []
    for expected_scenario, path in paths.items():
        frame = pd.read_csv(path)
        observed = set(frame["scenario"].unique())
        if observed != {expected_scenario}:
            raise ValueError(f"{path.name} contains scenarios {sorted(observed)}")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def validate_matrix(frame: pd.DataFrame) -> None:
    key = ["config_id", "scenario", "seed", "hardware_controller"]
    duplicated = frame.duplicated(key, keep=False)
    if duplicated.any():
        raise ValueError(f"Found {int(duplicated.sum())} rows with duplicate episode keys")
    if not frame["completed"].all():
        raise ValueError(f"Found {int((~frame['completed']).sum())} incomplete episodes")

    configs = tuple(sorted(frame["config_id"].unique()))
    expected = len(configs) * len(SCENARIOS) * len(SEEDS) * len(CONTROLLERS)
    if len(frame) != expected:
        raise ValueError(f"Expected {expected} rows, found {len(frame)}")
    if set(frame["scenario"]) != set(SCENARIOS):
        raise ValueError("Scenario set does not match the configured factorial design")
    if set(frame["seed"]) != set(SEEDS):
        raise ValueError("Seed set does not match the configured factorial design")
    if set(frame["hardware_controller"]) != set(CONTROLLERS):
        raise ValueError("Controller set does not match the configured factorial design")


def normalize_termination_reasons(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    for column in FAULT_COLUMNS:
        normalized.loc[normalized[column].astype(bool), "termination_reason"] = (
            column.removesuffix("_fault")
        )
    return normalized


def summarize_by_seed(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    working["survived"] = ~working["terminated"]
    working["warning_episode"] = working["thermal_warning_steps"] > 0
    group_columns = ["scenario", "hardware_controller", "seed"]
    grouped = working.groupby(group_columns, sort=False)

    seed_summary = grouped.agg(
        episodes=("config_id", "size"),
        survived_episodes=("survived", "sum"),
        survival_rate=("survived", "mean"),
        warning_episodes=("warning_episode", "sum"),
        warning_episode_rate=("warning_episode", "mean"),
        fast_steps=("fast_steps", "sum"),
        warning_steps=("thermal_warning_steps", "sum"),
        temp_max_mean=("temp_max", "mean"),
        temp_max_worst=("temp_max", "max"),
        return_mean=("return", "mean"),
        **{f"{column}_count": (column, "sum") for column in FAULT_COLUMNS},
    ).reset_index()
    seed_summary["warning_step_fraction"] = (
        seed_summary["warning_steps"] / seed_summary["fast_steps"]
    )
    return seed_summary


def summarize_scenarios(seed_summary: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "survival_rate",
        "warning_episode_rate",
        "warning_step_fraction",
        "temp_max_mean",
        "temp_max_worst",
        "return_mean",
    ]
    grouped = seed_summary.groupby(["scenario", "hardware_controller"], sort=False)
    summary = grouped[metric_columns].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()

    totals = grouped.agg(
        episodes=("episodes", "sum"),
        survived_episodes=("survived_episodes", "sum"),
        warning_episodes=("warning_episodes", "sum"),
        warning_steps=("warning_steps", "sum"),
        fast_steps=("fast_steps", "sum"),
        **{f"{column}_total": (f"{column}_count", "sum") for column in FAULT_COLUMNS},
    ).reset_index()
    return totals.merge(summary, on=["scenario", "hardware_controller"])


def summarize_configurations(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    working["survived"] = ~working["terminated"]
    working["warning_episode"] = working["thermal_warning_steps"] > 0
    group_columns = ["config_id", "scenario", "hardware_controller"]
    return working.groupby(group_columns, sort=False).agg(
        seeds=("seed", "nunique"),
        survival_rate=("survived", "mean"),
        warning_episode_rate=("warning_episode", "mean"),
        warning_step_fraction_mean=("thermal_warning_fraction", "mean"),
        warning_step_fraction_std=("thermal_warning_fraction", "std"),
        temp_max_mean=("temp_max", "mean"),
        temp_max_std=("temp_max", "std"),
        temp_max_worst=("temp_max", "max"),
        return_mean=("return", "mean"),
        return_std=("return", "std"),
        **{f"{column}_total": (column, "sum") for column in FAULT_COLUMNS},
    ).reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    args = parser.parse_args()

    frame = load_partitions(args.artifact_dir)
    validate_matrix(frame)
    frame = normalize_termination_reasons(frame)
    seed_summary = summarize_by_seed(frame)
    scenario_summary = summarize_scenarios(seed_summary)
    config_summary = summarize_configurations(frame)

    outputs = {
        "thermal_sensitivity_cross_scenario.csv": frame,
        "thermal_sensitivity_cross_scenario_seed_summary.csv": seed_summary,
        "thermal_sensitivity_cross_scenario_summary.csv": scenario_summary,
        "thermal_sensitivity_cross_scenario_config_summary.csv": config_summary,
    }
    for name, output in outputs.items():
        path = args.artifact_dir / name
        output.to_csv(path, index=False)
        print(f"Wrote {len(output):,} rows to {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()