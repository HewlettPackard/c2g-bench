from __future__ import annotations

import itertools
import subprocess
import sys
from pathlib import Path

import pytest


PYTHON = sys.executable
_ACTIONS = (
    "throttle_batch",
    "pump_speed_A",
    "hvac_effort",
    "bess_dispatch",
)


def _all_action_subsets() -> list[tuple[str, ...]]:
    subsets: list[tuple[str, ...]] = []
    for size in range(1, len(_ACTIONS) + 1):
        subsets.extend(itertools.combinations(_ACTIONS, size))
    return subsets


def _run_benchmark_with_ablation(disabled_actions: tuple[str, ...], output_path: Path) -> subprocess.CompletedProcess:
    cmd = [
        PYTHON,
        "evaluation/run_benchmark.py",
        "--agents",
        "random",
        "--scenarios",
        "default",
        "--n_episodes",
        "1",
        "--seed",
        "123",
        "--output",
        str(output_path),
        "--no-record_transitions",
        "--disable-actions",
        *disabled_actions,
    ]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=180,
    )


def _run_cmd(args: list[str], timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        [PYTHON] + args,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


@pytest.mark.smoke
class TestActionAblationSmoke:
    @pytest.mark.parametrize(
        "disabled_actions",
        _all_action_subsets(),
        ids=lambda actions: "disable_" + "_".join(actions),
    )
    def test_run_benchmark_all_15_disable_action_combinations(self, disabled_actions: tuple[str, ...], tmp_path: Path):
        output_path = tmp_path / f"ablation_{'_'.join(disabled_actions)}.csv"
        result = _run_benchmark_with_ablation(disabled_actions, output_path)

        assert result.returncode == 0, (
            f"run_benchmark.py failed for disabled actions {disabled_actions} with code {result.returncode}.\n"
            f"--- STDOUT ---\n{result.stdout}\n"
            f"--- STDERR ---\n{result.stderr}"
        )
        assert output_path.exists(), (
            f"Expected output CSV was not created for disabled actions {disabled_actions}: {output_path}"
        )


class TestAblationCliValidation:
    def test_run_benchmark_rejects_invalid_fixed_action_value(self):
        result = _run_cmd(
            [
                "evaluation/run_benchmark.py",
                "--agents", "random",
                "--scenarios", "default",
                "--n_episodes", "1",
                "--no-record_transitions",
                "--disable-actions", "hvac_effort",
                "--fixed-action", "hvac_effort=1.5",
            ]
        )
        assert result.returncode != 0
        assert "Invalid fixed value for action 'hvac_effort'" in (result.stderr + result.stdout)

    def test_run_benchmark_rejects_empty_disable_actions_flag(self):
        result = _run_cmd(
            [
                "evaluation/run_benchmark.py",
                "--agents", "random",
                "--scenarios", "default",
                "--n_episodes", "1",
                "--no-record_transitions",
                "--disable-actions",
            ]
        )
        assert result.returncode != 0
        assert "no actions were listed" in (result.stderr + result.stdout)

    def test_run_ha_benchmark_rejects_invalid_fixed_action_value(self):
        result = _run_cmd(
            [
                "evaluation/run_ha_benchmark.py",
                "--agents", "random",
                "--scenarios", "default",
                "--n_episodes", "1",
                "--no-record_transitions",
                "--disable-actions", "bess_dispatch",
                "--fixed-action", "bess_dispatch=-1.2",
            ]
        )
        assert result.returncode != 0
        assert "Invalid fixed value for action 'bess_dispatch'" in (result.stderr + result.stdout)

    def test_run_ha_benchmark_rejects_empty_disable_actions_flag(self):
        result = _run_cmd(
            [
                "evaluation/run_ha_benchmark.py",
                "--agents", "random",
                "--scenarios", "default",
                "--n_episodes", "1",
                "--no-record_transitions",
                "--disable-actions",
            ]
        )
        assert result.returncode != 0
        assert "no actions were listed" in (result.stderr + result.stdout)
