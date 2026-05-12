from __future__ import annotations
import csv, json, os
from pathlib import Path
import numpy as np
import pytest

from c2g_env import C2GFastEnv
from c2g_env.env_high_level import C2GMacroEnv

"""
tests/test_datalogging.py  —  Sanity checks for logged state-action-reward data from benchmark runs
"""

# Derive expected dimensions from the environment itself
_EXPECTED_OBS_DIM = C2GFastEnv(scenario="default").observation_space.shape[0]
_EXPECTED_ACT_DIM = C2GFastEnv(scenario="default").action_space.shape[0]

_EXPECTED_OBS_DIM_MACRO = C2GMacroEnv(scenario="default").observation_space.shape[0]
_EXPECTED_ACT_DIM_MACRO = C2GMacroEnv(scenario="default").action_space.shape[0]

class TestDataLogging:
    def test_eval_log_schema_and_continuity(self):
        # sanity check: previous state = next observation in log
        runs_dir = Path("runs")
        if not runs_dir.exists() or not runs_dir.is_dir():
            return

        subfolders = [p for p in runs_dir.iterdir() if p.is_dir()]
        non_empty_subfolders = [p for p in subfolders if any(p.iterdir())]

        for folder in non_empty_subfolders:
            for csv_path in folder.rglob("episode*.csv"):
                with open(csv_path, newline="") as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)

                fieldnames = reader.fieldnames or []
                
                # Extract state columns with semantic names (s_temp_A_norm, s_bess_soc, etc.)
                s_cols = sorted([name for name in fieldnames if name.startswith("s_")])

                # For every consecutive pair (prev_row, curr_row) from index 1 to
                # the last row: curr_row state must equal prev_row observation.
                for row_idx, (prev_row, curr_row) in enumerate(zip(rows, rows[1:]), start=1):
                    for s_col in s_cols:
                        o_col = s_col.replace("s_", "o_", 1)
                        
                        if o_col not in fieldnames:
                            raise Exception(f"Expected observation column {o_col} corresponding to state column {s_col} not found in {csv_path.name}")
                        
                        assert float(curr_row[s_col]) == pytest.approx(float(prev_row[o_col]), abs=1e-6), (
                            f"Transition mismatch in {csv_path.name} at row {row_idx}, {s_col}: "
                            f"{s_col}={curr_row[s_col]} vs prev {o_col}={prev_row[o_col]}"
                        )

    def test_datalogging_column_schema_hardware(self):
        """
        Verify that every non-macro CSV file in runs/ has the correct column schema:
        - {obs_dim} o_* columns (observations)
        - {obs_dim} s_* columns (states)
        - {act_dim} a_* columns (actions)
        - 7 r_* columns (reward components)
        - 1 r column (total reward)
        """
        runs_dir = "runs"
        
        # Exit if runs/ doesn't exist
        if not os.path.exists(runs_dir) or not os.path.isdir(runs_dir):
            pytest.skip("runs/ directory does not exist")
        
        # Recursively walk through runs/ and collect all CSV files
        all_csv_files = []
        for root, dirs, files in os.walk(runs_dir):
            for file in files:
                full_path = os.path.join(root, file)
                folder_name = os.path.basename(os.path.dirname(full_path)).lower()
                if file.endswith(".csv") and not folder_name.endswith("_macro"):
                    all_csv_files.append(full_path)
        
        # Exit if no CSV files found
        if not all_csv_files:
            pytest.skip("No CSV files found in runs/")
        
        obs_dim = _EXPECTED_OBS_DIM
        act_dim = _EXPECTED_ACT_DIM
        errors = []
        
        for csv_path in all_csv_files:
            with open(csv_path, newline="") as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames or []
            
            # Count columns by prefix
            o_cols = [col for col in fieldnames if col.startswith("o_")]
            s_cols = [col for col in fieldnames if col.startswith("s_")]
            a_cols = [col for col in fieldnames if col.startswith("a_")]
            r_component_cols = [col for col in fieldnames if col.startswith("r_")]
            r_total_cols = [col for col in fieldnames if col == "r"]
            
            file_errors = []
            
            if len(o_cols) != obs_dim:
                file_errors.append(f"o_* columns: expected {obs_dim}, found {len(o_cols)}")
            if len(s_cols) != obs_dim:
                file_errors.append(f"s_* columns: expected {obs_dim}, found {len(s_cols)}")
            if len(a_cols) != act_dim:
                file_errors.append(f"a_* columns: expected {act_dim}, found {len(a_cols)}")
            if len(r_component_cols) != 7:
                file_errors.append(f"r_* columns: expected 7, found {len(r_component_cols)}")
            if len(r_total_cols) != 1:
                file_errors.append(f"r column: expected 1, found {len(r_total_cols)}")
            
            if file_errors:
                rel_path = os.path.relpath(csv_path, runs_dir)
                error_msg = f"{rel_path}: " + "; ".join(file_errors)
                errors.append(error_msg)
        
        if errors:
            error_report = "\n".join(errors)
            raise AssertionError(f"Column schema violations found:\n{error_report}")

    def test_datalogging_column_schema_macro(self):
        """
        Verify that every macro CSV file in runs/ has the correct column schema:
        - {obs_dim} o_* columns (observations)
        - {obs_dim} s_* columns (states)
        - {act_dim} a_* columns (actions)
        - 7 r_* columns (reward components)
        - 1 r column (total reward)
        """
        runs_dir = "runs"

        # Exit if runs/ doesn't exist
        if not os.path.exists(runs_dir) or not os.path.isdir(runs_dir):
            pytest.skip("runs/ directory does not exist")

        # Recursively walk through runs/ and collect only macro CSV files
        all_csv_files = []
        for root, dirs, files in os.walk(runs_dir):
            for file in files:
                full_path = os.path.join(root, file)
                folder_name = os.path.basename(os.path.dirname(full_path)).lower()
                if file.endswith(".csv") and folder_name.endswith("_macro"):
                    all_csv_files.append(full_path)

        # Exit if no macro CSV files found
        if not all_csv_files:
            pytest.skip("No macro CSV files found in runs/")

        obs_dim = _EXPECTED_OBS_DIM_MACRO
        act_dim = _EXPECTED_ACT_DIM_MACRO
        errors = []

        for csv_path in all_csv_files:
            with open(csv_path, newline="") as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames or []

            # Count columns by prefix
            o_cols = [col for col in fieldnames if col.startswith("o_")]
            s_cols = [col for col in fieldnames if col.startswith("s_")]
            a_cols = [col for col in fieldnames if col.startswith("a_")]
            r_component_cols = [col for col in fieldnames if col.startswith("r_")]
            r_total_cols = [col for col in fieldnames if col == "r"]

            file_errors = []

            if len(o_cols) != obs_dim:
                file_errors.append(f"o_* columns: expected {obs_dim}, found {len(o_cols)}")
            if len(s_cols) != obs_dim:
                file_errors.append(f"s_* columns: expected {obs_dim}, found {len(s_cols)}")
            if len(a_cols) != act_dim:
                file_errors.append(f"a_* columns: expected {act_dim}, found {len(a_cols)}")
            if len(r_component_cols) != 4:
                file_errors.append(f"r_* columns: expected 4, found {len(r_component_cols)}")
            if len(r_total_cols) != 1:
                file_errors.append(f"r column: expected 1, found {len(r_total_cols)}")

            if file_errors:
                rel_path = os.path.relpath(csv_path, runs_dir)
                error_msg = f"{rel_path}: " + "; ".join(file_errors)
                errors.append(error_msg)

        if errors:
            error_report = "\n".join(errors)
            raise AssertionError(f"Column schema violations found:\n{error_report}")

    def test_smoke_benchmark_rule_macro(self):
        """Smoke test: run_benchmark.py with rule_macro agent."""
        import subprocess
        result = subprocess.run(
            ["uv", "run", "python", "evaluation/run_benchmark.py",
             "--record_transitions", "--agents", "rule_macro", "--n_episodes", "1"],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Command failed with exit code {result.returncode}.\nStdout:\n{result.stdout}\nStderr:\n{result.stderr}"

    def test_smoke_benchmark_rule_based(self):
        """Smoke test: run_benchmark.py with rule_based agent."""
        import subprocess
        result = subprocess.run(
            ["uv", "run", "python", "evaluation/run_benchmark.py",
             "--record_transitions", "--agents", "rule_based", "--n_episodes", "1"],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Command failed with exit code {result.returncode}.\nStdout:\n{result.stdout}\nStderr:\n{result.stderr}"

    def test_smoke_benchmark_rule_based_with_bess_ablation(self):
        """Smoke test: run_benchmark.py with rule_based agent and BESS fixed to 0.5."""
        import subprocess
        result = subprocess.run(
            ["uv", "run", "python", "evaluation/run_benchmark.py",
             "--record_transitions", "--agents", "rule_based", "--fixed-action", "bess_dispatch=0.5", "--n_episodes", "1"],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Command failed with exit code {result.returncode}.\nStdout:\n{result.stdout}\nStderr:\n{result.stderr}"

    def test_smoke_ha_benchmark_rule_based(self):
        """Smoke test: run_ha_benchmark.py with rule_based agent."""
        import subprocess
        result = subprocess.run(
            ["uv", "run", "python", "evaluation/run_ha_benchmark.py",
             "--record_transitions", "--agents", "rule_based", "--n_episodes", "1"],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Command failed with exit code {result.returncode}.\nStdout:\n{result.stdout}\nStderr:\n{result.stderr}"

    def test_smoke_ha_benchmark_rule_based_with_bess_ablation(self):
        """Smoke test: run_ha_benchmark.py with rule_based agent and BESS fixed to 0.5."""
        import subprocess
        result = subprocess.run(
            ["uv", "run", "python", "evaluation/run_ha_benchmark.py",
             "--record_transitions", "--agents", "rule_based", "--fixed-action", "bess_dispatch=0.5", "--n_episodes", "1"],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Command failed with exit code {result.returncode}.\nStdout:\n{result.stdout}\nStderr:\n{result.stderr}"