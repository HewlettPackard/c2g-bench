from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np
import pytest

"""
tests/test_datalogging.py  —  Sanity checks for logged state-action-reward data from benchmark runs
"""

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