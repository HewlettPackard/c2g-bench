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
                s_idxs = sorted(
                    int(name.split("_", 1)[1])
                    for name in fieldnames
                    if name.startswith("s_") and name.split("_", 1)[1].isdigit()
                )

                # For every consecutive pair (prev_row, curr_row) from index 1 to
                # the last row: curr_row state must equal prev_row observation.
                for row_idx, (prev_row, curr_row) in enumerate(zip(rows, rows[1:]), start=1):
                    for i in s_idxs:
                        s_key = f"s_{i}"
                        o_key = f"o_{i}"
                        if o_key not in fieldnames:
                            continue
                        assert float(curr_row[s_key]) == pytest.approx(float(prev_row[o_key]), abs=1e-6), (
                            f"Transition mismatch in {csv_path.name} at row {row_idx}, dim {i}: "
                            f"{s_key}={curr_row[s_key]} vs prev {o_key}={prev_row[o_key]}"
                        )