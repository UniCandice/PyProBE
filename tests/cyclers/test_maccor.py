"""Tests for the Maccor cycler class."""

from datetime import datetime

import pandas as pd
import polars as pl
import pytest

from pyprobe.cyclers.maccor import Maccor
from pyprobe.cyclers.maccor_cpi import MaccorCPI

from .test_basecycler import helper_read_and_process


def test_read_and_process_maccor(benchmark):
    """Test reading and processing a sample Maccor file."""
    maccor_cycler = Maccor(
        input_data_path="tests/sample_data/maccor/sample_data_maccor.csv",
    )
    last_row = pl.DataFrame(
        {
            "Date": datetime(2023, 11, 23, 15, 56, 24, 60000),
            "Time [s]": [13.06],
            "Step": [2],
            "Event": [1],
            "Current [A]": [28.798],
            "Voltage [V]": [3.716],
            "Capacity [Ah]": [0.048],
            "Temperature [C]": [22.2591],
        },
    )
    helper_read_and_process(
        benchmark,
        maccor_cycler,
        expected_final_row=last_row,
        expected_events={0, 1},
    )


@pytest.mark.parametrize("suffix", ["csv", "xlsx"])
def test_read_and_process_maccor_cpi(tmp_path, suffix):
    """Test reading and processing a CPI-style Maccor export."""
    export_path = tmp_path / f"sample_maccor_cpi.{suffix}"
    sample_rows = [
        ["metadata", "", ""],
        ["CPI export", "", ""],
        ["Rec", "Step", "Test Time (sec)", "DPT Time", "MD", "Current (A)", "Voltage (V)", "EVTemp (C)"],
        [1, 1, "0d 00:01:00", "27/03/2026 10:04", "C", 0.1, 3.2, 25],
        [2, 1, "0d 00:02:00", "27/03/2026 10:05", "D", 0.2, 3.3, 24],
    ]

    if suffix == "csv":
        export_path.write_text(
            "\n".join(
                ",".join(str(cell) for cell in row) for row in sample_rows
            ),
            encoding="utf-8",
        )
    else:
        pd.DataFrame(sample_rows).to_excel(export_path, index=False, header=False)

    cycler = MaccorCPI(input_data_path=str(export_path))
    result = cycler.get_pyprobe_dataframe()

    assert result["Time [s]"].to_list() == [60.0, 120.0]
    assert result["Current [A]"].to_list() == [0.1, -0.2]
    assert result["Voltage [V]"].to_list() == [3.2, 3.3]
    assert result["Temperature [C]"].to_list() == [25.0, 24.0]
    assert result["Date"].dt.strftime("%Y-%m-%d %H:%M:%S").to_list() == [
        "2026-03-27 10:04:00",
        "2026-03-27 10:05:00",
    ]
