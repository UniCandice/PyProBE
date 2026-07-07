import polars as pl

from pyprobe.cyclers import maccor_cpi


def test_capacity_from_mode_column_uses_charge_and_discharge_segments() -> None:
    dataframe = pl.DataFrame(
        {
            "MD": ["R", "C", "C", "D", "D", "R"],
            "Capacity": [0.0, 1.0, 2.0, 2.5, 3.0, 3.0],
        }
    )

    importer = maccor_cpi.MaccorCPICapacity("Capacity", "MD")
    importer.column_map = {
        "MD": {"Cycler name": "MD", "Cycler unit": ""},
        "Capacity": {"Cycler name": "Capacity", "Cycler unit": ""},
    }

    result = dataframe.select(importer.expr).to_series().to_list()

    assert result == [0.0, 1.0, 2.0, 1.5, 1.0, 1.0]


def test_normalize_header_accepts_ahr_capacity_name() -> None:
    assert maccor_cpi.normalize_header("Capacity (AHr)") == "Capacity"
